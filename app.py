"""
Singapore Tourism Arrivals Forecaster
======================================
Streamlit dashboard

Usage (local):
    streamlit run app.py

Usage (Hugging Face Spaces):
    - Place this file as app.py in your Space root
    - Include requirements.txt alongside it
    - Place data/ folder (from Week 4 outputs) alongside it
"""

import warnings
warnings.filterwarnings("ignore")
import statsmodels
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date

try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SG Tourism Forecast",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .section-header { font-size: 15px; font-weight: 600; color: white; margin: 24px 0 8px; }
    .note-box {
        background: #fff8e1;
        border-left: 3px solid #f9a825;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 13px;
        color: #555;
        margin: 8px 0;
    }
    .insight-box {
        background: #e8f5e9;
        border-left: 3px solid #2e7d32;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 13px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path("data")

PALETTE = {
    "ARIMA":         "#2c7bb6",
    "SARIMA":        "#4575b4",
    "Prophet":       "#d7191c",
    "NeuralProphet": "#1a9641",
}

BAND_ALPHA = 0.18


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_arrivals() -> pd.DataFrame:
    path = DATA_DIR / "arrivals_clean.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])

    rng    = np.random.default_rng(42)
    dates  = pd.date_range("2000-01-01", "2023-12-01", freq="MS")
    countries = ["China", "Indonesia", "India", "Malaysia", "Australia",
                 "Japan", "South Korea", "United Kingdom", "United States", "Philippines"]
    rows = []
    for country in countries:
        base  = rng.integers(30_000, 200_000)
        trend = rng.uniform(0.002, 0.008)
        for i, d in enumerate(dates):
            seasonal = 1 + 0.25 * np.sin(2 * np.pi * (i % 12) / 12 + rng.uniform(0, 1))
            covid    = 0.02 if (d.year == 2020 or (d.year == 2021 and d.month < 6)) else 1.0
            noise    = rng.lognormal(0, 0.08)
            val      = int(base * (1 + trend * i) * seasonal * covid * noise)
            rows.append({"date": d, "country": country, "arrivals": max(val, 0)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_comparison_table() -> pd.DataFrame | None:
    path = DATA_DIR / "model_comparison_table.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_test_predictions() -> pd.DataFrame | None:
    path = DATA_DIR / "test_set_predictions.csv"
    return pd.read_csv(path, parse_dates=["date"]) if path.exists() else None


# ══════════════════════════════════════════════════════════════════════════════
# FORECASTING  — inputs are tuples so @st.cache_data can hash them
# ══════════════════════════════════════════════════════════════════════════════

def _series_to_tuples(s: pd.Series):
    """Convert a Series to two hashable tuples for cache-key purposes."""
    return tuple(s.values), tuple(s.index.astype(str))


def _tuples_to_series(values_t, index_t) -> pd.Series:
    return pd.Series(list(values_t), index=pd.DatetimeIndex(list(index_t)))


def conformal_interval(residuals, preds, alpha=0.05):
    n     = len(residuals)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = np.quantile(np.abs(residuals), level)
    return preds - q_hat, preds + q_hat


@st.cache_data(show_spinner=False)
def fit_sarima_forecast(values_t: tuple, index_t: tuple,
                        horizon: int, alpha: float = 0.05) -> dict:
    """SARIMA + conformal intervals. Cached by (country series, horizon, alpha)."""
    series_log = _tuples_to_series(values_t, index_t)

    if not HAS_PMDARIMA:
        return _dummy_forecast(values_t, horizon, "SARIMA", alpha)

    # ── Single auto_arima fit on the full series ───────────────────────────────
    # Calibration residuals come from a trailing holdout of the SAME model,
    # avoiding a second expensive auto_arima search.
    cal_len    = min(24, len(series_log) // 3)
    short      = len(series_log.dropna()) < 60

    if short:
        model = pm.ARIMA(order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
        model.fit(series_log.dropna())
    else:
        model = pm.auto_arima(
            series_log,
            seasonal=True, m=12,
            stepwise=True,
            max_p=3, max_q=3,
            max_P=2, max_Q=2,
            max_order=8,
            information_criterion="aic",
            error_action="ignore",
            suppress_warnings=True,
            maxiter=30,
        )

    # ── Calibration: in-sample residuals on trailing cal_len months ───────────
    in_sample  = model.predict_in_sample()
    residuals  = series_log.values[-cal_len:] - in_sample[-cal_len:]

    # ── Forecast ──────────────────────────────────────────────────────────────
    point_log          = model.predict(n_periods=horizon)
    lower_log, upper_log = conformal_interval(residuals, point_log, alpha)

    return {
        "point": np.expm1(point_log),
        "lower": np.expm1(lower_log),
        "upper": np.expm1(upper_log),
        "model": "SARIMA",
    }


@st.cache_data(show_spinner=False)
def fit_prophet_forecast(values_t: tuple, index_t: tuple,
                         horizon: int, alpha: float = 0.05) -> dict:
    """Prophet + native PI. Cached by (country series, horizon, alpha).
    Returns plain arrays only — no Prophet objects stored (avoids RAM leak)."""
    series = _tuples_to_series(values_t, index_t)

    if not HAS_PROPHET:
        return _dummy_forecast(values_t, horizon, "Prophet", alpha)

    df = pd.DataFrame({"ds": series.index, "y": np.log1p(series.values)})
    df["covid"] = ((df["ds"].dt.year == 2020) |
                   ((df["ds"].dt.year == 2021) & (df["ds"].dt.month < 6))).astype(int)

    m = Prophet(
        interval_width=1 - alpha,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
    )
    m.add_regressor("covid")
    m.fit(df)

    future          = m.make_future_dataframe(periods=horizon, freq="MS")
    future["covid"] = 0
    forecast        = m.predict(future)
    fcast_rows      = forecast.tail(horizon)

    # ── Extract arrays only — do NOT store m or forecast (RAM leak) ───────────
    return {
        "point": np.expm1(fcast_rows["yhat"].values),
        "lower": np.expm1(fcast_rows["yhat_lower"].values),
        "upper": np.expm1(fcast_rows["yhat_upper"].values),
        "model": "Prophet",
    }


@st.cache_data(show_spinner=False)
def _dummy_forecast(values_t: tuple, horizon: int,
                    model_name: str, alpha: float = 0.05) -> dict:
    vals  = np.log1p(list(values_t))
    preds = [vals[-(h + 12)] if (h + 12) < len(vals) else vals[-1] for h in range(horizon)]
    preds = np.array(preds)
    spread = 0.15 * np.abs(preds)
    return {
        "point": np.expm1(preds),
        "lower": np.expm1(preds - spread),
        "upper": np.expm1(preds + spread),
        "model": f"{model_name} (naïve fallback)",
    }


def forecast_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=horizon, freq="MS",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def make_forecast_chart(hist, fcast, horizon, country, show_history_months=36):
    fdates    = forecast_dates(hist.index[-1], horizon)
    color     = PALETTE.get(fcast["model"].split()[0], "#333")
    hist_tail = hist.iloc[-show_history_months:]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor("white")
    ax.plot(hist_tail.index, hist_tail.values / 1_000,
            color="#333", lw=1.8, label="Historical arrivals", zorder=3)
    ax.fill_between(fdates, fcast["lower"] / 1_000, fcast["upper"] / 1_000,
                    color=color, alpha=BAND_ALPHA, label="95% prediction interval")
    ax.plot(fdates, fcast["point"] / 1_000,
            color=color, lw=2.2, ls="--", label=f"Forecast ({fcast['model']})", zorder=4)
    ax.scatter(fdates, fcast["point"] / 1_000, color=color, s=30, zorder=5)
    ax.axvline(hist.index[-1], color="#aaa", lw=1, ls=":", zorder=2)
    ax.text(hist.index[-1], ax.get_ylim()[1] * 0.95, "  forecast →",
            fontsize=9, color="#888", va="top")
    ax.set_title(f"{country} — {horizon}-month arrival forecast", fontsize=13, pad=10)
    ax.set_ylabel("Monthly arrivals (thousands)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}K"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.6)
    fig.tight_layout()
    return fig


def make_interval_width_chart(fcast, horizon):
    widths = (fcast["upper"] - fcast["lower"]) / 1_000
    color  = PALETTE.get(fcast["model"].split()[0], "#555")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(horizon), widths, color=color, alpha=0.7)
    ax.set_xticks(range(horizon))
    ax.set_xticklabels([f"M+{i+1}" for i in range(horizon)], fontsize=8, rotation=45)
    ax.set_ylabel("Interval width (thousands)", fontsize=10)
    ax.set_title("Prediction interval width by horizon — wider = more uncertainty", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _find_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def make_comparison_bar(comp_df):
    model_col = _find_col(comp_df, ["Model", "model", "name"]) or comp_df.columns[0]
    mae_col   = _find_col(comp_df, ["MAE", "mae", "mean_ae", "mean_absolute_error",
                                    "BT MAE (median)", "BT MAE (mean)"])
    models = comp_df[model_col].tolist()
    fig, ax = plt.subplots(figsize=(7, 3))
    if mae_col:
        mae    = pd.to_numeric(comp_df[mae_col].astype(str).str.replace(",", "").str.strip(), errors="coerce").tolist()
        colors = [PALETTE.get(str(m).replace("★ ", ""), "#999") for m in models]
        vals   = [m / 1_000 if m > 100 else m for m in mae]
        bars   = ax.barh(models, vals, color=colors, alpha=0.82)
        labels = [f"{m/1_000:,.1f}K" if m > 100 else f"{m:.3f}" for m in mae]
        ax.bar_label(bars, labels=labels, padding=4, fontsize=9)
        ax.set_xlabel("Median MAE", fontsize=10)
    else:
        num_cols = comp_df.select_dtypes("number").columns
        if len(num_cols):
            vals = comp_df[num_cols[0]].tolist()
            bars = ax.barh(models, vals, color="#2c7bb6", alpha=0.82)
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], padding=4, fontsize=9)
            ax.set_xlabel(num_cols[0], fontsize=10)
    ax.set_title("Backtested model comparison — lower is better", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

df_all   = load_arrivals()
comp_df  = load_comparison_table()
test_df  = load_test_predictions()

with st.sidebar:
    st.markdown("## 🗺️ Singapore Tourism Forecast")
    st.markdown("---")

    countries = sorted(df_all["country"].unique())
    country   = st.selectbox("**Source market**", countries,
                             index=countries.index("China") if "China" in countries else 0)
    st.markdown("---")
    horizon      = st.slider("**Forecast horizon (months)**", 3, 24, 12, step=3)
    model_choice = st.radio("**Forecasting model**",
                            ["SARIMA + Conformal", "Prophet (native PI)", "Both"], index=0)
    alpha        = st.slider("**Significance level (α)**", 0.01, 0.20, 0.05, step=0.01,
                             help="Prediction interval = 1–α confidence. Default 0.05 → 95% PI.")
    show_hist    = st.slider("**History shown (months)**", 12, 60, 36, step=6)
    st.markdown("---")
    st.markdown("**Data source:** SingStat / Data.gov.sg")
    st.markdown("**Models:** SARIMA · Prophet")
    st.markdown("**Intervals:** Conformal prediction")
    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

st.title("✈️ Singapore Tourism Arrivals Forecaster")
st.markdown(
    "Forecast monthly international arrivals to Singapore by source market, "
    "with calibrated uncertainty bands from conformal prediction."
)

if not (DATA_DIR.exists() and (DATA_DIR / "arrivals_clean.csv").exists()):
    st.markdown(
        '<div class="note-box">⚠️ <strong>Demo mode:</strong> '
        'Synthetic data is being used. Run Weeks 1–4 notebooks to generate '
        'real arrival data, then place the <code>data/</code> folder here.</div>',
        unsafe_allow_html=True,
    )

# ── Filter series ──────────────────────────────────────────────────────────────
country_df = (
    df_all[df_all["country"] == country]
    .sort_values("date")
    .set_index("date")["arrivals"]
    .dropna()
)
last_date  = country_df.index[-1]
series_log = np.log1p(country_df)

# ── KPI row ───────────────────────────────────────────────────────────────────
pre_covid  = country_df[country_df.index.year.isin([2018, 2019])].sum()
recent_12m = country_df.iloc[-12:].sum()
peak_month = country_df.idxmax()
peak_val   = country_df.max()
recovery   = recent_12m / (pre_covid / 2) * 100 if pre_covid > 0 else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("12-month total (most recent)", f"{recent_12m/1e6:.2f}M")
with col2:
    st.metric("Pre-COVID annual avg (2018–19)", f"{pre_covid/2/1e6:.2f}M")
with col3:
    st.metric("Recovery vs pre-COVID", f"{recovery:.0f}%",
              delta=f"{recovery-100:.0f}pp vs baseline")
with col4:
    st.metric("Historical peak month", f"{peak_val/1e3:.0f}K",
              delta=peak_month.strftime("%b %Y"))

st.markdown("---")

# ── Convert to hashable tuples ONCE before calling cached functions ────────────
values_t, index_t     = _series_to_tuples(series_log)
values_orig_t, index_t = _series_to_tuples(country_df)

# ── Run forecasts ──────────────────────────────────────────────────────────────
run_sarima  = model_choice in ["SARIMA + Conformal", "Both"]
run_prophet = model_choice in ["Prophet (native PI)", "Both"]
fcasts = {}

with st.spinner(f"Fitting model(s) for {country}…"):
    if run_sarima:
        fcasts["SARIMA"]  = fit_sarima_forecast(values_t, index_t, horizon, alpha)
    if run_prophet:
        fcasts["Prophet"] = fit_prophet_forecast(values_orig_t, index_t, horizon, alpha)

# ── Forecast chart ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Forecast with uncertainty bands</div>',
            unsafe_allow_html=True)

if model_choice == "Both" and len(fcasts) == 2:
    col_l, col_r = st.columns(2)
    for col, (mname, fcast) in zip([col_l, col_r], fcasts.items()):
        with col:
            fig = make_forecast_chart(country_df, fcast, horizon, country, show_hist)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
else:
    fcast = next(iter(fcasts.values()))
    fig   = make_forecast_chart(country_df, fcast, horizon, country, show_hist)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Forecast table ────────────────────────────────────────────────────────────
with st.expander("📋 Forecast table (all horizons)", expanded=False):
    for mname, fcast in fcasts.items():
        fdates = forecast_dates(last_date, horizon)
        tbl = pd.DataFrame({
            "Month":         fdates.strftime("%b %Y"),
            "Forecast":      [f"{v:,.0f}" for v in fcast["point"]],
            "Lower (95%PI)": [f"{v:,.0f}" for v in fcast["lower"]],
            "Upper (95%PI)": [f"{v:,.0f}" for v in fcast["upper"]],
            "Width":         [f"{(u-l):,.0f}" for u, l in zip(fcast["upper"], fcast["lower"])],
        })
        st.markdown(f"**{mname}**")
        st.dataframe(tbl, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Interval width chart ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">🎯 Prediction uncertainty across the horizon</div>',
            unsafe_allow_html=True)

for mname, fcast in fcasts.items():
    fig = make_interval_width_chart(fcast, horizon)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.markdown(
    '<div class="note-box">Interval width increases with horizon — '
    'further-out predictions are inherently less certain. '
    'The conformal method guarantees that <strong>95%</strong> of actual values fall '
    'within these bands on held-out data (from backtesting).</div>',
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Model comparison table ────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Backtested model comparison (Week 4)</div>',
            unsafe_allow_html=True)

if comp_df is not None:
    col_tbl, col_bar = st.columns([1, 1])
    with col_tbl:
        numeric_cols = comp_df.select_dtypes("number").columns.tolist()
        styled = comp_df.style.highlight_min(subset=numeric_cols, color="#c8e6c9") \
                 if numeric_cols else comp_df.style
        st.dataframe(styled, use_container_width=True)
    with col_bar:
        fig = make_comparison_bar(comp_df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    st.markdown(
        '<div class="insight-box" style="color: black;">✅ Metrics above come from a rolling '
        '<strong>expanding-window backtest</strong> across 9 historical origins (2010–2018). '
        'Green = lowest error per metric. This is a more honest evaluation than a single '
        'held-out window.</div>',
        unsafe_allow_html=True,
    )
else:
    st.info("Model comparison table not found. Run the Week 4 notebook to generate "
            "`data/model_comparison_table.csv`.")

st.markdown("---")

# ── Test set ──────────────────────────────────────────────────────────────────
if test_df is not None:
    st.markdown('<div class="section-header">🔓 Test set: Post-COVID recovery (≥ 2022)</div>',
                unsafe_allow_html=True)

    avail_models = test_df["model"].unique().tolist()
    sel_models   = st.multiselect("Show models:", avail_models, default=avail_models[:2])

    if sel_models:
        fig, ax = plt.subplots(figsize=(12, 5))
        actuals = test_df[test_df["model"] == avail_models[0]].set_index("date")["actual"]
        ax.plot(actuals.index, actuals.values / 1_000,
                color="black", lw=2, label="Actual", zorder=5)
        for mname in sel_models:
            sub   = test_df[test_df["model"] == mname].set_index("date")
            color = PALETTE.get(mname, "#999")
            ax.fill_between(sub.index, sub["lower"] / 1_000, sub["upper"] / 1_000,
                            color=color, alpha=BAND_ALPHA)
            ax.plot(sub.index, sub["forecast"] / 1_000,
                    color=color, lw=1.8, ls="--", label=mname)
        ax.set_title("Test set: actual vs forecast (post-COVID recovery)", fontsize=13)
        ax.set_ylabel("Monthly arrivals (thousands)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown(
            '<div class="note-box">⚠️ Post-COVID recovery patterns (rapid 30–60% growth) '
            'differ structurally from pre-COVID cycles. Higher test-set MAPE vs backtest MAPE '
            'is expected — this is the models encountering a regime they have never seen.</div>',
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Singapore Tourism Arrivals Forecaster · "
    "Data: SingStat / Data.gov.sg · "
    "Models: SARIMA (pmdarima), Prophet (Meta) · "
    "Uncertainty: split conformal prediction · "
    "Built with Streamlit"
)
