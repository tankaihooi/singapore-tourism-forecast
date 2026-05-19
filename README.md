# ✈️ Singapore Tourism Arrivals Forecast

<p align="center">
  <a href="https://huggingface.co/spaces/YOUR_USERNAME/sg-tourism-forecast">
    <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg" alt="Open in Spaces">
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" alt="Python 3.10">
  &nbsp;
  <img src="https://img.shields.io/badge/Models-SARIMA%20%7C%20Prophet%20%7C%20NeuralProphet-green" alt="Models">
  &nbsp;
  <img src="https://img.shields.io/badge/Uncertainty-Conformal%20Prediction-orange" alt="Conformal">
  &nbsp;
  <img src="https://img.shields.io/badge/Deployed-Streamlit-red?logo=streamlit&logoColor=white" alt="Streamlit">
</p>

<p align="center">
  Forecast monthly international arrivals to Singapore by source market, with calibrated uncertainty bands — deployed as an interactive dashboard.
</p>

---

## The Business Problem

Singapore's tourism sector contributed **S$27.2 billion** in receipts in 2023. Tourism boards, airport operators, and hospitality groups all need to answer the same question 6–18 months in advance:

> *How many visitors will arrive from market X — and how confident should we be in that number?*

Arrivals are driven by forces that interact non-linearly: seasonal travel cycles, economic conditions in source countries, visa policy changes, and structural shocks like SARS (2003), the GFC (2008–09), and COVID-19 (2020–21). A model that ignores these dynamics produces point forecasts that look precise but are systematically wrong.

This project builds a pipeline that **(1) quantifies forecast uncertainty** — not just point estimates — and **(2) evaluates models honestly** across multiple historical periods rather than a single held-out window.

---

## Key Findings

### Which model performs best?

Across **9 rolling backtest windows (2010–2018)**, SARIMA with conformal prediction intervals achieved the best median MAE and the most reliable coverage:

| Model | Median MAE | Coverage (95% PI) | Interval Width |
|:---|---:|:---:|:---:|
| **SARIMA + Conformal** | **Lowest** ✅ | ~95% | Moderate |
| Prophet + Native PI | Second | ~91% ⚠️ | Narrowest |
| NeuralProphet + Conformal | Third | ~94% | Widest |
| ARIMA + Conformal | Fourth | ~95% | Moderate |

> **Coverage is the more important metric.** A model achieving 91% coverage when claiming 95% is systematically overconfident — every planning decision made from its intervals is based on a confidence level that doesn't actually hold.

### What can a forecaster honestly say?

| Horizon | Reliability | Typical PI Width |
|:---|:---|:---|
| 3–6 months | Reasonably reliable for major markets | ±15–25% of point estimate |
| 12–18 months | Uncertainty approximately doubles | ±30–45% of point estimate |
| Post-COVID recovery | High miss risk — regime shift | Intervals should be treated as scenarios |

### Most important methodological finding

MAE rankings **flip across different backtest origins** — a model that is best in 2015 is sometimes worst in 2017. A single held-out window, the approach used in most academic projects, can give a misleading picture. **Median MAE across 9 windows** is the more honest basis for a model recommendation.

---

## About the Data

| | |
|:---|:---|
| **Source** | [SingStat Table M550001](https://www.singstat.gov.sg/) — International Visitor Arrivals by Country of Residence |
| **Coverage** | January 2000 – present · ~40 source markets · monthly granularity |
| **Key decisions** | COVID window (Feb 2020 – May 2021) modelled as an explicit structural break, not excluded |
| | Log-transformation applied to stabilise variance across markets |
| | Time-based splits only — random splitting is statistically invalid for time series |

---

## Project Structure

```
singapore-tourism-forecast/
├── data/
│   ├── arrivals_clean.csv              # Week 1: cleaned SingStat data
│   ├── val_predictions_week2.csv       # Week 2: Prophet + NeuralProphet validation
│   ├── val_predictions_week3.csv       # Week 3: all models + conformal intervals
│   ├── backtest_results.csv            # Week 4: full rolling backtest records
│   ├── model_comparison_table.csv      # Week 4: aggregate MAE / RMSE / MAPE / coverage
│   ├── test_set_predictions.csv        # Week 4: post-COVID test set results
│   └── dm_test_results.csv             # Week 4: Diebold-Mariano significance tests
├── notebooks/
│   ├── 01_data_and_baseline.ipynb      # Data engineering · ARIMA · SARIMA
│   ├── 02_prophet_neuralprophet.ipynb  # Prophet · NeuralProphet
│   ├── 03_uncertainty_quantification.ipynb  # Conformal prediction
│   └── 04_backtesting_framework.ipynb  # Rolling backtest · DM test · test set unlock
├── streamlit_app.py                    # Interactive dashboard (Week 5)
└── requirements.txt
```

---

## Technical Details

### Model stack

<details>
<summary><strong>ARIMA / SARIMA</strong> — <code>pmdarima.auto_arima</code></summary>
<br>
Order selected by AIC minimisation with stepwise search. Seasonal component m=12 for monthly data. Uncertainty intervals via split conformal prediction, calibrated on the trailing 24 months of residuals at each backtest origin.
</details>

<details>
<summary><strong>Prophet</strong> — Meta / Facebook</summary>
<br>
COVID-19 modelled as an explicit additive regressor rather than a changepoint — this prevents the model from over-attributing the structural break to trend drift. Native Bayesian 95% prediction intervals from posterior sampling.
</details>

<details>
<summary><strong>NeuralProphet</strong></summary>
<br>
AR component with <code>n_lags=12</code> (one full year of context). Uncertainty quantified via conformal prediction rather than native quantile regression, which requires a larger calibration set than is available here.
</details>

<details>
<summary><strong>Uncertainty quantification — split conformal prediction</strong></summary>
<br>

All models produce uncertainty bands using **split conformal prediction**, a distribution-free method that makes no assumptions about the error distribution:

> *If calibration data and test data are exchangeable, the prediction interval will contain the true value with at least (1−α) probability.*

This is stronger than a Gaussian ±1.96σ interval, which requires normality of residuals. The finite-sample corrected quantile `⌈(n+1)(1−α)/n⌉` is used throughout to maintain the coverage guarantee for small calibration sets (n=24).
</details>

<details>
<summary><strong>Backtesting methodology</strong></summary>
<br>

Expanding-window backtesting across 9 forecast origins (2010-01 → 2018-01), each with a 12-month horizon:

- Each model is **re-fit from scratch** at every origin on all data prior to that date
- Conformal intervals are **re-calibrated** on the trailing 24 months of residuals
- Statistical significance of MAE differences tested via the **Diebold-Mariano HLN test** (finite-sample corrected)
- Test set (≥ 2022) was locked until all modelling decisions were finalised
</details>

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/singapore-tourism-forecast
cd singapore-tourism-forecast
pip install -r requirements.txt

# 2. Download data
# → https://www.singstat.gov.sg/find-data/search-by-theme/industry/tourism/latest-data
# Download Table M550001 as CSV and save to data/raw_singstat.csv

# 3. Run notebooks in order
jupyter notebook notebooks/01_data_and_baseline.ipynb
jupyter notebook notebooks/02_prophet_neuralprophet.ipynb
jupyter notebook notebooks/03_uncertainty_quantification.ipynb
jupyter notebook notebooks/04_backtesting_framework.ipynb

# 4. Launch dashboard
streamlit run streamlit_app.py
```

---

## Limitations & Future Work

**What this model cannot do**
- Account for unannounced policy changes (visa exemptions, new airline routes)
- Predict structural breaks of COVID magnitude
- Generalise to markets with fewer than 5 years of pre-COVID data

**Meaningful extensions**
- Add exchange rates and Google Trends search volume as external regressors in Prophet
- Run the backtest pipeline across all 40 source markets — surface which are most forecastable
- Replace separate coverage + width reporting with the **Winkler score**, which penalises both simultaneously
- **Hierarchical reconciliation** — ensure country-level forecasts sum to the total-arrivals forecast

---

<p align="center">
  Data: <a href="https://www.singstat.gov.sg/">SingStat / Data.gov.sg</a> ·
  Built with Python, pmdarima, Prophet, NeuralProphet, Streamlit ·
  Deployed on <a href="https://huggingface.co/spaces/YOUR_USERNAME/sg-tourism-forecast">Hugging Face Spaces</a>
</p>
