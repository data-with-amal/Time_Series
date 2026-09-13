# 📈 Predictive Forecasting of Care Load & Placement Demand

Forecasts the number of children in HHS care using historical CBP custody and
placement data from the **HHS Unaccompanied Alien Children (UAC) Program**.
The project walks through six stages — data cleaning, quality checks, EDA,
feature engineering, multi-horizon modeling, and results comparison — and is
available both as a Jupyter notebook and as an interactive Streamlit app.

## Project structure

```
.
├── app.py                                          # Streamlit app (interactive pipeline)
├── Predictive_Forecasting.ipynb                     # Original analysis notebook
├── requirements.txt                                 # Python dependencies
├── HHS_Unaccompanied_Alien_Children_Program.csv      # Raw source data
├── cleaned_data.csv                                  # Cleaned/parsed data (dates, numerics)
└── cleaned_hhs_data.csv                              # Cleaned data reindexed to a continuous daily series
```

## What it does

1. **Data Loading & Cleaning** — parses the `Date` column, strips thousands
   separators from numeric fields, and drops blank/invalid rows.
2. **Data Quality Checks** — flags gaps between consecutive report dates and
   surfaces the largest day-over-day swings in care load for sanity-checking.
3. **Exploratory Data Analysis** — plots the target series over time,
   reindexes to a continuous daily frequency (linear interpolation over
   gaps), and runs a seasonal decomposition (trend / weekly seasonality /
   residual).
4. **Feature Engineering** — builds lag features (t-1, t-7, t-14), rolling
   mean/variance (7-day, 14-day), a "net pressure" signal (transfers in minus
   discharges), and calendar features (day of week, month, weekend flag).
5. **Multi-Horizon Modeling** — trains and backtests models to forecast
   **1-, 7-, and 14-day-ahead** care load:
   - **Baselines:** Naive Persistence, Moving Average
   - **Statistical:** SARIMA, Exponential Smoothing (rolling-origin
     walk-forward backtest so each forecast only uses data available at
     that point in time)
   - **Machine Learning:** Random Forest, Gradient Boosting
6. **Results Summary** — compares every model on MAE, RMSE, and MAPE per
   horizon, with a bar chart of MAE by model and horizon, plus a downloadable
   results CSV.

The notebook additionally includes uncertainty quantification (90% prediction
intervals per model) and an operational KPI / early-warning panel that
translates forecasts into capacity-breach probabilities and intake/exit
imbalance signals for program planners.

## Target & key columns

| Column | Meaning |
|---|---|
| `Children in HHS Care` | **Target** — total children currently in HHS care |
| `Children apprehended and placed in CBP custody*` | New CBP apprehensions |
| `Children in CBP custody` | Current CBP custody count |
| `Children transferred out of CBP custody` | Inflow into HHS care |
| `Children discharged from HHS Care` | Outflow from HHS care |

## Setup

```bash
git clone <this-repo-url>
cd <this-repo>
pip install -r requirements.txt
```

## Usage

### Run the notebook
```bash
jupyter notebook Predictive_Forecasting.ipynb
```
Update `RAW_DATA_PATH` / `CLEANED_DATA_PATH` at the top of the notebook if
you're pointing it at a different data file.

### Run the Streamlit app
```bash
streamlit run app.py
```
Then upload the raw CSV in the sidebar, choose which models/horizons to run,
and click **🚀 Run pipeline**.

## Notes for future iterations

- The best model per horizon is whichever has the lowest MAE for that
  horizon in the results table.
- If ML models underperform the statistical baselines at short horizons, it
  usually means the engineered features aren't adding signal beyond recent
  lags — check `feature_importances_` on the Random Forest model for a clue.
- The forecast horizons, model list, and test window size are all
  configurable from the Streamlit sidebar without touching the code.

## Tech stack

Python · pandas · numpy · matplotlib · statsmodels (SARIMA, Exponential
Smoothing, seasonal decomposition) · scikit-learn (Random Forest, Gradient
Boosting) · Streamlit
