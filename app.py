"""
Predictive Forecasting of Care Load & Placement Demand — Streamlit App
========================================================================
Interactive version of the forecasting notebook. Upload the raw CSV
(or use the bundled sample path), then walk through cleaning, EDA,
feature engineering, and multi-horizon model comparison (baseline →
statistical → ML) in the browser.
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

st.set_page_config(page_title="Care Load Forecasting", layout="wide")

TARGET_COL = "Children in HHS Care"
DISCHARGE_COL = "Children discharged from HHS Care"
TRANSFER_COL = "Children transferred out of CBP custody"
NUMERIC_COLS = [
    "Children apprehended and placed in CBP custody*",
    "Children in CBP custody",
    TRANSFER_COL,
    TARGET_COL,
    DISCHARGE_COL,
]

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df = df.dropna(how="all")
    df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], format="%B %d, %Y", errors="coerce")
    df = df.dropna(subset=["Date"])

    for col in NUMERIC_COLS:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def make_continuous(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("Date")
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_range)
    df.index.name = "Date"
    df[NUMERIC_COLS] = df[NUMERIC_COLS].interpolate(method="linear")
    return df


@st.cache_data(show_spinner=False)
def engineer_features(df: pd.DataFrame, horizons: tuple) -> tuple:
    df = df.copy()

    for lag in [1, 7, 14]:
        df[f"{TARGET_COL}_lag{lag}"] = df[TARGET_COL].shift(lag)
        df[f"{DISCHARGE_COL}_lag{lag}"] = df[DISCHARGE_COL].shift(lag)

    for window in [7, 14]:
        df[f"{TARGET_COL}_roll_mean{window}"] = df[TARGET_COL].shift(1).rolling(window).mean()
        df[f"{TARGET_COL}_roll_var{window}"] = df[TARGET_COL].shift(1).rolling(window).var()

    df["net_pressure"] = df[TRANSFER_COL] - df[DISCHARGE_COL]
    df["net_pressure_roll7"] = df["net_pressure"].shift(1).rolling(7).mean()

    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df_features = df.dropna()

    for h in horizons:
        df_features[f"{TARGET_COL}_target_h{h}"] = df_features[TARGET_COL].shift(-h)
    df_features = df_features.dropna()

    exclude_cols = NUMERIC_COLS + ["net_pressure"] + [f"{TARGET_COL}_target_h{h}" for h in horizons]
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    return df_features, feature_cols


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(y_true, y_pred, model_name, horizon):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_val = mape(y_true, y_pred)
    return {"model": model_name, "horizon": horizon, "MAE": mae, "RMSE": rmse, "MAPE": mape_val}


def naive_forecast(train_data, test_len):
    last_val = train_data[TARGET_COL].iloc[-1]
    return np.full(test_len, last_val)


def moving_avg_forecast(train_data, test_len, window=7):
    avg_val = train_data[TARGET_COL].tail(window).mean()
    return np.full(test_len, avg_val)


def sarima_forecast(train_series, steps, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    return fit.forecast(steps=steps)


def exp_smoothing_forecast(train_series, steps, seasonal_periods=7):
    model = ExponentialSmoothing(train_series, trend="add", seasonal="add",
                                  seasonal_periods=seasonal_periods)
    fit = model.fit()
    return fit.forecast(steps=steps)


# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------

st.sidebar.title("⚙️ Settings")
uploaded = st.sidebar.file_uploader("Upload CSV (HHS custody/placement data)", type=["csv"])

test_size_days = st.sidebar.slider("Test window (days)", min_value=14, max_value=120, value=60, step=1)

available_models = ["Naive Persistence", "Moving Average", "SARIMA", "Exponential Smoothing",
                     "Random Forest", "Gradient Boosting"]
selected_models = st.sidebar.multiselect("Models to run", available_models, default=available_models)

horizons_selected = st.sidebar.multiselect("Forecast horizons (days ahead)", [1, 7, 14], default=[1, 7, 14])

run_button = st.sidebar.button("🚀 Run pipeline", type="primary")

st.title("📈 Predictive Forecasting of Care Load & Placement Demand")
st.caption(
    "Forecasts the number of children in HHS care using historical CBP custody "
    "and placement data. Data loading → EDA → feature engineering → model comparison."
)

if uploaded is None:
    st.info("Upload the raw CSV in the sidebar to get started.")
    st.stop()

# ----------------------------------------------------------------------
# 1. Data loading & cleaning
# ----------------------------------------------------------------------

df_raw = load_and_clean(uploaded.getvalue())

with st.expander("1️⃣ Data Loading & Cleaning", expanded=True):
    c1, c2 = st.columns(2)
    c1.metric("Rows", df_raw.shape[0])
    c2.metric("Date range", f"{df_raw['Date'].min().date()} → {df_raw['Date'].max().date()}")
    missing = df_raw.isna().sum()
    if missing.sum() > 0:
        st.warning("Missing values detected:")
        st.dataframe(missing[missing > 0])
    st.dataframe(df_raw.head(), use_container_width=True)

# ----------------------------------------------------------------------
# 2. Data quality checks
# ----------------------------------------------------------------------

with st.expander("2️⃣ Data Quality Checks"):
    gap_df = df_raw.copy()
    gap_df["gap_days"] = gap_df["Date"].diff().dt.days
    st.write("**Gap between consecutive report dates**")
    st.dataframe(gap_df["gap_days"].describe().to_frame().T, use_container_width=True)
    big_gaps = gap_df[gap_df["gap_days"] > 3]
    if not big_gaps.empty:
        st.write(f"Rows preceded by a gap > 3 days ({len(big_gaps)} found):")
        st.dataframe(big_gaps[["Date", "gap_days"]], use_container_width=True)

    gap_df["HHS_care_change"] = gap_df[TARGET_COL].diff()
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Biggest drops**")
        st.dataframe(gap_df.sort_values("HHS_care_change").head()[["Date", TARGET_COL, "HHS_care_change"]])
    with c2:
        st.write("**Biggest jumps**")
        st.dataframe(gap_df.sort_values("HHS_care_change").tail()[["Date", TARGET_COL, "HHS_care_change"]])

# ----------------------------------------------------------------------
# 3. EDA
# ----------------------------------------------------------------------

df_cont = make_continuous(df_raw)

with st.expander("3️⃣ Exploratory Data Analysis"):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_cont.index, df_cont[TARGET_COL])
    ax.set_title(f"{TARGET_COL} Over Time")
    st.pyplot(fig)
    plt.close(fig)

    try:
        decomposition = seasonal_decompose(df_cont[TARGET_COL], model="additive", period=7)
        fig2 = decomposition.plot()
        fig2.set_size_inches(12, 7)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
    except Exception as e:
        st.warning(f"Could not compute seasonal decomposition: {e}")

# ----------------------------------------------------------------------
# 4. Feature engineering
# ----------------------------------------------------------------------

if not horizons_selected:
    st.error("Select at least one forecast horizon in the sidebar.")
    st.stop()

horizons_tuple = tuple(sorted(horizons_selected))
df_features, feature_cols = engineer_features(df_cont, horizons_tuple)

with st.expander("4️⃣ Feature Engineering"):
    st.write(f"Shape after feature engineering: **{df_features.shape}**")
    st.write("Feature columns used for ML models:")
    st.code(", ".join(feature_cols))
    st.dataframe(df_features.head(), use_container_width=True)

# ----------------------------------------------------------------------
# 5. Train/test split
# ----------------------------------------------------------------------

if test_size_days >= len(df_features):
    st.error("Test window is larger than the available feature-engineered data. Reduce it in the sidebar.")
    st.stop()

split_date = df_features.index[-test_size_days]
train_df = df_features[df_features.index < split_date]
test_df = df_features[df_features.index >= split_date]

with st.expander("5️⃣ Train / Test Split"):
    c1, c2 = st.columns(2)
    c1.metric("Train rows", len(train_df))
    c2.metric("Test rows", len(test_df))
    st.write(f"Train range: {train_df.index.min().date()} → {train_df.index.max().date()}")
    st.write(f"Test range: {test_df.index.min().date()} → {test_df.index.max().date()}")

# ----------------------------------------------------------------------
# 6. Model comparison
# ----------------------------------------------------------------------

if not run_button:
    st.info("Adjust settings in the sidebar, then click **Run pipeline** to train and compare models.")
    st.stop()

results = []
X_train_ml = train_df[feature_cols]
X_test_ml = test_df[feature_cols]

with st.spinner("Training and evaluating models..."):
    for h in horizons_tuple:
        target = f"{TARGET_COL}_target_h{h}"
        y_test_h = test_df[target]
        y_train_h = train_df[target]

        if "Naive Persistence" in selected_models:
            pred = naive_forecast(train_df, len(test_df))
            results.append(evaluate(y_test_h, pred, "Naive Persistence", h))

        if "Moving Average" in selected_models:
            pred = moving_avg_forecast(train_df, len(test_df))
            results.append(evaluate(y_test_h, pred, "Moving Average", h))

        if "SARIMA" in selected_models:
            try:
                pred = sarima_forecast(train_df[TARGET_COL], steps=len(test_df))
                results.append(evaluate(y_test_h, pred.values[: len(y_test_h)], "SARIMA", h))
            except Exception as e:
                st.warning(f"SARIMA failed for horizon {h}: {e}")

        if "Exponential Smoothing" in selected_models:
            try:
                pred = exp_smoothing_forecast(train_df[TARGET_COL], steps=len(test_df))
                results.append(evaluate(y_test_h, pred.values[: len(y_test_h)], "Exponential Smoothing", h))
            except Exception as e:
                st.warning(f"Exponential Smoothing failed for horizon {h}: {e}")

        if "Random Forest" in selected_models:
            rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
            rf.fit(X_train_ml, y_train_h)
            pred = rf.predict(X_test_ml)
            results.append(evaluate(y_test_h, pred, "Random Forest", h))

        if "Gradient Boosting" in selected_models:
            gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
            gb.fit(X_train_ml, y_train_h)
            pred = gb.predict(X_test_ml)
            results.append(evaluate(y_test_h, pred, "Gradient Boosting", h))

results_df = pd.DataFrame(results).sort_values(["horizon", "MAE"]).reset_index(drop=True)

st.header("6️⃣ Results Summary")

if results_df.empty:
    st.warning("No models were run — check your selections in the sidebar.")
    st.stop()

for h in horizons_tuple:
    h_results = results_df[results_df["horizon"] == h].sort_values("MAE")
    st.subheader(f"Horizon: {h} day(s) ahead")
    best_model = h_results.iloc[0]
    st.success(f"🏆 Best model: **{best_model['model']}** — MAE {best_model['MAE']:.1f}, RMSE {best_model['RMSE']:.1f}, MAPE {best_model['MAPE']:.2f}%")
    st.dataframe(
        h_results.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.2f}"}),
        use_container_width=True,
    )

st.subheader("MAE by Model and Forecast Horizon")
fig3, ax3 = plt.subplots(figsize=(10, 5))
pivot = results_df.pivot(index="model", columns="horizon", values="MAE")
pivot.plot(kind="bar", ax=ax3)
ax3.set_ylabel("MAE")
ax3.set_title("Model MAE by Forecast Horizon (days)")
ax3.legend(title="Horizon (days)")
plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

csv_bytes = results_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download results as CSV", data=csv_bytes, file_name="model_results.csv", mime="text/csv")

st.caption(
    "Note: if ML models underperform the statistical baselines at short horizons, it usually "
    "means the engineered features aren't adding signal beyond recent lags — check "
    "`feature_importances_` on the Random Forest model for a clue."
)
