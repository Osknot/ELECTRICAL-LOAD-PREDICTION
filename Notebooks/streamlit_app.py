import json
import os
import math
import datetime as dt
from typing import List

import pandas as pd
import streamlit as st
from xgboost import XGBRegressor


ARTIFACT_DIR = os.path.join("Notebooks", "ui_artifacts")


@st.cache_resource
def load_artifacts():
    with open(os.path.join(ARTIFACT_DIR, "core_feature_cols.json"), "r") as f:
        core_feature_cols = json.load(f)

    with open(os.path.join(ARTIFACT_DIR, "target_col.json"), "r") as f:
        target_col = json.load(f)

    med_df = pd.read_csv(os.path.join(
        ARTIFACT_DIR, "train_medians.csv"), index_col=0)
    train_medians = med_df.iloc[:, 0]  # pandas Series indexed by feature name

    model = XGBRegressor()
    model.load_model(os.path.join(ARTIFACT_DIR, "xgb_load_model.json"))

    return core_feature_cols, target_col, train_medians, model


def build_feature_row(
    *,
    timestamp_str: str,
    temperature_C: float,
    humidity: float,
    wind_speed: float,
    lag_1: float = math.nan,
    lag_24: float = math.nan,
    rolling_mean_3: float = math.nan,
    train_medians: pd.Series,
    core_feature_cols: List[str],
) -> pd.DataFrame:
    ts = pd.to_datetime(timestamp_str, errors="coerce")
    if pd.isna(ts):
        raise ValueError(
            "timestamp could not be parsed. Use e.g. '2022-01-01 13:00:00'.")

    hour = int(ts.hour)
    # Monday=0 ... Sunday=6 (matches typical dataset conventions)
    day_of_week = int(ts.dayofweek)
    is_weekend = int(day_of_week >= 5)

    row = {
        "temperature_C": temperature_C,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "lag_1": lag_1,
        "lag_24": lag_24,
        "rolling_mean_3": rolling_mean_3,
    }

    X_row = pd.DataFrame([row], columns=core_feature_cols)

    # Apply same imputation strategy as the notebook for missing feature values:
    # fill NaNs with training medians.
    X_row = X_row.fillna(train_medians)

    return X_row


def compute_lags_from_history(timestamp_str: str, history_df: pd.DataFrame):
    ts = pd.to_datetime(timestamp_str, errors="coerce")
    if pd.isna(ts):
        raise ValueError(
            "timestamp could not be parsed. Use e.g. '2022-01-01 13:00:00'.")

    if history_df is None or history_df.empty:
        return math.nan, math.nan, math.nan, "No history file provided; lag features were imputed."

    if "timestamp" not in history_df.columns or "load_MW" not in history_df.columns:
        raise ValueError(
            "History CSV must contain columns: timestamp, load_MW")

    h = history_df.copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"], errors="coerce")
    h = h.dropna(subset=["timestamp"]).sort_values("timestamp")
    h = h[["timestamp", "load_MW"]].dropna(subset=["load_MW"])

    series = h.set_index("timestamp")["load_MW"]
    lag_1 = series.get(ts - pd.Timedelta(hours=1), math.nan)
    lag_24 = series.get(ts - pd.Timedelta(hours=24), math.nan)

    rm_vals = [
        series.get(ts - pd.Timedelta(hours=1), math.nan),
        series.get(ts - pd.Timedelta(hours=2), math.nan),
        series.get(ts - pd.Timedelta(hours=3), math.nan),
    ]
    rolling_mean_3 = (
        float(sum(rm_vals) / 3.0) if all(pd.notna(v)
                                         for v in rm_vals) else math.nan
    )

    missing = [name for name, val in [
        ("lag_1", lag_1), ("lag_24", lag_24), ("rolling_mean_3", rolling_mean_3)] if pd.isna(val)]
    if missing:
        msg = (
            "Some lag values were unavailable from history "
            f"({', '.join(missing)}); those were imputed."
        )
    else:
        msg = "Lag features were computed automatically from history."

    return lag_1, lag_24, rolling_mean_3, msg


def main():
    st.title("Electrical Energetics Project\nSmart Grid Load Prediction\n Built by Iniobong Oscar Ebong and Libera Mervin Ninziza")

    if not os.path.exists(ARTIFACT_DIR):
        st.error(
            f"Missing `{ARTIFACT_DIR}` directory. Run the notebook and export artifacts "
            f"using the section `Export Model Artifacts (for the UI)`, then restart."
        )
        return

    core_feature_cols, target_col, train_medians, model = load_artifacts()

    st.markdown("### Enter inputs")
    st.caption("Select date and time for when you want the load prediction.")
    default_dt = dt.datetime.now().replace(minute=0, second=0, microsecond=0)
    pred_date = st.date_input("Prediction date", value=default_dt.date())
    pred_time = st.time_input(
        "Prediction time", value=default_dt.time(), step=dt.timedelta(minutes=15))
    selected_ts = dt.datetime.combine(pred_date, pred_time)
    timestamp_str = selected_ts.strftime("%Y-%m-%d %H:%M:%S")
    st.write(f"Selected timestamp: `{timestamp_str}`")

    temperature_C = st.number_input("temperature_C", value=10.0)
    humidity = st.number_input("humidity", value=70.0)
    wind_speed = st.number_input("wind_speed", value=5.0)

    st.markdown("### Historical load input (for automatic lag computation)")
    st.caption(
        "Upload a CSV with columns `timestamp` and `load_MW`. "
        "The app computes lag_1, lag_24, and rolling_mean_3 automatically."
    )
    history_file = st.file_uploader(
        "Upload history CSV (optional)", type=["csv"])

    if st.button("Predict", type="primary"):
        try:
            history_df = pd.read_csv(
                history_file) if history_file is not None else pd.DataFrame()
            lag_1, lag_24, rolling_mean_3, lag_msg = compute_lags_from_history(
                timestamp_str, history_df)

            X_row = build_feature_row(
                timestamp_str=timestamp_str,
                temperature_C=temperature_C,
                humidity=humidity,
                wind_speed=wind_speed,
                lag_1=lag_1,
                lag_24=lag_24,
                rolling_mean_3=rolling_mean_3,
                train_medians=train_medians,
                core_feature_cols=core_feature_cols,
            )

            pred = model.predict(X_row[core_feature_cols])[0]

            st.success(f"Predicted {target_col}: {pred:,.3f} MW")
            st.info(lag_msg)
            with st.expander("Show features used"):
                st.dataframe(X_row)
        except Exception as e:
            st.error(str(e))


if __name__ == "__main__":
    main()
