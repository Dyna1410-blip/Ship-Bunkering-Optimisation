import pandas as pd
import numpy as np
import json
from pathlib import Path

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

# =====================================================
# CONFIG
# =====================================================

CSV_FILE = "port_fuel_prices.csv"
FUEL_TYPE = "VLSFO"

TARGET_PORTS = [
    "Fujairah",
    "Singapore"
]

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TEST_SIZE = 0.15
VAL_SIZE = 0.15

# =====================================================
# FEATURE ENGINEERING
# =====================================================

def create_features(df):

    for lag in [1, 3, 7, 14, 30]:
        df[f"lag_{lag}"] = df["price"].shift(lag)

    for window in [7, 14, 30]:
        df[f"roll_mean_{window}"] = (
            df["price"]
            .rolling(window)
            .mean()
        )

        df[f"roll_std_{window}"] = (
            df["price"]
            .rolling(window)
            .std()
        )

    df["month"] = df["timestamp"].dt.month
    df["quarter"] = df["timestamp"].dt.quarter
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    df["target"] = df["price"].shift(-1)

    return df.dropna()

# =====================================================
# TRAIN ONE PORT
# =====================================================

def train_port(port_name):

    print("\n" + "=" * 60)
    print(f"PORT : {port_name}")
    print("=" * 60)

    df = pd.read_csv(CSV_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df[
        (df["port"] == port_name)
        &
        (df["fuel_type"].str.upper() == FUEL_TYPE)
    ].copy()

    df = df.sort_values("timestamp")

    df = create_features(df)

    feature_cols = sorted([
        c for c in df.columns
        if c.startswith("lag_")
        or c.startswith("roll_")
    ])

    feature_cols += [
        "month",
        "quarter",
        "dayofweek"
    ]

    X = df[feature_cols]
    y = df["target"]

    n = len(df)

    train_end = int(
        n * (1 - TEST_SIZE - VAL_SIZE)
    )

    val_end = int(
        n * (1 - TEST_SIZE)
    )

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    model = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        preds
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            preds
        )
    )

    mape = (
        mean_absolute_percentage_error(
            y_test,
            preds
        ) * 100
    )

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")

    pred_df = pd.DataFrame({
        "Date": df.iloc[val_end:]["timestamp"].values,
        "Actual": y_test.values,
        "Predicted": preds
    })

    pred_file = OUTPUT_DIR / f"{port_name}_predictions.csv"
    pred_df.to_csv(pred_file, index=False)

    latest_row = X.iloc[[-1]]

    next_day_forecast = model.predict(
        latest_row
    )[0]

    print(
        f"Next Day Forecast = "
        f"{next_day_forecast:.2f}"
    )

    feature_importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importance":
            model.feature_importances_
    })

    feature_importance = (
        feature_importance
        .sort_values(
            "Importance",
            ascending=False
        )
    )

    feature_importance.to_csv(
        OUTPUT_DIR /
        f"{port_name}_feature_importance.csv",
        index=False
    )

    return {
        "observations": len(df),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape_percent": float(mape),
        "next_day_forecast":
            float(next_day_forecast)
    }

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    summary = {}

    for port in TARGET_PORTS:

        metrics = train_port(port)

        summary[port] = metrics

    summary_file = (
        OUTPUT_DIR /
        "summary_metrics.json"
    )

    with open(summary_file, "w") as f:
        json.dump(
            summary,
            f,
            indent=4
        )

    print("\n")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        json.dumps(
            summary,
            indent=4
        )
    )