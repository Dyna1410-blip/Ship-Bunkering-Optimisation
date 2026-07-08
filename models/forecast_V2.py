import pandas as pd
import numpy as np
import json
from pathlib import Path

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# =====================================================
# CONFIG
# =====================================================

BUNKER_FILE = "Datasets/port_fuel_prices.csv"
OIL_FILE = "Datasets/WTI_Brent.csv"

TARGET_PORTS = ["Fujairah", "Singapore"]

FUEL_TYPE = "VLSFO"

TEST_SIZE = 0.15
VAL_SIZE = 0.15

OUTPUT_DIR = Path("outputs_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

# =====================================================
# FEATURE ENGINEERING
# =====================================================

def create_features(df):

    # bunker lags

    for lag in [1,3,7,14,30]:
        df[f"price_lag_{lag}"] = df["price"].shift(lag)

    # bunker rolling stats

    for window in [7,14,30]:

        df[f"price_roll_mean_{window}"] = df["price"].rolling(window).mean()
        df[f"price_roll_std_{window}"] = df["price"].rolling(window).std()

    # bunker returns

    df["price_return_1d"] = df["price"].pct_change(1)
    df["price_return_7d"] = df["price"].pct_change(7)

    # oil lags

    for lag in [1,3,7]:

        df[f"brent_lag_{lag}"] = df["Brent_USD_per_bbl"].shift(lag)
        df[f"wti_lag_{lag}"] = df["WTI_USD_per_bbl"].shift(lag)

    # oil returns

    df["brent_return_1d"] = df["Brent_USD_per_bbl"].pct_change(1)
    df["brent_return_7d"] = df["Brent_USD_per_bbl"].pct_change(7)

    df["wti_return_1d"] = df["WTI_USD_per_bbl"].pct_change(1)
    df["wti_return_7d"] = df["WTI_USD_per_bbl"].pct_change(7)

    # oil spread

    df["brent_wti_spread"] = df["Brent_USD_per_bbl"] - df["WTI_USD_per_bbl"]

    # date features

    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["dayofweek"] = df["date"].dt.dayofweek

    # target

    df["target"] = df["price"].shift(-1)

    

    return df.dropna()

# =====================================================
# TRAIN PORT
# =====================================================

def train_port(port_name):

    print("\n" + "="*70)
    print(f"PORT : {port_name}")
    print("="*70)

    bunker = pd.read_csv(BUNKER_FILE)
    oil = pd.read_csv(OIL_FILE)

    bunker["date"] = pd.to_datetime(bunker["timestamp"]).dt.tz_localize(None)
    bunker["date"] = bunker["date"].dt.normalize()

    oil["date"] = pd.to_datetime(oil["date"])

    bunker = bunker[
        (bunker["port"] == port_name) &
        (bunker["fuel_type"].str.upper() == FUEL_TYPE)
    ].copy()

    merged = bunker.merge(
        oil,
        on="date",
        how="inner"
    )


    merged = merged.sort_values("date")

    merged = create_features(merged)

    feature_cols = [c for c in merged.columns if c not in [
    "timestamp",
    "date",
    "port",
    "fuel_type",
    "note",
    "target",
    "price"
    ]]

    X = merged[feature_cols]
    y = merged["target"]

    n = len(merged)

    train_end = int(n * (1 - TEST_SIZE - VAL_SIZE))
    val_end = int(n * (1 - TEST_SIZE))

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    model = ExtraTreesRegressor(
        n_estimators=1000,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train,y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test,preds)
    rmse = np.sqrt(mean_squared_error(y_test,preds))
    mape = mean_absolute_percentage_error(y_test,preds)*100

    print(f"Observations : {len(merged)}")
    print(f"MAE          : {mae:.2f}")
    print(f"RMSE         : {rmse:.2f}")
    print(f"MAPE         : {mape:.2f}%")

    pred_df = pd.DataFrame({
        "Date": merged.iloc[val_end:]["date"].values,
        "Actual": y_test.values,
        "Predicted": preds
    })

    pred_df.to_csv(
        OUTPUT_DIR / f"{port_name}_predictions.csv",
        index=False
    )

    importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    importance.to_csv(
        OUTPUT_DIR / f"{port_name}_feature_importance.csv",
        index=False
    )

    print("\nTop 15 Features:\n")
    print(importance.head(15))

    latest_row = X.iloc[[-1]]

    next_day_forecast = model.predict(latest_row)[0]

    print(f"\nNext Day Forecast : {next_day_forecast:.2f}")

    return {
        "observations": int(len(merged)),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape_percent": float(mape),
        "next_day_forecast": float(next_day_forecast)
    }

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    summary = {}

    for port in TARGET_PORTS:

        metrics = train_port(port)

        summary[port] = metrics

    with open(
        OUTPUT_DIR / "summary_metrics.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print("\n")
    print("="*70)
    print("FINAL SUMMARY")
    print("="*70)

    print(json.dumps(summary,indent=4))
    
        