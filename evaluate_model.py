import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def run_evaluation():
    print("=" * 60)
    print("🚀 INITIATING ML MODEL EVALUATION PIPELINE (STATIONARY)")
    print("=" * 60)

    # 1. LOAD AND WRANGLE THE DATA
    filepath = "data/raw/fuel_prices.csv" 
    
    try:
        df_raw = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {filepath}.")
        return

    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
    print("[PROCESSING] Filtering data for Fujairah VLSFO benchmark...")
    df = df_raw[(df_raw['port'] == 'Fujairah') & (df_raw['fuel_type'] == 'VLSFO')].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.rename(columns={'price': 'vlsfo_fujairah'})

    # 2. ADVANCED FEATURE ENGINEERING (STATIONARY TARGET)
    print("[ENGINEERING] Converting absolute prices to stationary price changes...")
    
    # Context features
    df['ma_7'] = df['vlsfo_fujairah'].rolling(window=7).mean()
    df['ma_14'] = df['vlsfo_fujairah'].rolling(window=14).mean()
    df['volatility_7'] = df['vlsfo_fujairah'].rolling(window=7).std()
    
    # Lag and Momentum
    df['price_lag_1'] = df['vlsfo_fujairah'].shift(1)
    df['price_lag_7'] = df['vlsfo_fujairah'].shift(7)
    df['momentum_7d'] = df['vlsfo_fujairah'] - df['price_lag_7']
    
    # THE TARGET: The actual future absolute price (to test against later)
    df['actual_future_price'] = df['vlsfo_fujairah'].shift(-7)
    
    # THE FIX: The ML model will ONLY train on the 7-day CHANGE in price
    df['target_delta_7d'] = df['actual_future_price'] - df['vlsfo_fujairah']
    
    df = df.dropna()

    X = df[['vlsfo_fujairah', 'ma_7', 'ma_14', 'volatility_7', 'price_lag_1', 'price_lag_7', 'momentum_7d']]
    y_delta = df['target_delta_7d']
    y_absolute = df['actual_future_price']

    # 3. TRAIN/TEST SPLIT
    split_index = int(len(df) * 0.8) 
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train_delta, y_test_delta = y_delta.iloc[:split_index], y_delta.iloc[split_index:]
    y_test_abs = y_absolute.iloc[split_index:]
    
    print(f"[SPLIT] Training on {len(X_train)} days, Testing on {len(X_test)} unseen future days.")

    # 4. TRAIN BASELINE MODEL (Predicting the Delta)
    print("[MODEL] Training Baseline (Linear Regression)...")
    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train_delta)
    base_delta_preds = baseline_model.predict(X_test)
    
    # Reconstruct absolute price: Today's Price + Predicted Change
    baseline_preds = X_test['vlsfo_fujairah'] + base_delta_preds

    # 5. TRAIN YOUR MODEL (LightGBM on the Delta)
    print("[MODEL] Training Advanced Model (LightGBM Quantile Regressor)...")
    lgb_params = {
        'objective': 'quantile',
        'alpha': 0.5, 
        'learning_rate': 0.05,
        'n_estimators': 150,
        'random_state': 42
    }
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train_delta)
    lgb_delta_preds = lgb_model.predict(X_test)
    
    # Reconstruct absolute price: Today's Price + Predicted Change
    lgb_preds = X_test['vlsfo_fujairah'] + lgb_delta_preds

    # 6. CALCULATE METRICS (Against the actual absolute prices)
    base_rmse = np.sqrt(mean_squared_error(y_test_abs, baseline_preds))
    base_mape = mean_absolute_percentage_error(y_test_abs, baseline_preds) * 100

    lgb_rmse = np.sqrt(mean_squared_error(y_test_abs, lgb_preds))
    lgb_mape = mean_absolute_percentage_error(y_test_abs, lgb_preds) * 100
    
    improvement_rmse = ((base_rmse - lgb_rmse) / base_rmse) * 100

    # 7. OUTPUT RESULTS
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE REPORT (STATIONARY FORECASTING)")
    print("=" * 60)
    print(f"Legacy Baseline (Linear Regression):")
    print(f"  - RMSE: ${base_rmse:.2f} / MT")
    print(f"  - MAPE: {base_mape:.2f}% error\n")
    
    print(f"Digital Twin Engine (LightGBM):")
    print(f"  - RMSE: ${lgb_rmse:.2f} / MT")
    print(f"  - MAPE: {lgb_mape:.2f}% error\n")
    
    print(f"🏆 TOTAL IMPROVEMENT:")
    print(f"  - LightGBM outperformed the baseline by {improvement_rmse:.1f}% in RMSE.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_evaluation()