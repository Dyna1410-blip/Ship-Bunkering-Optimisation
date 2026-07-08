# src/models/train.py
import os
import joblib
import lightgbm as lgb
import pandas as pd
from src.features.build_features import FeatureBuilder

class ModelTrainer:
    """ Configures and executes training for asymmetric quantile pipelines. """
    def __init__(self, model_dir: str = "src/models/saved_models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.quantiles = {'Low': 0.10, 'Base': 0.50, 'High': 0.90}

    def train_quantile_ensemble(self, X: pd.DataFrame, y: pd.Series):
        """ Trains three independent models to encapsulate structural risk. """
        for name, q in self.quantiles.items():
            print(f"Fitting LightGBM Regressor | Scenario: {name:<5} (Quantile: {q:.2f})")
            
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=q,
                n_estimators=150,
                learning_rate=0.04,
                max_depth=5,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X, y)
            
            # Save artifact to disk
            model_path = os.path.join(self.model_dir, f"lgb_quantile_{name}.pkl")
            joblib.dump(model, model_path)
            print(f"Model saved successfully to {model_path}")