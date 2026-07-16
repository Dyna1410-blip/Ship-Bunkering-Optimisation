# src/models/predict.py
import os
import joblib
import numpy as np
import pandas as pd

class ScenarioPredictor:
    """ Evaluates current features to output expected Gurobi price dictionaries. """
    def __init__(self, model_dir: str = "src/models/saved_models"):
        self.model_dir = model_dir
        self.scenarios = ['Low', 'Base', 'High']
        self.probabilities = {'Low': 0.20, 'Base': 0.60, 'High': 0.20}

    def generate_gurobi_scenarios(self, latest_row: pd.DataFrame, current_spot: float) -> dict:
        """
        Maps current indicators to future scenarios.
        Formula: Price_{t+h} = Price_t * exp(predicted_log_return)
        """
        feature_cols = [
            'brent_return_7d', 'brent_return_14d', 
            'spread_zscore', 'vlsfo_volatility_14d', 
            'bdti_momentum_7d'
        ]
        
        X_latest = latest_row[feature_cols]
        output_scenarios = {}
        
        for name in self.scenarios:
            model_path = os.path.join(self.model_dir, f"lgb_quantile_{name}.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file missing: {model_path}. Run training first.")
                
            model = joblib.load(model_path)
            pred_log_return = model.predict(X_latest)[0]
            
            # Reconstruct nominal price boundary
            future_price = current_spot * np.exp(pred_log_return)
            
            output_scenarios[f"S_{name}"] = {
                'price': round(float(future_price), 2),
                'prob': self.probabilities[name]
            }
            
        return output_scenarios