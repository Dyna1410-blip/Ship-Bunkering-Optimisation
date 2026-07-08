# src/features/build_features.py
import numpy as np
import pandas as pd

class FeatureBuilder:
    """
    Transforms raw non-stationary time-series data into stationary features
    and structures target vectors for quantile regression.
    """
    def __init__(self, forward_horizon_days: int = 7):
        self.horizon = forward_horizon_days

    def create_stationarity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes raw market indicators into mathematical features.
        Required columns in df: 'vlsfo_fujairah', 'brent_crude', 'bdti'
        """
        df = df.copy().sort_values('date').reset_index(drop=True)
        
        # 1. Target Generation: Forward Log Return
        df['target_forward_return'] = np.log(
            df['vlsfo_fujairah'].shift(-self.horizon) / df['vlsfo_fujairah']
        )
        
        # 2. Crude Oil Momentum Features
        df['brent_return_7d'] = np.log(df['brent_crude'] / df['brent_crude'].shift(7))
        df['brent_return_14d'] = np.log(df['brent_crude'] / df['brent_crude'].shift(14))
        
        # 3. Refining Economics (The Crack Spread Z-Score)
        df['crack_spread'] = df['vlsfo_fujairah'] - df['brent_crude']
        df['spread_30d_mean'] = df['crack_spread'].rolling(window=30).mean()
        df['spread_30d_std'] = df['crack_spread'].rolling(window=30).std()
        df['spread_zscore'] = (df['crack_spread'] - df['spread_30d_mean']) / df['spread_30d_std']
        
        # 4. Volatility Archetypes (Uncertainty Band Scaling)
        df['vlsfo_volatility_14d'] = df['vlsfo_fujairah'].pct_change().rolling(window=14).std()
        
        # 5. Downstream Shipping Demand Momentum (Freight Index)
        df['bdti_momentum_7d'] = df['bdti'].pct_change(periods=7)
        
        # Drop boundary NaNs resulting from rolling transformations
        # Keep forward target NaNs only for operational prediction tracking
        return df

    def get_training_matrices(self, df_features: pd.DataFrame):
        """ Separates feature columns from the future target vector. """
        feature_cols = [
            'brent_return_7d', 'brent_return_14d', 
            'spread_zscore', 'vlsfo_volatility_14d', 
            'bdti_momentum_7d'
        ]
        
        # Eliminate rows where target is missing (the most recent historical window)
        train_clean = df_features.dropna(subset=['target_forward_return'] + feature_cols)
        
        X = train_clean[feature_cols]
        y = train_clean['target_forward_return']
        return X, y