# src/data_loader.py
import os
import pandas as pd

class DataLoader:
    """ Handles ingestion and initial cleaning of routing and market datasets. """
    def __init__(self, data_dir: str = None):
        # Dynamically locate the project root (one level up from the 'src' folder)
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if data_dir is None:
            data_dir = os.path.join(PROJECT_ROOT, "data")
            
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
    def load_routes(self) -> pd.DataFrame:
        """ Loads and validates the fixed routes database. """
        path = os.path.join(self.raw_dir, "routes_cl.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Routes file missing at: {path}")
        
        df = pd.read_csv(path)
        # Strip potential whitespaces from critical routing strings
        for col in ['Route_ID', 'Loading_Port', 'Discharge_Port', 'Bunkering_Port']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        return df

    def get_route_by_id(self, route_id: str) -> dict:
        """ Extracts specific parameters for a chosen voyage row. """
        df = self.load_routes()
        route_data = df[df['Route_ID'] == route_id]
        if route_data.empty:
            raise ValueError(f"Route ID '{route_id}' not found in database.")
        return route_data.iloc[0].to_dict()

    def load_market_indicators(self) -> pd.DataFrame:
        """ 
        Ingests the macroeconomic file and the long-format fuel prices file.
        Uses exact headers: 'date', 'WTI_USD_per_bbl', 'Brent_USD_per_bbl'.
        """
        macro_path = os.path.join(self.raw_dir, "market_indicators.csv")
        fuel_path = os.path.join(self.raw_dir, "fuel_prices.csv")
        
        # 1. LOAD MACRO DATA
        df_macro = pd.read_csv(macro_path)
        
        # Mapping based on your provided headers
        df_macro = df_macro.rename(columns={
            'date': 'date', # Ensure lowercase
            'Brent_USD_per_bbl': 'brent_crude',
            # We don't have BDTI in this specific list, so we'll 
            # create a placeholder so the features still work
            'WTI_USD_per_bbl': 'wti_crude' 
        })
        
        # Create a dummy BDTI if it's missing from the file
        if 'bdti' not in df_macro.columns:
            df_macro['bdti'] = 1000.0 
            
        df_macro['date'] = pd.to_datetime(df_macro['date']).dt.normalize()
        
        # 2. LOAD FUEL DATA (Same as before)
        df_fuel = pd.read_csv(fuel_path)
        df_fuel = df_fuel[(df_fuel['port'] == 'Fujairah') & (df_fuel['fuel_type'] == 'VLSFO')]
        df_fuel = df_fuel.rename(columns={'timestamp': 'date', 'price': 'vlsfo_fujairah'})
        df_fuel['date'] = pd.to_datetime(df_fuel['date']).dt.tz_localize(None).dt.normalize()
        
        # 3. MERGE
        df_merged = pd.merge(df_fuel, df_macro, on='date', how='inner')
        return df_merged.sort_values('date').reset_index(drop=True)
        
        # ---------------------------------------------------------
        # 1. LOAD & CLEAN MACRO DATA
        # ---------------------------------------------------------
        df_macro = pd.read_csv(macro_path)
        df_macro = df_macro.rename(columns={
            'Date': 'date',
            'Brent Crude': 'brent_crude',
            'Baltic Dirty Tanker Index (BDTI)': 'bdti'
        })
        # Convert to datetime and strip any time/timezone data for clean merging
        df_macro['date'] = pd.to_datetime(df_macro['date']).dt.normalize()
        
        # ---------------------------------------------------------
        # 2. LOAD & PIVOT FUEL DATA
        # ---------------------------------------------------------
        df_fuel = pd.read_csv(fuel_path)
        
        # Filter strictly for Fujairah and VLSFO
        df_fuel = df_fuel[(df_fuel['port'] == 'Fujairah') & (df_fuel['fuel_type'] == 'VLSFO')]
        
        # Rename columns to match pipeline expectations
        df_fuel = df_fuel.rename(columns={
            'timestamp': 'date',
            'price': 'vlsfo_fujairah'
        })
        
        # Keep only the two columns we need
        df_fuel = df_fuel[['date', 'vlsfo_fujairah']]
        
        # Convert to datetime. The 'Z' in '2023-06-19T00:00:00Z' makes it timezone-aware.
        # We must remove the timezone (.tz_localize(None)) to merge with the macro dates.
        df_fuel['date'] = pd.to_datetime(df_fuel['date']).dt.tz_localize(None).dt.normalize()
        
        # ---------------------------------------------------------
        # 3. MERGE THE DATASETS
        # ---------------------------------------------------------
        # Inner merge will keep only dates that exist in BOTH files
        df_merged = pd.merge(df_fuel, df_macro, on='date', how='inner')
        df_merged = df_merged.sort_values('date').reset_index(drop=True)
        
        # Safety check
        required_cols = ['date', 'vlsfo_fujairah', 'brent_crude', 'bdti']
        for col in required_cols:
            if col not in df_merged.columns:
                raise KeyError(f"Merge failed: Column '{col}' is missing.")
                
        return df_merged