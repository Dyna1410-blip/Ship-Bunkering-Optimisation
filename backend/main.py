import os
import numpy as np
import pandas as pd
from src.data_loader import DataLoader
from src.features.build_features import FeatureBuilder
from src.models.train import ModelTrainer
from src.models.predict import ScenarioPredictor
from src.optimization.vessel_profile import VesselProfile
from src.optimization.solver import BunkeringStochasticSolver

def run_pipeline(origin: str, load_port: str, destination: str, dist_leg1: float, dist_leg2: float, current_bunker_onboard: float, target_return_inventory: float, train_models: bool = False):
    print("=" * 75)
    print("STARTING CONTINUOUS LOOP OPTIMIZATION PIPELINE")
    print("=" * 75)

    # ----------------------------------------------------------------
    # 1. INITIALIZE DATA INGESTION & ROUTE MAPPING
    # ----------------------------------------------------------------
    loader = DataLoader()
    
    # Construct route_info dynamically from UI inputs
    route_info = {
        'dist_leg1_nm': dist_leg1,
        'dist_leg2_nm': dist_leg2,
        'Loading_Port': load_port,
        'Discharge_Port': destination
    }
            
    print(f"[DATA] Selected Corridor: {origin} -> {load_port} -> {destination}")
    print(f"[DATA] Leg 1 (Ballast) Distance: {dist_leg1} NM")
    print(f"[DATA] Leg 2 (Laden) Distance: {dist_leg2} NM")
    
    # Generate or load global market indicators context
    try:
        market_df = loader.load_market_indicators()
    except Exception as e:
        print(f"[WARNING] Global Market Data load failed ({e}). Generating simulated context...")
        dates = pd.date_range(end='2026-06-25', periods=200, freq='D')
        np.random.seed(42)
        vlsfo_base = 550.0 + np.cumsum(np.random.normal(0, 4, 200))
        brent_base = 80.0 + np.cumsum(np.random.normal(0, 0.8, 200))
        bdti_base = 1000 + np.cumsum(np.random.normal(0, 15, 200))
        market_df = pd.DataFrame({'date': dates, 'vlsfo_fujairah': vlsfo_base, 'brent_crude': brent_base, 'bdti': bdti_base})

    # ----------------------------------------------------------------
    # 2. BASIS RISK CALCULATION (THE MUMBAI PREMIUM)
    # ----------------------------------------------------------------
    try:
        # Load your synthesized Mumbai pricing dataset
        # Adjust path if mumbai.csv is in data/raw/
        # Dynamically build the absolute path for the Docker container
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        mumbai_path = os.path.join(BASE_DIR, 'data', 'raw', 'mumbai.csv')
        regional_df = pd.read_csv(mumbai_path)
        fujairah_mean = regional_df[regional_df['port'] == 'Fujairah']['price'].mean()
        mumbai_mean = regional_df[regional_df['port'] == 'Mumbai']['price'].mean()
        static_india_premium = mumbai_mean - fujairah_mean
        print(f"[DATA] Ingested Regional Pricing. Calculated Constant Basis Spread: +${static_india_premium:.2f}/MT")
    except Exception as e:
        print(f"[WARNING] Could not process 'mumbai.csv' ({e}). Defaulting to standard logistics premium.")
        static_india_premium = 25.0

    # ----------------------------------------------------------------
    # 3. MACHINE LEARNING ENGINE: RISK FORECASTING (GLOBAL HUB ONLY)
    # ----------------------------------------------------------------
    feature_builder = FeatureBuilder(forward_horizon_days=7)
    df_features = feature_builder.create_stationarity_features(market_df)
    X, y = feature_builder.get_training_matrices(df_features)

    # Optional training trigger
    if train_models or not os.path.exists("src/models/saved_models/lgb_quantile_Base.pkl"):
        print("\n[ML] Triggering LightGBM Quantile Ensemble Training Phase...")
        trainer = ModelTrainer()
        trainer.train_quantile_ensemble(X, y)

    # Extract current real-time state for inference (Fujairah Baseline)
    latest_market_row = df_features.iloc[-1:]
    current_fujairah_spot = float(df_features['vlsfo_fujairah'].iloc[-1])
    
    # Calculate Node 1 (India) Price using the Basis Premium
    current_india_spot = current_fujairah_spot + static_india_premium

    predictor = ScenarioPredictor()
    scenarios = predictor.generate_gurobi_scenarios(latest_market_row, current_fujairah_spot)

    print(f"\n[ML] Current Market Spot Price (Node 1 - {origin}): ${current_india_spot:.2f}/MT")
    print(f"[ML] Current Market Spot Price (Node 2 - {load_port}): ${current_fujairah_spot:.2f}/MT")
    print(f"[ML] Generated Downstream Recourse Scenarios (Node 2 - {load_port}):")
    for s_name, data in scenarios.items():
        print(f"  -> Scenario {s_name:<7} | Forward Price: ${data['price']:<7.2f} | Probability: {data['prob']*100:.0f}%")

    # ----------------------------------------------------------------
    # 4. OPERATIONS RESEARCH: GUROBI CONTINUOUS LOOP OPTIMIZATION
    # ----------------------------------------------------------------
    print("\n[OR] Initializing VLCC Digital Twin (Dual Physics) & Gurobi Solver...")
    desh_viraat = VesselProfile()
    solver = BunkeringStochasticSolver(vessel=desh_viraat)

    # Define the Time Charter Equivalent (TCE) rate
    daily_tce_rate = 34260.0  # $34,260/day opportunity cost

    optimization_results = solver.solve_bunkering_problem(
        route_info=route_info,
        current_price=current_india_spot,  
        scenarios=scenarios,               
        current_bunker=current_bunker_onboard,
        daily_charter_rate=daily_tce_rate,
        terminal_target=target_return_inventory
    )

    # ----------------------------------------------------------------
    # 5. REPORTING LOG OUTPUTS
    # ----------------------------------------------------------------
    print("\n" + "=" * 75)
    print("CONTINUOUS LOOP OPTIMAL DECISION SUMMARY MATRIX")
    print("=" * 75)
    
    if optimization_results["status"] == "OPTIMAL":
        print(f"RECOMMENDED NODE-1 PURCHASE ({origin})  : {optimization_results['immediate_bunker_node1_mt']:.2f} MT")
        print(f"Immediate Capital Outlay Requirement     : ${optimization_results['first_stage_cost_usd']:,.2f}")
        print(f"Stochastic Expected Total Loop Cost      : ${optimization_results['expected_total_loop_cost_usd']:,.2f}")
        print(f"Target Terminal Inventory ({destination}) : {optimization_results['terminal_target_mt']:.2f} MT")
        
        print(f"\nRecourse Contingency Fleet Directives (Node 2 - {load_port}):")
        for s_name, s_data in optimization_results["scenarios"].items():
            print(f"  If Market Realizes [{s_name}]:")
            print(f"    - Secure at Gulf Loading Port     : {s_data['bunker_gulf_node2_mt']:.2f} MT")
            
            # Formatting the nested physics outputs
            leg1 = s_data.get('leg1_ballast', {})
            leg2 = s_data.get('leg2_laden', {})
            
            if leg1 and leg2:
                print(f"    - Leg 1 (Ballast) Transit Profile : {leg1['speed_knots']:.2f} knots | {leg1['duration_days']:.2f} days | {leg1['burn_mt']:.2f} MT burn")
                print(f"    - Leg 2 (Laden) Transit Profile   : {leg2['speed_knots']:.2f} knots | {leg2['duration_days']:.2f} days | {leg2['burn_mt']:.2f} MT burn")
                print(f"    - Loop Time Opportunity Cost (TCE): ${s_data['loop_opportunity_cost_usd']:,.2f}")
            else:
                # Fallback format for single-leg testing gracefully
                print(f"    - Optimal Transit Speed           : {s_data.get('optimized_speed_knots', 0):.2f} knots")
                print(f"    - Projected Physical Burn         : {s_data.get('predicted_burn_mt', 0):.2f} MT")

            print("    " + "-"*55)
    else:
        print("[ERROR] Optimization failed or returned an infeasible boundary matrix.")
    print("=" * 75 + "\n")
    
    # Return the dictionary so the Streamlit UI or FastAPI endpoint can ingest it
    return optimization_results

if __name__ == "__main__":
    # Example execution if run from terminal instead of Streamlit
    run_pipeline(
        origin="Paradip", 
        load_port="Ras Tanura", 
        destination="Paradip", 
        dist_leg1=3450.0, 
        dist_leg2=3450.0, 
        current_bunker_onboard=495.0, 
        target_return_inventory=3000.0,
        train_models=False
    )