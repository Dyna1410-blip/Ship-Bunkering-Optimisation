import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from main import run_pipeline

# --- 1. INITIALIZE FASTAPI ---
app = FastAPI(
    title="VLCC Digital Twin Engine",
    version="1.0",
    description="API for Stochastic Bunkering & Continuous Loop Routing Optimization."
)

# --- 2. DEFINE THE JSON INPUT SCHEMA (Strict Typing) ---
class RouteOptimizationRequest(BaseModel):
    origin: str
    load_port: str
    destination: str
    current_bunker_onboard: float = 300.0
    target_return_inventory: float = 1000.0
    manual_tce: Optional[float] = None

# --- 3. HELPER: LOAD DISTANCE MATRIX ---
def load_distance_matrix():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, "data", "raw", "full_distance_matrix.csv")
    distance_matrix = {}
    
    try:
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            p1, p2, dist = row['Origin'], row['Destination'], row['Distance_NM']
            if pd.isna(p1) or pd.isna(p2):
                continue
            if p1 not in distance_matrix: distance_matrix[p1] = {}
            if p2 not in distance_matrix: distance_matrix[p2] = {}
            distance_matrix[p1][p2] = float(dist)
            distance_matrix[p2][p1] = float(dist)
        return distance_matrix
    except Exception as e:
        print(f"[API ERROR] Failed to load distance matrix: {e}")
        return None

# Pre-load the matrix into memory when the server starts
DIST_MATRIX = load_distance_matrix()

# --- 4. EXPOSE THE OPTIMIZATION ENDPOINT ---
@app.post("/api/v1/optimize")
def execute_digital_twin(request: RouteOptimizationRequest):
    """
    Ingests routing parameters, looks up geospatial distances, 
    and triggers the ML/Gurobi continuous-loop pipeline.
    """
    if not DIST_MATRIX:
        raise HTTPException(status_code=500, detail="Server failed to initialize the spatial network graph.")
        
    # 1. Look up the distances natively
    try:
        dist_leg1 = DIST_MATRIX[request.origin][request.load_port]
        dist_leg2 = DIST_MATRIX[request.load_port][request.destination]
    except KeyError:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid Route. No physical network link found for {request.origin} -> {request.load_port} -> {request.destination}."
        )

    # 2. Trigger the Core Optimization Engine
    try:
        results = run_pipeline(
            origin=request.origin,
            load_port=request.load_port,
            destination=request.destination,
            dist_leg1=dist_leg1,
            dist_leg2=dist_leg2,
            current_bunker_onboard=request.current_bunker_onboard,
            target_return_inventory=request.target_return_inventory,
            manual_tce=request.manual_tce,
            train_models=False
        )
        
        if results["status"] != "OPTIMAL":
            raise HTTPException(status_code=422, detail="Solver failed to find a mathematically feasible solution for these parameters.")
            
        return {
            "message": "Optimization Solved to Global Optimality",
            "voyage_metrics": {
                "total_loop_distance_nm": dist_leg1 + dist_leg2,
                "leg1_distance_nm": dist_leg1,
                "leg2_distance_nm": dist_leg2
            },
            "directives": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Engine Error: {str(e)}")