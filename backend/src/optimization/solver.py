import gurobipy as gp
from gurobipy import GRB
from src.optimization.vessel_profile import VesselProfile

class BunkeringStochasticSolver:
    """
    Formulates and solves the continuous Two-Leg, Three-Node Stochastic Program 
    for VLCC bunkering and speed optimization using Gurobi.
    Leg 1: Ballast (Current Port to Gulf Load Port)
    Leg 2: Laden (Gulf Load Port to Destination Refinery)
    """
    def __init__(self, vessel: VesselProfile):
        self.vessel = vessel

    def solve_bunkering_problem(self, route_info: dict, current_price: float, scenarios: dict, 
                                current_bunker: float, daily_charter_rate: float = 40000.0, 
                                terminal_target: float = 3000.0) -> dict:
        """
        Solves the continuous loop optimization problem.
        route_info requires: 'dist_leg1_nm' (Ballast) and 'dist_leg2_nm' (Laden)
        """
        dist_leg1 = float(route_info.get('dist_leg1_nm', 1650.0))
        dist_leg2 = float(route_info.get('dist_leg2_nm', 1650.0))
        
        # Initialize Gurobi Model
        model = gp.Model("VLCC_Continuous_Stochastic_Bunkering")
        model.setParam('OutputFlag', 0)
        
        # Enforce Spatial Branch-and-Bound for Bilinear Time/Speed relationships
        model.setParam('NonConvex', 2) 

        # ==========================================
        # 1. NODE 1 DECISION (Immediate Purchase at Current Port)
        # ==========================================
        max_allowable_purchase = self.vessel.max_tank_capacity - current_bunker
        bunker_node1 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_allowable_purchase, name="Bunker_India_Node1")

        # ==========================================
        # 2. STOCHASTIC RECOURSE VARIABLES
        # ==========================================
        speed_leg1, speed_leg2 = {}, {}
        time_leg1, time_leg2 = {}, {}
        burn_leg1, burn_leg2 = {}, {}
        bunker_node2 = {}

        for s in scenarios.keys():
            # Speed Decisions
            speed_leg1[s] = model.addVar(vtype=GRB.CONTINUOUS, lb=self.vessel.min_speed, ub=self.vessel.max_speed, name=f"Speed_Leg1_{s}")
            speed_leg2[s] = model.addVar(vtype=GRB.CONTINUOUS, lb=self.vessel.min_speed, ub=self.vessel.max_speed, name=f"Speed_Leg2_{s}")
            
            # Time & Daily Burn
            time_leg1[s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Time_Leg1_{s}")
            time_leg2[s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Time_Leg2_{s}")
            burn_leg1[s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"DailyBurn_Leg1_{s}")
            burn_leg2[s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"DailyBurn_Leg2_{s}")
            
            # Node 2 Recourse Purchase (Gulf)
            bunker_node2[s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=self.vessel.max_tank_capacity, name=f"Bunker_Gulf_Node2_{s}")

        # ==========================================
        # 3. PHYSICS & NON-LINEAR CONSTRAINTS
        # ==========================================
        for s in scenarios.keys():
            # Exact Bilinear Time Calculation (Time * Speed = Distance / 24)
            model.addConstr(time_leg1[s] * speed_leg1[s] == dist_leg1 / 24.0, f"TimeCalc_Leg1_{s}")
            model.addConstr(time_leg2[s] * speed_leg2[s] == dist_leg2 / 24.0, f"TimeCalc_Leg2_{s}")
            
            # Piecewise Linear Mapping: Speed -> Daily Burn (Using Dual Physics Profiles)
            model.addGenConstrPWL(speed_leg1[s], burn_leg1[s], self.vessel.speed_breakpoints, self.vessel.ballast_fuel_burn, f"PWL_Ballast_{s}")
            model.addGenConstrPWL(speed_leg2[s], burn_leg2[s], self.vessel.speed_breakpoints, self.vessel.laden_fuel_burn, f"PWL_Laden_{s}")

        # ==========================================
        # 4. CHRONOLOGICAL FLOW BALANCE CONSTRAINTS
        # ==========================================
        for s in scenarios.keys():
            # 1. Isolate Leg 1 Burn (Bilinear: Var * Var) into an explicit auxiliary variable
            total_burn_leg1 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"TotalBurn_Leg1_{s}")
            model.addConstr(total_burn_leg1 == time_leg1[s] * burn_leg1[s], f"Def_Burn_Leg1_{s}")
            
            # 2. Isolate Leg 2 Base Burn (Bilinear: Var * Var) into an explicit auxiliary variable
            base_burn_leg2 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"BaseBurn_Leg2_{s}")
            model.addConstr(base_burn_leg2 == time_leg2[s] * burn_leg2[s], f"Def_BaseBurn_Leg2_{s}")
            
            # 3. Calculate Draft Multiplier (Linear Expression dependent on bunker_node1)
            approx_laden_inventory = current_bunker + bunker_node1 - (dist_leg1 / (24 * 11.0) * 35.0) + 1500.0
            draft_mult = self.vessel.calculate_draft_multiplier(approx_laden_inventory)
            
            # 4. Final Leg 2 Burn is now explicitly Quadratic (Auxiliary Var * Linear Expression)
            total_burn_leg2 = base_burn_leg2 * draft_mult
            
            # Constraint 1: Survival at Gulf (Must arrive with reserve)
            model.addConstr(
                current_bunker + bunker_node1 - total_burn_leg1 >= self.vessel.min_safety_reserve, 
                f"Survival_Gulf_{s}"
            )
            
            # Constraint 2: Capacity at Gulf (Cannot overflow tanks when buying Stage 2 fuel)
            model.addConstr(
                current_bunker + bunker_node1 - total_burn_leg1 + bunker_node2[s] <= self.vessel.max_tank_capacity, 
                f"Max_Cap_Gulf_{s}"
            )
            
            # Constraint 3: Terminal Inventory (Must return to India with fuel for next charter)
            model.addConstr(
                current_bunker + bunker_node1 - total_burn_leg1 + bunker_node2[s] - total_burn_leg2 >= terminal_target, 
                f"Terminal_Inv_India_{s}"
            )

        # ==========================================
        # 5. OBJECTIVE: COST + OPPORTUNITY PENALTY
        # ==========================================
        immediate_capital_outlay = bunker_node1 * current_price
        
        expected_recourse_cost = gp.quicksum(
            scenarios[s]['prob'] * (
                (bunker_node2[s] * scenarios[s]['price']) + 
                ((time_leg1[s] + time_leg2[s]) * daily_charter_rate)
            )
            for s in scenarios.keys()
        )
        
        model.setObjective(immediate_capital_outlay + expected_recourse_cost, GRB.MINIMIZE)
        model.optimize()

        # ==========================================
        # 6. PARSE OPTIMAL SOLUTIONS
        # ==========================================
        if model.Status == GRB.OPTIMAL:
            results = {
                "status": "OPTIMAL",
                "immediate_bunker_node1_mt": round(bunker_node1.X, 2),
                "first_stage_cost_usd": round(bunker_node1.X * current_price, 2),
                "expected_total_loop_cost_usd": round(model.ObjVal, 2),
                "terminal_target_mt": terminal_target,
                "scenarios": {}
            }
            
            for s in scenarios.keys():
                results["scenarios"][s] = {
                    "bunker_gulf_node2_mt": round(bunker_node2[s].X, 2),
                    "leg1_ballast": {
                        "speed_knots": round(speed_leg1[s].X, 2),
                        "duration_days": round(time_leg1[s].X, 2),
                        "burn_mt": round(time_leg1[s].X * burn_leg1[s].X, 2)
                    },
                    "leg2_laden": {
                        "speed_knots": round(speed_leg2[s].X, 2),
                        "duration_days": round(time_leg2[s].X, 2),
                        "burn_mt": round(time_leg2[s].X * burn_leg2[s].X, 2)
                    },
                    "loop_opportunity_cost_usd": round((time_leg1[s].X + time_leg2[s].X) * daily_charter_rate, 2)
                }

            model.dispose()
            gp.disposeDefaultEnv()    
            return results
        else:
            return {"status": "INFEASIBLE/FAILED"}