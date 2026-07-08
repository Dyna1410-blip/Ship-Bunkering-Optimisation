import numpy as np

class VesselProfile:
    """
    Defines the structural, operational, and performance parameters 
    for the VLCC Desh Viraat using a Strict Convex Theoretical model.
    Supports both Laden (fully loaded) and Ballast (empty) conditions 
    for continuous multi-node fleet routing.
    """
    def __init__(self):
        # Physical Attributes
        self.name = "Desh Viraat (Theoretical Cubic)"
        self.vessel_class = "VLCC"
        
        # Volumetric to Mass Conversion
        self.total_tank_volume_m3 = 7845.5
        self.fuel_density_mt_m3 = 0.96
        self.max_fill_factor = 0.98  
        
        # Calculated Maximum Usable Capacity (Metric Tons)
        self.max_tank_capacity = round(
            self.total_tank_volume_m3 * self.fuel_density_mt_m3 * self.max_fill_factor, 2
        ) 
        
        # Operational Safety Constraints
        self.min_safety_reserve = 500.0  
        self.draft_penalty_per_1000mt = 0.005  
        
        # Engine Boundaries
        self.min_speed = 8.0   # knots
        self.max_speed = 14.0  # knots

        # ==========================================
        # THEORETICAL STRICTLY CONVEX PHYSICS (CUBIC LAW)
        # ==========================================
        # Generate 25 perfectly smooth breakpoints to give Gurobi high-resolution options
        self.speed_breakpoints = list(np.round(np.linspace(self.min_speed, self.max_speed, 25), 2))
        
        # --- PROFILE 1: LADEN (Fully Loaded - Gulf to India) ---
        self.laden_base_load = 9.96
        self.laden_c_constant = 0.01496
        self.laden_fuel_burn = [
            round(self.laden_base_load + (self.laden_c_constant * (v ** 3)), 2) 
            for v in self.speed_breakpoints
        ]

        # --- PROFILE 2: BALLAST (Empty Cargo - India to Gulf) ---
        self.ballast_base_load = 8.50
        self.ballast_c_constant = 0.01195
        self.ballast_fuel_burn = [
            round(self.ballast_base_load + (self.ballast_c_constant * (v ** 3)), 2) 
            for v in self.speed_breakpoints
        ]

    def get_base_daily_burn(self, speed: float, state: str = "laden") -> float:
        """
        Linearly interpolates baseline fuel burn (MT/day) based on vessel state.
        Defaults to 'laden' if no state is explicitly passed.
        """
        if state.lower() == "ballast":
            return float(np.interp(speed, self.speed_breakpoints, self.ballast_fuel_burn))
        else:
            return float(np.interp(speed, self.speed_breakpoints, self.laden_fuel_burn))

    def calculate_draft_multiplier(self, current_bunker_mass: float) -> float:
        """Returns the consumption multiplier based on fuel weight deadweight."""
        return 1.0 + (self.draft_penalty_per_1000mt * (current_bunker_mass / 1000.0))