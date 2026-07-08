# fuel_cost_analysis.py

import pandas as pd
from fuel_model import DeshViraatFuelModel

# ===================================
# LOAD ROUTES
# ===================================

routes = pd.read_csv("Datasets/routes_cl.csv")

print("\nAvailable Routes:\n")
print(routes[["Route_ID", "Loading_Port", "Discharge_Port"]])

# ===================================
# USER INPUTS
# ===================================

route_id = input(
    "\nEnter Route ID (e.g. R05): "
).strip()

speed = float(
    input("Enter vessel speed (knots): ")
)

vlsfo_price = float(
    input("Enter VLSFO Price ($/MT): ")
)

# ===================================
# FIND ROUTE
# ===================================

route = routes[
    routes["Route_ID"] == route_id
]

if route.empty:
    print(f"\nRoute {route_id} not found.")
    exit()

route = route.iloc[0]

# ===================================
# MODEL
# ===================================

model = DeshViraatFuelModel()

distance = route["Distance_Nautical_Miles"]

voyage_days = model.voyage_days(
    distance,
    speed
)

daily_consumption = model.daily_consumption(
    speed
)

fuel_required = model.fuel_required(
    distance,
    speed
)

fuel_cost = (
    fuel_required *
    vlsfo_price
)

# ===================================
# RESULTS
# ===================================

print("\n")
print("=" * 60)
print("VOYAGE FUEL COST ANALYSIS")
print("=" * 60)

print(f"Route ID          : {route['Route_ID']}")
print(
    f"Route             : "
    f"{route['Loading_Port']} -> "
    f"{route['Discharge_Port']}"
)

print(
    f"Distance          : "
    f"{distance:,.0f} NM"
)

print(
    f"Bunkering Port(s) : "
    f"{route['Bunkering_Port']}"
)

print(
    f"Speed             : "
    f"{speed:.1f} knots"
)

print(
    f"Voyage Days       : "
    f"{voyage_days:.2f}"
)

print(
    f"Daily Consumption : "
    f"{daily_consumption:.2f} MT/day"
)

print(
    f"Fuel Required     : "
    f"{fuel_required:.2f} MT"
)

print(
    f"VLSFO Price       : "
    f"${vlsfo_price:.2f}/MT"
)

print(
    f"Total Fuel Cost   : "
    f"${fuel_cost:,.2f}"
)

print("=" * 60)