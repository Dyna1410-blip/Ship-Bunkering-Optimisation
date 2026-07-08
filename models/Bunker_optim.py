# bunkering_optimizer.py

import pandas as pd
from fuel_model import DeshViraatFuelModel

# ===================================
# LOAD DATA
# ===================================

routes = pd.read_csv("Datasets/routes_cl.csv")
prices = pd.read_csv("Datasets/port_fuel_prices.csv")

# ===================================
# USER INPUTS
# ===================================

route_id = input(
    "Enter Route ID (e.g. R05): "
).strip()

speed = float(
    input("Enter vessel speed (knots): ")
)

# ===================================
# GET ROUTE
# ===================================

route = routes[
    routes["Route_ID"] == route_id
]

if route.empty:
    print("Route not found.")
    exit()

route = route.iloc[0]

# ===================================
# FUEL REQUIREMENT
# ===================================

model = DeshViraatFuelModel()

fuel_required = model.fuel_required(
    route["Distance_Nautical_Miles"],
    speed
)

# ===================================
# GET LATEST VLSFO PRICES
# ===================================

vlsfo = prices[
    prices["fuel_type"].str.upper() == "VLSFO"
].copy()

vlsfo["timestamp"] = pd.to_datetime(
    vlsfo["timestamp"]
)

latest = (
    vlsfo
    .sort_values("timestamp")
    .groupby("port")
    .tail(1)
)

price_map = dict(
    zip(
        latest["port"],
        latest["price"]
    )
)

# ===================================
# BUNKERING PORTS
# ===================================

candidate_ports = [
    p.strip()
    for p in route["Bunkering_Port"].split("+")
]

candidate_ports = [
    p for p in candidate_ports
    if p in price_map
]

if len(candidate_ports) == 0:
    print(
        "No valid bunker prices found."
    )
    exit()

# ===================================
# STRATEGY 1
# Cheapest Port
# ===================================

cheapest_port = min(
    candidate_ports,
    key=lambda x: price_map[x]
)

cost_cheapest = (
    fuel_required *
    price_map[cheapest_port]
)

# ===================================
# STRATEGY 2
# Equal Split
# ===================================

cost_split = 0

for port in candidate_ports:

    qty = (
        fuel_required /
        len(candidate_ports)
    )

    cost_split += (
        qty *
        price_map[port]
    )

# ===================================
# STRATEGY 3
# Most Expensive Port
# ===================================

expensive_port = max(
    candidate_ports,
    key=lambda x: price_map[x]
)

cost_expensive = (
    fuel_required *
    price_map[expensive_port]
)

# ===================================
# RESULTS
# ===================================

print("\n")
print("=" * 70)
print("BUNKERING STRATEGY ANALYSIS")
print("=" * 70)

print(
    f"Route: "
    f"{route['Loading_Port']} -> "
    f"{route['Discharge_Port']}"
)

print(
    f"Distance: "
    f"{route['Distance_Nautical_Miles']} NM"
)

print(
    f"Fuel Required: "
    f"{fuel_required:.2f} MT"
)

print("\nCandidate Ports")

for port in candidate_ports:

    print(
        f"{port:15s} "
        f"${price_map[port]:.2f}/MT"
    )

print("\n")

print(
    f"Strategy 1 "
    f"(All at Cheapest Port - {cheapest_port})"
)

print(
    f"Cost = "
    f"${cost_cheapest:,.2f}"
)

print("\n")

print(
    "Strategy 2 "
    "(50/50 Split)"
)

print(
    f"Cost = "
    f"${cost_split:,.2f}"
)

print("\n")

print(
    f"Strategy 3 "
    f"(All at Most Expensive Port - {expensive_port})"
)

print(
    f"Cost = "
    f"${cost_expensive:,.2f}"
)

print("\n")

print(
    f"Savings vs Worst Case = "
    f"${cost_expensive - cost_cheapest:,.2f}"
)

print("=" * 70)