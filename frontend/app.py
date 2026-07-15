import os 
import requests
import streamlit as st
import pandas as pd

# --- DYNAMIC DATA INGESTION ---
@st.cache_data
def load_route_data(filepath="data/raw/full_distance_matrix.csv"):
    """
    Reads the full_distance_matrix.csv file and builds a universal distance lookup matrix.
    Allows A -> B -> C unrestricted routing.
    """
    try:
        df = pd.read_csv(filepath)
        
        # 1. Extract ALL unique ports into one master list
        all_ports = pd.concat([df['Origin'], df['Destination']]).dropna().unique().tolist()
        all_ports.sort()
        
        # 2. Build a nested dictionary for fast distance lookups
        distance_matrix = {}
        for index, row in df.iterrows():
            p1 = row['Origin']          
            p2 = row['Destination']     
            dist = row['Distance_NM']   
            
            if pd.isna(p1) or pd.isna(p2):
                continue
                
            # Initialize sub-dictionaries
            if p1 not in distance_matrix: distance_matrix[p1] = {}
            if p2 not in distance_matrix: distance_matrix[p2] = {}
            
            # Populate bidirectionally 
            distance_matrix[p1][p2] = float(dist)
            distance_matrix[p2][p1] = float(dist)
            
        return all_ports, distance_matrix
        
    except FileNotFoundError:
        st.error(f"Could not find '{filepath}'. Please ensure it is in the same folder as app.py.")
        return [], {}
    except KeyError as e:
        st.error(f"Column missing in CSV: {e}. Please ensure it matches your full_distance_matrix file.")
        return [], {}

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="VLCC Digital Twin", layout="wide", page_icon="🚢")

custom_css = """
<style>
    /* Import Merriweather from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;0,900;1,300;1,400;1,700;1,900&display=swap');

    /* Apply the font to the entire app */
    html, body, [class*="css"]  {
        font-family: 'Merriweather', serif;
    }
    
    /* Ensure headers also use the font */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Merriweather', serif !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("🚢 Vessel Bunkering Optimization ")
st.markdown("### Stochastic Routing & Bunkering Optimization Engine")
st.write("Dynamically route the vessel, set inventory boundaries, and execute spatial arbitrage.")
st.divider()

# Load the data dynamically
all_ports, distance_matrix = load_route_data("data/raw/full_distance_matrix.csv")

# --- SIDEBAR INPUTS (The Control Room) ---
st.sidebar.header("Geographic Parameters")

if all_ports:
    origin_port = st.sidebar.selectbox("Point A: Current Ship Location", all_ports, index=0)
    load_port = st.sidebar.selectbox("Point B: Intermediate Load Port", all_ports, index=min(1, len(all_ports)-1))
    dest_port = st.sidebar.selectbox("Point C: Final Destination", all_ports, index=0)
else:
    st.sidebar.warning("No ports loaded. Check your CSV file.")
    origin_port, load_port, dest_port = None, None, None

st.sidebar.divider()
st.sidebar.header("Inventory Parameters")

current_bunker = st.sidebar.number_input(
    "Current Fuel Onboard (MT)", 
    min_value=0.0, max_value=2500.0, value=200.0, step=1.0,
    help="Fuel currently sitting in the tanks at the origin port."
)

target_inventory = st.sidebar.number_input(
    "Target End Inventory (MT)", 
    min_value=0.0, max_value=5000.0, value=200.0, step=10.0,
    help="Fuel required to remain in tanks upon arriving at the final destination for the next charter."
)

st.sidebar.divider()
st.sidebar.header("Economic Parameters")

# Toggle between auto-calculated or user-defined TCE override
tce_mode = st.sidebar.radio("TCE Rate Mode", ["Auto-Calculate", "Manual Input"])

if tce_mode == "Manual Input":
    manual_tce = st.sidebar.number_input(
        "Target TCE (USD/day)", 
        min_value=0.0, 
        value=34260.0, 
        step=500.0,
        help="Manually override the optimization pipeline calculation with a static charter rate."
    )
else:
    manual_tce = None

st.sidebar.divider()
run_button = st.sidebar.button("Execute Optimization", use_container_width=True, type="primary")

# --- DISTANCE CALCULATIONS ---
def get_distance(p1, p2):
    try:
        return distance_matrix[p1][p2]
    except KeyError:
        return None

# --- MAIN DASHBOARD AREA ---
if run_button and origin_port and load_port and dest_port:
    # Look up the distances dynamically from the CSV data
    dist_leg1 = get_distance(origin_port, load_port)
    dist_leg2 = get_distance(load_port, dest_port)
    
    if dist_leg1 is None or dist_leg2 is None:
        st.error(f"Distance data is missing in your 'full_distance_matrix.csv' for the selected combination. The dataset might not contain a direct path between these specific ports.")
    else:
        with st.spinner("Connecting to FastAPI Optimization Engine..."):
            try:
                # 1. Define where the API lives 
                api_url = os.getenv("API_URL", "http://localhost:8000/api/v1/optimize")
                
                # 2. Package the Streamlit UI inputs into a JSON payload
                payload = {
                    "origin": origin_port,
                    "load_port": load_port,
                    "destination": dest_port,
                    "current_bunker_onboard": current_bunker,
                    "target_return_inventory": target_inventory,
                    "manual_tce": manual_tce
                }
                
                # 3. Send the POST request to the backend
                response = requests.post(api_url, json=payload)
                response.raise_for_status() 
                
                # 4. Extract the Gurobi directives
                api_data = response.json()
                results = api_data["directives"]
                
                if results["status"] == "OPTIMAL":
                    st.success(f"Optimization Solved. Total Loop Distance: {dist_leg1 + dist_leg2} NM")
                    
                    # --- ROW 1: TOP LEVEL METRICS ---
                    st.subheader(f"Node 1 Action Plan (Immediate Execution in {origin_port})")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    col1.metric("Node-1 Purchase (MT)", f"{results['immediate_bunker_node1_mt']:,.2f}")
                    col2.metric("CapEx Required (USD)", f"${results['first_stage_cost_usd']:,.2f}")
                    col3.metric("Expected Loop Cost (USD)", f"${results['expected_total_loop_cost_usd']:,.2f}")
                    col4.metric("Target Return Inv (MT)", f"{results['terminal_target_mt']:,.0f}")
                    
                    st.divider()
                    
                    # --- ROW 2: STOCHASTIC CONTINGENCY MATRIX ---
                    st.subheader(f"🌐 Node 2 Recourse Strategy (Hedging Contingencies at {load_port})")
                    
                    tab1, tab2, tab3 = st.tabs(["📉 Low Price Market (S_Low)", "⚖️ Base Market (S_Base)", "📈 Spike Market (S_High)"])
                    
                    def render_scenario(s_key, tab):
                        s_data = results["scenarios"][s_key]
                        with tab:
                            st.markdown(f"**Secure at {load_port}:** `{s_data['bunker_gulf_node2_mt']:,.2f} MT`")
                            
                            physics_df = pd.DataFrame({
                                "Voyage Leg": [f"Leg 1: {origin_port} ➔ {load_port} (Ballast)", f"Leg 2: {load_port} ➔ {dest_port} (Laden)"],
                                "Distance (NM)": [dist_leg1, dist_leg2],
                                "Speed (Knots)": [s_data.get('leg1_ballast', {}).get('speed_knots', 0), s_data.get('leg2_laden', {}).get('speed_knots', 0)],
                                "Duration (Days)": [s_data.get('leg1_ballast', {}).get('duration_days', 0), s_data.get('leg2_laden', {}).get('duration_days', 0)],
                                "Fuel Burn (MT)": [s_data.get('leg1_ballast', {}).get('burn_mt', 0), s_data.get('leg2_laden', {}).get('burn_mt', 0)]
                            })
                            st.table(physics_df.style.format(precision=2))
                            st.info(f"⏱️ **Loop Time Opportunity Cost (TCE):** ${s_data['loop_opportunity_cost_usd']:,.2f}")

                    render_scenario("S_Low", tab1)
                    render_scenario("S_Base", tab2)
                    render_scenario("S_High", tab3)
                    
                else:
                    st.error("Solver failed. The Target End Inventory might be too high for the tank capacity, or the Current Fuel is too low to survive Leg 1.")
            
            except requests.exceptions.RequestException as e:
                st.error(f"Backend Engine Connection Failed: {e}")
                st.warning("Ensure the FastAPI server is running and accessible.")
            except Exception as e:
                st.error(f"UI Rendering Error: {e}")
else:
    if not run_button:
        st.info("Kindly Select your routing and inventory parameters in the sidebar, then click **Execute Optimization**.")