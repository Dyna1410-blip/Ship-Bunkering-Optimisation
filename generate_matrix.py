import pandas as pd
import networkx as nx
import itertools

def generate_full_distance_matrix():
    print("Building Maritime Network Graph...")
    
    # 1. Initialize the Graph
    G = nx.Graph()

    # 2. Define known adjacent maritime distances (Edges in Nautical Miles)
    # These base distances allow the algorithm to route around landmasses natively
    maritime_links = [
        # --- The Gulf Coast ---
        ("Basrah", "Mina Saud / Ras Al Zour", 120),
        ("Mina Saud / Ras Al Zour", "Ras Tanura", 150),
        ("Ras Tanura", "Siri Island", 180),
        ("Siri Island", "Lavan Island", 60),
        ("Lavan Island", "Fujairah", 200), # Through Strait of Hormuz
        
        # --- The Arabian Sea Crossing (Gulf to West India) ---
        ("Fujairah", "Mundra", 950),
        ("Fujairah", "Vadinar", 980),
        ("Fujairah", "Mumbai", 1000),
        ("Fujairah", "Mumbai (Jawahar Dweep)", 1000),
        ("Fujairah", "Kochi", 1450),
        
        # --- The Indian Coast (West to East) ---
        ("Mundra", "Vadinar", 50),
        ("Vadinar", "Mumbai", 260),
        ("Vadinar", "Mumbai (Jawahar Dweep)", 260),
        ("Mumbai", "Mumbai (Jawahar Dweep)", 5), # Practically same location
        ("Mumbai", "Kochi", 600),
        ("Mumbai (Jawahar Dweep)", "Kochi", 600),
        ("Kochi", "Chennai", 750),           # Around Sri Lanka
        ("Kochi", "Chennai (Ennore)", 760),
        ("Chennai", "Chennai (Ennore)", 15),
        ("Chennai", "Vishakhapatnam (Vizag)", 350),
        ("Chennai (Ennore)", "Vishakhapatnam (Vizag)", 340),
        ("Vishakhapatnam (Vizag)", "Paradip", 250),
        ("Paradip", "Haldia", 150)
    ]

    # Add the edges to the graph
    for p1, p2, dist in maritime_links:
        G.add_edge(p1, p2, weight=dist)

    # 3. Calculate shortest paths for ALL combinations
    print("Executing Dijkstra's Algorithm for all port combinations...")
    all_ports = list(G.nodes())
    
    # Generate every unique pair of ports
    port_pairs = list(itertools.combinations(all_ports, 2))
    
    matrix_data = []
    
    for p1, p2 in port_pairs:
        try:
            # Find the shortest physical maritime path between the two ports
            shortest_dist = nx.dijkstra_path_length(G, p1, p2, weight='weight')
            
            # Store in both directions so lookups never fail
            matrix_data.append({"Origin": p1, "Destination": p2, "Distance_NM": shortest_dist})
            matrix_data.append({"Origin": p2, "Destination": p1, "Distance_NM": shortest_dist})
            
        except nx.NetworkXNoPath:
            print(f"Warning: No physical sea path found between {p1} and {p2}")

    # 4. Export to CSV
    df_matrix = pd.DataFrame(matrix_data)
    
    # Save it to your raw data folder
    output_path = "data/raw/full_distance_matrix.csv"
    
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_matrix.to_csv(output_path, index=False)
    
    print(f"Success! Generated a dense matrix with {len(df_matrix)} routing combinations.")
    print(f"File saved to: {output_path}")

if __name__ == "__main__":
    generate_full_distance_matrix()