import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_thesis_visualizations():
    print("Loading batch results for visualization...")
    try:
        df = pd.read_csv("sensitivity_analysis_results.csv")
    except FileNotFoundError:
        print("[ERROR] 'sensitivity_analysis_results.csv' not found. Run batch_run.py first.")
        return

    # Set academic/professional styling
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Create a directory to save the plots
    import os
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # =========================================================
    # PLOT 1: The Speed Optimization Curve (Speed vs. TCE)
    # Shows how Gurobi pushes the ship faster as time becomes more valuable
    # =========================================================
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df, 
        x="TCE_Rate_USD", 
        y="Optimized_Speed_Knots", 
        hue="Route_ID", 
        marker="o", 
        linewidth=2.5, 
        markersize=8,
        palette="viridis"
    )
    plt.title("Optimal Fleet Speed vs. Opportunity Cost (TCE)", fontweight="bold", pad=15)
    plt.xlabel("Time Charter Equivalent (USD/day)", fontweight="bold")
    plt.ylabel("Optimized Speed (Knots)", fontweight="bold")
    plt.xticks([20000, 40000, 80000], ['Weak Market\n($20k)', 'Base Market\n($40k)', 'Boom Market\n($80k)'])
    plt.legend(title="Voyage Route")
    plt.tight_layout()
    plt.savefig("plots/1_Speed_vs_TCE.png", dpi=300)
    print("Saved: plots/1_Speed_vs_TCE.png")

    # =========================================================
    # PLOT 2: Stage 1 Bunkering Strategy (Purchase vs. Route)
    # Shows the exact tonnage Gurobi decides to buy at the first port
    # =========================================================
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df, 
        x="Route_ID", 
        y="Stage_1_Purchase_MT", 
        hue="TCE_Rate_USD",
        palette="Blues"
    )
    plt.title("Initial Bunker Purchasing Strategy by Route", fontweight="bold", pad=15)
    plt.xlabel("Voyage Route ID", fontweight="bold")
    plt.ylabel("Optimal Stage-1 Purchase (MT)", fontweight="bold")
    plt.legend(title="TCE Rate ($/day)", alignment="left")
    plt.tight_layout()
    plt.savefig("plots/2_Bunkering_Strategy.png", dpi=300)
    print("Saved: plots/2_Bunkering_Strategy.png")

    # =========================================================
    # PLOT 3: The Efficiency Scatter (Cost vs. Speed)
    # Visualizes the financial scale of the optimization decisions
    # =========================================================
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x="Optimized_Speed_Knots", 
        y="Expected_Total_Cost_USD", 
        hue="Route_ID", 
        size="TCE_Rate_USD", 
        sizes=(100, 400),
        alpha=0.8,
        palette="deep"
    )
    plt.title("Financial Scale: Expected Total Cost vs. Speed Matrix", fontweight="bold", pad=15)
    plt.xlabel("Optimized Speed (Knots)", fontweight="bold")
    plt.ylabel("Expected Total Voyage Cost (USD)", fontweight="bold")
    
    # Format Y-axis as currency
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['${:,.0f}'.format(x) for x in current_values])
    
    # Fix legend layout for readability
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("plots/3_Efficiency_Scatter.png", dpi=300, bbox_inches='tight')
    print("Saved: plots/3_Efficiency_Scatter.png")

    print("All visualizations generated successfully! Check the 'plots' folder.")

if __name__ == "__main__":
    generate_thesis_visualizations()