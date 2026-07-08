import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set academic styling
sns.set_theme(style="whitegrid", context="talk")

def plot_continuous_speed_frontier():
    # 1. Generate 500 continuous market steps from $10k to $80k
    tce_range = np.linspace(10000, 80000, 500)
    
    # Current Market & Physics Variables
    p_fuel = 517.0 
    base_load = 9.96
    c_const = 0.01496
    
    # 2. Apply the Analytical Derivative Formula
    numerator = (p_fuel * base_load) + tce_range
    denominator = 2 * p_fuel * c_const
    v_unrestricted = np.cbrt(numerator / denominator)
    
    # 3. Apply Physical Constraints (The ship cannot exceed 14 knots or drop below 8)
    v_actual = np.clip(v_unrestricted, 8.0, 14.0)

    # 4. Plotting
    plt.figure(figsize=(11, 6))
    
    # Plot Theoretical vs Actual
    plt.plot(tce_range, v_unrestricted, label="Theoretical Unrestricted Speed", color="grey", linestyle="--", linewidth=2)
    plt.plot(tce_range, v_actual, label="Actual Vessel Speed", color="#2980b9", linewidth=4)
    
    # Add physical boundary lines
    plt.axhline(14.0, color='#c0392b', linestyle=':', linewidth=2, label='Maximum Engine Capacity (14.0 kts)')
    
    plt.title("Continuous Speed Frontier: Navigating Engine Limits", fontweight="bold", pad=20)
    plt.xlabel("Time Charter Equivalent (USD/day)", fontweight="bold")
    plt.ylabel("Optimal Speed (Knots)", fontweight="bold")
    
    # Format X-axis as currency
    current_values = plt.gca().get_xticks()
    plt.gca().set_xticklabels(['${:,.0f}'.format(x) for x in current_values])
    
    # Focus the Y-axis to highlight the curve
    plt.ylim(9, 19)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig("Presentation_Continuous_Frontier.png", dpi=300)
    print("Generated: Presentation_Continuous_Frontier.png")

if __name__ == "__main__":
    plot_continuous_speed_frontier()