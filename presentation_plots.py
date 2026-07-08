import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set academic styling for presentation
sns.set_theme(style="whitegrid", context="talk")

def generate_multi_scenario_u_curve():
    """Generates a multi-scenario U-Curve showing the optimal speed shift."""
    speeds = np.linspace(8.0, 14.0, 100)
    distance = 3450.0  # R05 (Ras Tanura -> Paradip)
    fuel_price = 517.0 # $/MT
    
    tce_rates = [20000, 40000, 80000]
    labels = ["Weak Market ($20k/day)", "Base Market ($40k/day)", "Boom Market ($80k/day)"]
    colors = ["#5bc0de", "#f0ad4e", "#d9534f"] 
    
    base_load = 9.96
    c_constant = 0.01496
    
    voyage_days = distance / (speeds * 24.0)
    daily_burn = base_load + (c_constant * (speeds**3))
    total_burn = daily_burn * voyage_days
    fuel_cost = total_burn * fuel_price
    
    plt.figure(figsize=(12, 7))
    
    # REMOVED the grey baseline line to allow the Y-axis to auto-scale and deepen the U-curves

    for tce, label, color in zip(tce_rates, labels, colors):
        time_cost = voyage_days * tce
        total_cost = fuel_cost + time_cost
        
        min_idx = np.argmin(total_cost)
        opt_speed = speeds[min_idx]
        opt_cost = total_cost[min_idx]

        plt.plot(speeds, total_cost, label=label, color=color, linewidth=3)
        plt.scatter([opt_speed], [opt_cost], color=color, s=150, edgecolors='black', zorder=5)
        
        # Adjusted annotation to sit cleanly above the dots to avoid crossing lines
        plt.annotate(f'{opt_speed:.1f} kts', 
                     xy=(opt_speed, opt_cost), 
                     xytext=(opt_speed, opt_cost + 40000), 
                     ha='center',
                     fontsize=12, fontweight='bold', color=color)

    plt.title("Dynamic Speed Optimization: The TCE Shift Effect", fontweight="bold", pad=20)
    plt.xlabel("Transit Speed (Knots)", fontweight="bold")
    plt.ylabel("Total Voyage Cost (USD)", fontweight="bold")
    
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['${:,.0f}'.format(x) for x in current_values])
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("Presentation_Multi_UCurve.png", dpi=300)
    print("Generated: Presentation_Multi_UCurve.png (Cleaned Scale)")


def generate_arbitrage_bars():
    """Generates the Bunkering Strategy reaction to ML predictions."""
    data = {
        'AI Forecast': ['Downstream Market\nin Contango (Cheaper)', 'Flat Market\n(No Price Delta)', 'Downstream Market\nin Backwardation (Spike)'],
        'Stage 1 (Buy Now)': [350, 600, 784], 
        'Stage 2 (Buy Later)': [400, 150, 0]
    }
    df = pd.DataFrame(data)
    
    ax = df.set_index('AI Forecast').plot(
        kind='bar', stacked=True, figsize=(11, 7), color=['#2980b9', '#e67e22'], width=0.5 # Thinned the bars slightly
    )
    
    plt.title("Stochastic Arbitrage: Bunkering Response to AI Forecasts", fontweight="bold", pad=20)
    plt.xlabel("") 
    plt.ylabel("Bunker Purchase (Metric Tons)", fontweight="bold")
    plt.xticks(rotation=0, fontweight="bold")
    plt.legend(["Stage 1 (Immediate Purchase @ Ras Tanura)", "Stage 2 (Downstream Purchase)"], loc='upper right', framealpha=0.9)
    
    plt.axhline(y=784, color='#c0392b', linestyle='--', linewidth=2, label='Tank Capacity Limit')
    plt.text(1, 800, 'Maximum Tank Capacity (784 MT)', color='#c0392b', ha='center', va='bottom', fontweight='bold')
    
    # Adjusted annotation position to point cleanly from the right side
    plt.annotate('Physical Survival Minimum\n(Burn + Safety Reserve)', 
                 xy=(0.28, 350), xytext=(0.6, 200),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                 fontsize=11, color='black', bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))

    plt.tight_layout()
    plt.savefig("Presentation_Arbitrage.png", dpi=300)
    print("Generated: Presentation_Arbitrage.png (Cleaned Annotations)")




if __name__ == "__main__":
    generate_multi_scenario_u_curve()
    generate_arbitrage_bars()