import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python plot_bunker_predictions.py test_predictions.csv")
    sys.exit(1)

csv_file = sys.argv[1]

df = pd.read_csv(csv_file)

required_cols = [
    "timestamp",
    "port",
    "fuel_type",
    "actual_price",
    "lstm_prediction",
    "naive_prediction",
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

df["timestamp"] = pd.to_datetime(df["timestamp"])

groups = df.groupby(["port", "fuel_type"])

for (port, fuel), group in groups:

    group = group.sort_values("timestamp")

    plt.figure(figsize=(14, 7))

    plt.plot(
        group["timestamp"],
        group["actual_price"],
        label="Actual",
        linewidth=2,
    )

    plt.plot(
        group["timestamp"],
        group["lstm_prediction"],
        label="LSTM",
        linewidth=2,
    )

    plt.plot(
        group["timestamp"],
        group["naive_prediction"],
        label="Naive Previous Price",
        linewidth=2,
        linestyle="--",
    )

    plt.title(f"{port} / {fuel}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    filename = (
        f"{port}_{fuel}"
        .replace("/", "_")
        .replace(" ", "_")
        + ".png"
    )

    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved: {filename}")

print("Done.")