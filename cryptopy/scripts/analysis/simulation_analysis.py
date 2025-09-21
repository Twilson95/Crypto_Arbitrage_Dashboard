import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


from cryptopy import JsonHelper

matplotlib.use("TkAgg")  # Or another backend like 'Qt5Agg' depending on your system

simulation_names = [
    "long_history_no_predictor",
    # "long_history_all_trades_sensible_parameters",
    "long_history_river_adaptive_random_forest",
    "long_history_river_adaptive_tree",
    "long_history_river_aggregated_mondrian_forest_classifier",
    # "long_history_river_aggregated_mondrian_forest_classifier_2",
    "long_history_river_naive_bayes",
    # "long_history_river_PAClassifier",
    # "long_history_river_PAClassifier_train_with_rejected_trades",
    "long_history_river_perceptron",
    # "long_history_river_perceptron_train_on_failed",
    "long_history_river_predictor+",
]

flattened_data = []
for name in simulation_names:
    path = f"../../../data/simulations/portfolio_sim/{name}.json"
    json_data = JsonHelper.read_from_json(path)
    results = json_data["trade_events"]
    for entry in results:
        flattened_entry = {
            "path": path,
            "pair": entry["pair"],
            "open_date": entry["open_event"]["date"],
            "open_spread": entry["open_event"]["spread_data"]["spread"],
            "open_direction": entry["open_event"]["direction"],
            "open_avg_price_ratio": entry["open_event"]["avg_price_ratio"],
            # "volume_ratio": entry["open_event"]["volume_ratio"],
            # "volatility_ratio": entry["open_event"]["volatility_ratio"],
            "open_stop_loss": entry["open_event"]["stop_loss"],
            "open_expected_profit": entry["open_event"]["expected_profit"],
            "close_date": entry["close_event"]["date"],
            "close_spread": entry["close_event"]["spread_data"]["spread"],
            "close_reason": entry["close_event"]["reason"],
            "profit": entry["profit"],
        }
        flattened_data.append(flattened_entry)

# Convert to DataFrame
df = pd.DataFrame(flattened_data)

df["model_name"] = df["path"].apply(
    lambda x: x.split("/")[-1].replace(".json", "").replace("long_history_", "")
)
print(df.head())

# Ensure datetime type for close_date
df["close_date"] = pd.to_datetime(df["close_date"])
df = df[df["close_date"] > pd.Timestamp("2024-04-09")]
df = df[df["close_date"] < pd.Timestamp("2024-12-09")]

# Sort by date
df = df.sort_values("close_date")

# Group by model and calculate cumulative profit
df["cumulative_profit"] = df.groupby("model_name")["profit"].cumsum()

# Plot
plt.figure(figsize=(12, 6))
for model, group in df.groupby("model_name"):
    plt.plot(group["close_date"], group["cumulative_profit"], label=model)

plt.xlabel("Date", fontsize=14)
plt.ylabel("Cumulative Profit", fontsize=14)
plt.title("Cumulative Profit Over Time by Model", fontsize=16)
plt.legend(title="Model", fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
for model, group in df.groupby("model_name"):
    if model == "no_predictor":
        plt.plot(
            group["close_date"],
            group["cumulative_profit"],
            label=model,
            linewidth=1.6,  # bold
            color="black",  # distinguishable color
        )
    else:
        plt.plot(
            group["close_date"], group["cumulative_profit"], label=model, linewidth=1.5
        )

plt.xlabel("Date", fontsize=16)
plt.ylabel("Cumulative Profit", fontsize=16)
plt.title("Cumulative Profit Over Time by Model", fontsize=16)
plt.legend(title="Model", fontsize=12, loc="upper left")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()

plt.savefig(
    r"C:\Users\thoma\Documents\Learning\Masters\Year 2\Project\cumulative_profit_by_model.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

df["close_reason"] = df["close_reason"].astype(str)

# Row-normalized percentages (one row per model)
perc = pd.crosstab(df["model_name"], df["close_reason"], normalize="index").multiply(
    100
)

# Total number of trades per model
counts = df.groupby("model_name").size().rename("TotalTrades")

# Join and tidy
summary = perc.join(counts).fillna(0)

# Optional: round percentages and sort columns (move TotalTrades to the end)
cols = [c for c in summary.columns if c != "TotalTrades"]
summary = summary[cols + ["TotalTrades"]].round(2)

print("\n=== Close reason breakdown (% of trades) and totals by model ===")
pd.set_option("display.max_columns", None)

print(summary)

# Optional: save to CSV
# summary.to_csv("close_reason_breakdown_by_model.csv", index=True)
print("\nSaved: close_reason_breakdown_by_model.csv")
