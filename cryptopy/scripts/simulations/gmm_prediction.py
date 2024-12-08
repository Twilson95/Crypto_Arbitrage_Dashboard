from cryptopy import JsonHelper
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report
import pandas as pd

# Load data
simulation_name = "long_history_unconstrained_parameter_expire_15"
simulation_path = f"../../../data/simulations/all_trades/{simulation_name}.json"
data = JsonHelper.read_from_json(simulation_path)

rows = []
for trade in data["trade_events"]:
    open_event = trade["open_event"]
    close_event = trade["close_event"]

    profit = trade["profit"]
    row = {
        "coin_1": trade["pair"][0].split("/")[0],  # First coin
        "coin_2": trade["pair"][1].split("/")[0],  # Second coin
        # "spread": open_event["spread_data"]["spread"],
        # "spread_mean": open_event["spread_data"]["spread_mean"],
        # "spread_std": open_event["spread_data"]["spread_std"],
        # "upper_threshold": open_event["spread_data"]["upper_threshold"],
        # "lower_threshold": open_event["spread_data"]["lower_threshold"],
        "p_value": open_event["p_value"],
        "spread_deviation": open_event["spread_data"]["spread_deviation"],
        "hedge_ratio": open_event["hedge_ratio"],
        "direction": 1 if open_event["direction"] == "long" else 0,
        "avg_price_ratio": open_event["avg_price_ratio"],
        # "stop_loss": open_event["stop_loss"],
        "expected_profit": open_event["expected_profit"],
        "volume_ratio": open_event["volume_ratio"],
        "volatility_ratio": open_event["volatility_ratio"],
        # "target": 1 if profit > 0 else 0,
        # "target": 1 if close_event["reason"] == "crossed_mean" else 0,
        "profit": trade["profit"],
    }
    rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)
print(df.columns)

X = df.drop(columns="profit")
y = df["profit"]

# Select numerical features for GMM
numeric_features = [
    "p_value",
    "spread_deviation",
    "hedge_ratio",
    "avg_price_ratio",
    "expected_profit",
    "volume_ratio",
    "volatility_ratio",
]

X_numeric = df[numeric_features]  # Only the numerical features
y_binary = (df["profit"] > 0).astype(
    int
)  # Binary success label (1 for profitable, 0 for non-profitable)

# Fit a GMM with 2 clusters (assuming 2 clusters for successful vs. unsuccessful)
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X_numeric)

# Predict the clusters for the training data
df["cluster"] = gmm.predict(X_numeric)

# Calculate average profit for each cluster to label them as successful or unsuccessful
cluster_avg_profit = df.groupby("cluster")["profit"].mean()
print("Average profit per cluster:\n", cluster_avg_profit)

# Label each cluster as successful (1) or unsuccessful (0) based on average profit
# Assume cluster with higher mean profit is "successful"
successful_cluster = cluster_avg_profit.idxmax()
df["predicted_success"] = df["cluster"].apply(
    lambda x: 1 if x == successful_cluster else 0
)

# Classification report for GMM-based predictions
print("Classification Report for GMM-based Success Prediction:")
print(classification_report(y_binary, df["predicted_success"]))
