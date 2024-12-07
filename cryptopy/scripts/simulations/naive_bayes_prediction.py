from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from cryptopy import JsonHelper
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
# Prepare features and binary target variable
# Binary success label (1 for profitable, 0 for non-profitable)
y_binary = (df["profit"] > 0).astype(int)  # Convert profit to binary classification

# Only use numerical features for Naive Bayes
numeric_features = [
    "p_value",
    "spread_deviation",
    "hedge_ratio",
    "avg_price_ratio",
    "expected_profit",
    "volume_ratio",
    "volatility_ratio",
]
X_numeric = df[numeric_features]  # Use only the numerical features

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_numeric, y_binary, test_size=0.2, random_state=42
)

# Initialize and train Gaussian Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predict success on the test set
y_test_pred = nb_model.predict(X_test)

print("Classification Report for Naive Bayes Success Prediction:")
print(classification_report(y_test, y_test_pred))

accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy on Test Set:", accuracy)
