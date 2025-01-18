import pandas as pd
from cryptopy import JsonHelper
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

simulation_name = "long_history_all_trades_sensible_parameters"
simulation_path = f"../../../data/simulations/portfolio_sim/{simulation_name}.json"
# simulation_path = f"../../../data/simulations/all_trades/{simulation_name}.json"

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
        # "expected_profit": open_event["expected_profit"],
        "volume_ratio": open_event["volume_ratio"],
        "volatility_ratio": open_event["volatility_ratio"],
        "market_trend": open_event["market_trend"],
        "coin_1_trend": open_event["coin_1_trend"],
        "coin_2_trend": open_event["coin_2_trend"],
        # "target": 1 if profit > 0 else 0,
        "target": 1 if close_event["reason"] == "crossed_mean" else 0,
        "profit": profit,
    }
    rows.append(row)

df = pd.DataFrame(rows)
df["market_coin_1_match"] = df["market_trend"] == df["coin_1_trend"]
df["market_coin_2_match"] = df["market_trend"] == df["coin_2_trend"]
df["coin_1_coin_2_match"] = df["coin_1_trend"] == df["coin_2_trend"]

X = df.drop(columns=["target", "profit"])
y = df["target"]

numeric_features = [
    "p_value",
    "spread_deviation",
    "hedge_ratio",
    "avg_price_ratio",
    # "expected_profit",
    "volume_ratio",
    "volatility_ratio",
]
# categorical_features = ["coin_1", "coin_2", "direction"]
categorical_features = [
    "direction",
    "market_trend",
    "coin_1_trend",
    "coin_2_trend",
    "market_coin_1_match",
    "market_coin_2_match",
    "coin_1_coin_2_match",
]
# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(),
        ),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)
print(X_test)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

df_test = df.loc[X_test.index].copy()
df_test["predicted"] = y_pred
total_profit = df_test.loc[df_test["predicted"] == 1, "profit"].sum()
print(total_profit)


rf_model = pipeline.named_steps["classifier"]
tree = rf_model.estimators_[0]  # You can change the index to view other trees

# Plot the best tree
plt.figure(figsize=(20, 10))
plot_tree(
    tree,
    feature_names=numeric_features
    + list(
        pipeline.named_steps["preprocessor"]
        .transformers_[1][1]
        .get_feature_names_out(categorical_features)
    ),
    class_names=["Not Successful", "Successful"],
    filled=True,
    rounded=True,
    max_depth=4,  # Limit the depth for readability
)
plt.title("Best Decision Tree from RandomForestClassifier")
plt.show()

from dtreeviz.trees import model

# Assuming `tree` is one of the estimators from the random forest
viz = model(
    tree,
    X_train,
    y_train,
    target_name="Success",
    feature_names=numeric_features
    + list(
        pipeline.named_steps["preprocessor"]
        .transformers_[1][1]
        .get_feature_names_out(categorical_features)
    ),
    class_names=["Not Successful", "Successful"],
)
# viz.show()
