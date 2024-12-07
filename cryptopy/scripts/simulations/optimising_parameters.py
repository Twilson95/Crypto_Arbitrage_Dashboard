from cryptopy import JsonHelper
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Load data
simulation_name = "long_history_unconstrained_parameter_expire_15"
simulation_path = f"../../../data/simulations/all_trades/{simulation_name}.json"
data = JsonHelper.read_from_json(simulation_path)

# Parse data into a DataFrame
rows = []
for trade in data["trade_events"]:
    open_event = trade["open_event"]
    close_event = trade["close_event"]
    profit = trade["profit"]
    row = {
        "coin_1": trade["pair"][0].split("/")[0],
        "coin_2": trade["pair"][1].split("/")[0],
        "p_value": open_event["p_value"],
        "spread_deviation": open_event["spread_data"]["spread_deviation"],
        "hedge_ratio": open_event["hedge_ratio"],
        "direction": 1 if open_event["direction"] == "long" else 0,
        "avg_price_ratio": open_event["avg_price_ratio"],
        "expected_profit": open_event["expected_profit"],
        "volume_ratio": open_event["volume_ratio"],
        "volatility_ratio": open_event["volatility_ratio"],
        "profit": trade["profit"],
    }
    rows.append(row)

df = pd.DataFrame(rows)
print(df.columns)

X = df.drop(columns="profit")
y = df["profit"]

numeric_features = [
    "p_value",
    "spread_deviation",
    "hedge_ratio",
    "avg_price_ratio",
    "expected_profit",
    "volume_ratio",
    "volatility_ratio",
]
categorical_features = ["coin_1", "coin_2", "direction"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

pipeline = Pipeline(
    [("preprocessor", preprocessor), ("regressor", RandomForestRegressor())]
)

param_space = {
    "regressor__n_estimators": Integer(50, 300),
    "regressor__max_depth": Integer(3, 15),
    "regressor__max_features": Real(0.1, 0.9),
    "regressor__min_samples_split": Integer(2, 10),
    "regressor__min_samples_leaf": Integer(1, 10),
}


def profit_stability_scorer(estimator, X, y):
    profit_preds = estimator.predict(X)
    expected_profit = np.mean(profit_preds)
    profit_volatility = np.std(profit_preds)
    stability_score = expected_profit / (
        profit_volatility + 1e-5
    )  # Add a small term to avoid division by zero
    return stability_score


opt = BayesSearchCV(
    estimator=pipeline,
    search_spaces=param_space,
    n_iter=50,
    # scoring="neg_mean_squared_error",
    scoring=profit_stability_scorer,
    cv=3,
    n_jobs=-1,
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit the optimizer to the training data
opt.fit(X_train, y_train)

print("Best hyperparameters for maximizing profit prediction:", opt.best_params_)
print("Best cross-validated score (MSE) with these hyperparameters:", -opt.best_score_)

y_pred = opt.predict(X_test)
print(classification_report(y_test, y_pred))
