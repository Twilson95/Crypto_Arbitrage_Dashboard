import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report
from cryptopy import JsonHelper

simulation_name = "long_history_all_trades_sensible_parameters"
simulation_path = f"../../../data/simulations/portfolio_sim/{simulation_name}.json"
# simulation_path = f"../../../data/simulations/all_trades/{simulation_name}.json"

data = JsonHelper.read_from_json(simulation_path)
# ----------------------------------------------------
# 1) READ YOUR DATA (simulated example)
# ----------------------------------------------------

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
        "volume_ratio": open_event["volume_ratio"],
        "volatility_ratio": open_event["volatility_ratio"],
        "market_trend": open_event["market_trend"],
        "coin_1_trend": open_event["coin_1_trend"],
        "coin_2_trend": open_event["coin_2_trend"],
        "target": 1 if close_event["reason"] == "crossed_mean" else 0,
        "profit": profit,
    }
    rows.append(row)

df = pd.DataFrame(rows)

# ----------------------------------------------------
# 2) FEATURE ENGINEERING (Example: match vs. mismatch)
# ----------------------------------------------------
df["market_coin_1_match"] = df["market_trend"] == df["coin_1_trend"]
df["market_coin_2_match"] = df["market_trend"] == df["coin_2_trend"]
df["coin_1_coin_2_match"] = df["coin_1_trend"] == df["coin_2_trend"]

# ----------------------------------------------------
# 3) TRAIN-TEST SPLIT
# ----------------------------------------------------
X = df.drop(columns=["target", "profit"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------------------------------
# 4) BUILD PIPELINE
# ----------------------------------------------------
numeric_features = [
    "p_value",
    "spread_deviation",
    "hedge_ratio",
    "avg_price_ratio",
    "volume_ratio",
    "volatility_ratio",
]

# Decide whether to treat match features as numeric (0/1) or categorical (True/False)
# Here we treat them as categorical booleans:
categorical_features = [
    "direction",
    "market_trend",
    "coin_1_trend",
    "coin_2_trend",
    "market_coin_1_match",
    "market_coin_2_match",
    "coin_1_coin_2_match",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# ----------------------------------------------------
# 5) DEFINE PARAMETER GRID FOR RANDOM FOREST
#    (Adjust as needed)
# ----------------------------------------------------
param_grid = {
    "classifier__n_estimators": [200],
    "classifier__max_depth": [None, 10, 15, 20, 25],
    "classifier__min_samples_split": [2, 3, 4],
    "classifier__min_samples_leaf": [3, 4, 5, 6, 7],
    "classifier__max_features": ["sqrt", 0.7, 0.9],
    "classifier__bootstrap": [True],
    "classifier__class_weight": [None],
    "classifier__criterion": ["gini"],
}

# ----------------------------------------------------
# 6) STRATIFIEDKFold + GRIDSEARCH
# ----------------------------------------------------
# We'll do 5-fold stratified cross-validation on X_train
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring="accuracy",  # or "f1", "precision", etc.
    cv=skf,
    n_jobs=-1,
)

# ----------------------------------------------------
# 7) FIT GRIDSEARCH ON TRAIN DATA
# ----------------------------------------------------
grid_search.fit(X_train, y_train)

print("Best CV Score: ", grid_search.best_score_)
print("Best Params: ", grid_search.best_params_)

# Retrieve the best model (pipeline) found
best_model = grid_search.best_estimator_

# ----------------------------------------------------
# 8) EVALUATE ON TEST DATA
# ----------------------------------------------------
y_pred = best_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------------------------------
# 9) CALCULATE PROFIT ON TEST SET
# ----------------------------------------------------
# Match the rows from df that went to X_test
df_test = df.loc[X_test.index].copy()
df_test["predicted"] = y_pred

# Sum the profit where predicted == 1
total_profit = df_test.loc[df_test["predicted"] == 1, "profit"].sum()
print("Total profit for predicted trades on the test set:", total_profit)
