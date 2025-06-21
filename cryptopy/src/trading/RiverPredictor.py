from river import (
    compose,
    preprocessing,
    linear_model,
    feature_extraction,
    naive_bayes,
    tree,
    forest,
    ensemble,
)
import copy


class RiverPredictor:
    def __init__(self, prediction_threshold):
        # self.cat_features = ["coin_1", "coin_2"]
        self.num_features = [
            "p_value",
            "spread_deviation",
            "hedge_ratio",
            "direction",
            "avg_price_ratio",
            "expected_profit",
            "volume_ratio",
            "volatility_ratio",
        ]
        self.prediction_threshold = prediction_threshold

        self.model = (
            # feature_extraction.HashingEncoder(on=["coin_1", "coin_2"], n_features=64)
            preprocessing.StandardScaler()
            |
            # models
            # linear_model.LogisticRegression()
            # linear_model.Perceptron()
            # linear_model.PAClassifier()
            # naive_bayes.GaussianNB()
            # tree.HoeffdingTreeClassifier()
            # tree.HoeffdingAdaptiveTreeClassifier()
            # forest.ARFClassifier()
            forest.AMFClassifier()
        )

    def learn_from_data(self, trade_data):
        """
        Incrementally update the model with new training data.
        trade_data: A list of (X, y) or a structure like daily_trade_results
                    where each element has the needed features and target.
        """
        for trade in trade_data:
            open_event = trade["open_event"]
            features = {
                # "coin_1": trade["pair"][0].split("/")[0],
                # "coin_2": trade["pair"][1].split("/")[0],
                "p_value": open_event.get("p_value", 0),
                "spread_deviation": open_event["spread_data"]["spread_deviation"],
                "hedge_ratio": open_event["hedge_ratio"],
                "direction": 1 if open_event["direction"] == "long" else 0,
                "avg_price_ratio": open_event["avg_price_ratio"],
                "expected_profit": open_event["expected_profit"],
                "volume_ratio": open_event["volume_ratio"],
                "volatility_ratio": open_event["volatility_ratio"],
            }

            # y = 1 if trade["profit"] > 0 else 0
            y = 1 if trade["close_event"]["reason"] == "crossed_mean" else 0

            self.model.learn_one(features, y)

    def predict_opportunity(self, opportunity):
        """
        Predict whether a given opportunity is likely to succeed.
        opportunity: dict similar to a daily opportunity, containing "pair" and "open_event".
        """
        open_event = opportunity["open_event"]

        features = {
            # "coin_1": opportunity["pair"][0].split("/")[0],
            # "coin_2": opportunity["pair"][1].split("/")[0],
            "p_value": open_event.get("p_value", 0),
            "spread_deviation": open_event["spread_data"]["spread_deviation"],
            "hedge_ratio": open_event["hedge_ratio"],
            "direction": 1 if open_event["direction"] == "long" else 0,
            "avg_price_ratio": open_event["avg_price_ratio"],
            "expected_profit": open_event["expected_profit"],
            "volume_ratio": open_event["volume_ratio"],
            "volatility_ratio": open_event["volatility_ratio"],
        }

        proba = self.model.predict_proba_one(features)
        print(f"Predicted probability of success: {proba[0]} {proba[1]:.2f}")
        return proba[1] > self.prediction_threshold, proba[1]
