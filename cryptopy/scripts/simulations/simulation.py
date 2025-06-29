import pandas as pd

from cryptopy import PortfolioManager, JsonHelper, ArbitrageSimulator, RiverPredictor
from cryptopy.scripts.simulations.simulation_helpers import get_combined_df_of_data

simulation_name = "long_history_all_trades"
exchange_name = "Kraken"
historic_data_folder = f"../../../data/historical_data/{exchange_name}_long_history/"
cointegration_pairs_path = f"../../../data/historical_data/cointegration_pairs.csv"
simulation_path = f"../../../data/simulations/portfolio_sim/{simulation_name}.json"

parameters = {
    "days_back": 100,  # hedge ratio and p_value based off this
    "rolling_window": 30,  # controls moving avg for mean and thresholds
    "p_value_open_threshold": 0.03,  # optimised, maximizes opportunities while keeping success rate
    "p_value_close_threshold": 1,  # optimised
    "expiry_days_threshold": 30,  # optimised, try 15 for portfolio to allow more trades
    "spread_threshold": 1.8,  # optimised 1.8 - 2
    "spread_limit": 3,  # optimised at 3-4
    "hedge_ratio_positive": True,
    "stop_loss_multiplier": 1.5,  # optimised 1.5-1.8
    "max_coin_price_ratio": 50,  # default 50
    "max_concurrent_trades": 999,  # default 12
    "min_expected_profit": 0.0025,  # must expect at least half a percent of the portfolio amount
    "max_expected_profit": 0.025,  # no more at risk as 5% percent of the portfolio amount
    "trade_size": 0.05,  # proportion of portfolio bought in each trade - default 0.06
    "trade_size_same_risk": True,
    "volume_period": 30,  # default 30
    "volume_threshold": 999,  # default 2
    "volatility_period": 30,  # default 30
    "volatility_threshold": 999,  # default 1.5
    "max_each_coin": 999,  # default 3
    "use_ml_predictor": False,
    "trades_before_predictions": 100,  # default 100
    "trend_parameters": {
        "short_window": 30,
        "long_window": 100,
        "change_threshold": 0.01,
    },
}

folder_path = "../../../data/historical_data/Kraken_long_history"
price_df = get_combined_df_of_data(folder_path, "close")
volume_df = get_combined_df_of_data(folder_path, "volume")

pair_combinations_df = pd.read_csv(cointegration_pairs_path)
pair_combinations = list(pair_combinations_df.itertuples(index=False, name=None))

portfolio_manager = PortfolioManager(
    parameters["max_concurrent_trades"],
    funds=1000,
    max_each_coin=parameters["max_each_coin"],
)

if parameters["use_ml_predictor"]:
    river_predictor = RiverPredictor(prediction_threshold=0.5)
else:
    river_predictor = None

arbitrage_simulator = ArbitrageSimulator(
    parameters,
    price_df,
    volume_df,
    portfolio_manager,
    pair_combinations,
    ml_model=river_predictor,
    trades_before_prediction=parameters["trades_before_predictions"],
)
trade_results, cumulative_profit = arbitrage_simulator.run_simulation()

total_profit = sum(result["profit"] for result in trade_results)
number_of_trades = len(trade_results)
positive_trades = len([trade for trade in trade_results if trade["profit"] > 0])
successful_trades = len(
    [
        trade
        for trade in trade_results
        if trade["close_event"]["reason"] == "crossed_mean"
    ]
)

print(f"Total Expected Profit: {total_profit:.2f}")
simulation_data = {
    "parameters": parameters,
    "stats": {
        "total_profit": total_profit,
        "success_rate": successful_trades / number_of_trades,
        "positive_results": positive_trades / number_of_trades,
        "number_of_trades": number_of_trades,
    },
    "trade_events": trade_results,
}
JsonHelper.save_to_json(simulation_data, simulation_path)
