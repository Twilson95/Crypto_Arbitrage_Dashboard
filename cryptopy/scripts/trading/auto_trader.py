import yaml
import pandas as pd
import ccxt
import time
from cryptopy import DataManager, DataFetcher, PortfolioManager, CointegrationCalculator


with open("cryptopy/config/trading_config.yaml", "r") as f:
    exchange_config = yaml.safe_load(f)

data_manager = DataManager(exchange_config, live_trades=False)
exchange_name = "Kraken"
data_fetcher = data_manager.get_exchange(exchange_name)
current_balance = data_fetcher.get_balance()
print(f"current_balance {current_balance}")
open_trades = data_fetcher.get_open_trades()
print(f"open_trades {open_trades}")
# Portfolio Manager needs to store info on open trades and dates
# Must compare cached portfolio data against what is from datafetcher

data_manager.shutdown_loop()
#
# cointegration_pairs_path = f"../../../data/historical_data/cointegration_pairs.csv"
# pair_combinations_df = pd.read_csv(cointegration_pairs_path)
# pair_combinations = list(pair_combinations_df.itertuples(index=False, name=None))
