import asyncio
import sys
import time
from threading import Thread

import pandas as pd
import taipy as tp
import yaml
from taipy.gui import invoke_long_callback, get_state_id, invoke_callback

from cryptopy.src.arbitrage.ArbitrageHandler import ArbitrageHandler
from cryptopy.src.layout.AppLayout import AppLayout
from cryptopy.src.layout.FilterComponents import FilterComponent
from cryptopy.src.news.NewsChart import NewsChart
from cryptopy.src.news.NewsFetcher import NewsFetcher
from cryptopy.src.prices.DataManager import DataManager
from cryptopy.src.prices.PriceChart import PriceChart
from cryptopy.src.prices.TechnicalIndicators import TechnicalIndicators
from cryptopy.src.taipy_src.arbitrage_page import arbitrage_page
from cryptopy.src.taipy_src.callbacks import (
    update_live_price_chart,
    update_historic_price_chart,
    update_depth_chart,
    update_news_chart,
)
from cryptopy.src.taipy_src.simulation_page import simulation_page
from cryptopy.src.taipy_src.summary_page import summary_page
from cryptopy.src.taipy_src.helper import (
    arbitrage_options,
    exchange_options,
    currency_options,
    default_figure,
)
from cryptopy.src.trading.PortfolioManager import PortfolioManager

# if sys.platform == "win32":
# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

with open("cryptopy/config/exchange_config.yaml", "r") as f:
    exchange_config = yaml.safe_load(f)

with open("cryptopy/config/news_config.yaml", "r") as f:
    news_config = yaml.safe_load(f)

with open("cryptopy/config/network_fees.yaml", "r") as f:
    network_fees_config = yaml.safe_load(f)

filter_component = FilterComponent()
technical_indicators = TechnicalIndicators()
arbitrage_handler = ArbitrageHandler()

app_layout = AppLayout(filter_component, technical_indicators, 10)
data_manager = DataManager(exchange_config, network_fees_config, live_trades=True)

news_fetcher = NewsFetcher(news_config)
news_chart = NewsChart()
price_chart = PriceChart()
trades_path = r"data/portfolio_data/Kraken/trades.json"
portfolio_manager = PortfolioManager(trades_path=trades_path)
portfolio_manager.read_open_events()

# User selections
arbitrage_selector = arbitrage_options[0]
exchange_selector = exchange_options[0]
currency_selector = currency_options[0]
indicator_selector = []
cointegration_pairs_input = None
p_value_slider = 0.05
funds_input = 100

arbitrage_value = "simple"  # default value

# data to be fed into charts
historic_price_chart_data = {}
live_price_chart_data = {}
depth_chart_data = {}
# news_table_data = [
#     {
#         "Index": 0,
#         "Source": "N/A",
#         "Title": "No news data found",
#         "URL": "",
#         "Published": "",
#     }
# ]
news_table_data = pd.DataFrame(
    [{"Source": "N/A", "Title": "No news data found", "URL": "", "Published": ""}]
)
pages = {
    "Summary": summary_page,
    "Arbitrage": arbitrage_page,
    "Simulation": simulation_page,
}


def update_charts(state):
    # This is where the chart gets refreshed every N ms
    print(f"live data state {state.live_price_chart_data}")
    print(f"historic data state {state.historic_price_chart_data}")
    print(f"depth data state {state.depth_chart_data}")

    update_live_price_chart(state)
    update_historic_price_chart(state)
    update_depth_chart(state)
    update_news_chart(state)

    print(f"post live data state {state.live_price_chart_data}")
    print(f"post historic data state {state.historic_price_chart_data}")
    print(f"post depth data state {state.depth_chart_data}")


state_id_list = []
stop_request = False


def on_init(state):
    print("running on_init")
    state_id = get_state_id(state)
    if state_id := get_state_id(state):
        state_id_list.append(state_id)
    print(state_id_list)


def refresh(gui: tp.Gui):
    # wait for server to start
    global stop_request
    global state_id_list
    time.sleep(10)
    while not stop_request:
        for state_id in state_id_list:
            invoke_callback(gui, state_id, update_charts)
        time.sleep(5)


if __name__ == "__main__":
    gui = tp.Gui(pages=pages)
    refresh_th = Thread(target=refresh, args=[gui])
    refresh_th.start()
    try:
        gui.run(title="Crypto Dashboard", use_reloader=True, debug=False)
    except KeyboardInterrupt as e:
        pass
    finally:
        store_request = True
        refresh_th.join()
