from dash import Dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from src.layout.AppLayout import AppLayout
from src.layout.FilterComponents import FilterComponent

from src.prices.TechnicalIndicators import TechnicalIndicators
from src.prices.PriceChart import PriceChart
from src.prices.DataManager import DataManager

from src.news.NewsFetcher import NewsFetcher
from src.news.NewsChart import NewsChart

from time import time
import asyncio
from threading import Thread

import src.Warnings_to_ignore
import configparser
import yaml


app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "Crypto Dashboard"

with open("./src/config/exchange_config.yaml", "r") as f:
    exchange_config = yaml.safe_load(f)

with open("./src/config/news_config.yaml", "r") as f:
    news_config = yaml.safe_load(f)

filter_component = FilterComponent()
technical_indicators = TechnicalIndicators()
app_layout = AppLayout(filter_component, technical_indicators)
app.layout = app_layout.generate_layout()
start_time = time()
data_manager = DataManager(exchange_config)
end_time = time()
print(f"finished querying data: {end_time-start_time}")
# news_fetcher = NewsFetcher(news_config)
print("data enabled")

price_chart = PriceChart()
# news_chart = NewsChart()


# @app.callback(
#     [
#         Input("interval-component", "n_intervals"),
#     ]
# )
# def fetch_all_live_prices(n_intervals):
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(data_manager.fetch_all_live_prices())
# asyncio.create_task(data_manager.fetch_all_live_prices())
# asyncio.ensure_future
# asyncio.run(data_manager.fetch_all_live_prices())
# get_current_prices()
# identify_arbitrage()
# display_arbitrage()
# trade_arbitrage()


@app.callback(
    Output("historic-price-chart", "figure"),
    [
        Input("currency-selector", "value"),
        Input("exchange-selector", "value"),
        # Input("interval-component", "n_intervals"),
        Input("indicator-selector", "value"),
    ],
)
def update_historic_price_chart(currency, exchange, selected_indicators):
    if not (currency and exchange):
        return {}

    prices = data_manager.get_historical_prices(exchange, currency)
    if not prices:
        return {}

    indicators = technical_indicators.apply_indicators(prices, selected_indicators)

    return price_chart.create_chart(
        prices, indicators, title="Historic Price", mark_limit=60
    )


@app.callback(
    Output("live-price-chart", "figure"),
    [
        Input("currency-selector", "value"),
        Input("exchange-selector", "value"),
        Input("interval-component", "n_intervals"),
        Input("indicator-selector", "value"),
    ],
)
def update_live_price_chart(currency, exchange, n_intervals, indicator):
    if not (currency or exchange):
        return {}

    prices = data_manager.get_live_prices(exchange, currency)
    if not prices:
        return {}

    return price_chart.create_chart(prices, mark_limit=20, title="Live Price")


# @app.callback(Output("news-table", "children"), [Input("currency-selector", "value")])
# def update_news_chart(currency):
#     if not currency:
#         return {}
#
#     news = news_fetcher.get_news_data(currency)
#     if not news:
#         return {}
#
#     return news_chart.create_table(news)


# async def fetch_all_live_data():
#     """Another async function"""
#     while True:
#         await data_manager.fetch_all_live_prices()
#         await asyncio.sleep(10)
#         print("Fetched live data")
#
#
# async def async_main():
#     """Main async function"""
#     await fetch_all_live_data()
#     # await asyncio.gather(fetch_all_live_data())
#
#
# def async_main_wrapper():
#     """Not async Wrapper around async_main to run it as target function of Thread"""
#     asyncio.run(async_main())


if __name__ == "__main__":
    # th = Thread(target=async_main_wrapper, daemon=True)
    # th.start()
    app.run_server(debug=True, use_reloader=False)
    # th.join()
