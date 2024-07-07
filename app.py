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

import src.Warnings_to_ignore
import configparser

app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "Crypto Dashboard"

exchange_config = configparser.ConfigParser()
exchange_config.read("./src/config/exchange_api_keys.ini")

news_config = configparser.ConfigParser()
news_config.read("./src/config/news_api_keys.ini")

filter_component = FilterComponent()
technical_indicators = TechnicalIndicators()
app_layout = AppLayout(filter_component, technical_indicators)
app.layout = app_layout.generate_layout()
start_time = time()
data_manager = DataManager(exchange_config)
end_time = time()
print(f"finished querying data: {end_time-start_time}")
news_fetcher = NewsFetcher(news_config)
print("data enabled")

price_chart = PriceChart()
news_chart = NewsChart()


@app.callback(
    [
        Input("interval-component", "n_intervals"),
    ]
)
def fetch_all_live_prices(n_intervals):
    data_manager.fetch_all_live_prices()
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
    # print("history price update", exchange)
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
    # print("live chart update", exchange)
    if not (currency or exchange):
        return {}

    prices = data_manager.get_live_prices(exchange, currency)
    # print(prices)
    if not prices:
        return {}

    return price_chart.create_chart(prices, mark_limit=20, title="Live Price")


@app.callback(Output("news-table", "children"), [Input("currency-selector", "value")])
def update_news_chart(currency):
    if not currency:
        return {}

    news = news_fetcher.get_news_data(currency)
    if not news:
        return {}

    return news_chart.create_table(news)


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
