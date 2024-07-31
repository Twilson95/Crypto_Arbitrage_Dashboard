from dash import Dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from src.layout.AppLayout import AppLayout
from src.layout.FilterComponents import FilterComponent

from src.prices.TechnicalIndicators import TechnicalIndicators
from src.prices.PriceChart import PriceChart
from src.prices.DataManager import DataManager
from src.prices.ArbitrageHandler import ArbitrageHandler

from src.news.NewsFetcher import NewsFetcher
from src.news.NewsChart import NewsChart

from time import time

import src.Warnings_to_ignore
import yaml


app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "Crypto Dashboard"

with open("./src/config/exchange_config.yaml", "r") as f:
    exchange_config = yaml.safe_load(f)

with open("./src/config/news_config.yaml", "r") as f:
    news_config = yaml.safe_load(f)

filter_component = FilterComponent()
technical_indicators = TechnicalIndicators()
arbitrage_handler = ArbitrageHandler()

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

# @app.callback(
#     [
#         Input("interval-component", "n_intervals"),
#     ]
# )
# def fetch_all_live_prices(n_intervals):
# get_current_prices()
# identify_arbitrage()
# display_arbitrage()
# trade_arbitrage()


@app.callback(
    [
        Output("grid-container", "style"),
        Output("arbitrage-container", "style"),
        Output("exchange-selector", "style"),
        Output("currency-selector", "style"),
        Output("indicator-selector", "style"),
        Output("arbitrage-selector", "style"),
    ],
    [Input("tabs", "value")],
    [
        State("grid-container", "style"),
        State("arbitrage-container", "style"),
        State("exchange-selector", "style"),
        State("currency-selector", "style"),
        State("indicator-selector", "style"),
        State("arbitrage-selector", "style"),
    ],
)
def render_tab_content(
    active_tab,
    grid_style,
    arbitrage_style,
    exchange_filter_style,
    currency_filter_style,
    indicator_filter_style,
    arbitrage_filter_style,
):
    if active_tab == "tab-1":
        grid_style["display"] = "flex"
        arbitrage_style["display"] = "none"
        exchange_filter_style["display"] = "block"
        currency_filter_style["display"] = "block"
        indicator_filter_style["display"] = "block"
        arbitrage_filter_style["display"] = "none"
    elif active_tab == "tab-2":
        grid_style["display"] = "none"
        arbitrage_style["display"] = "flex"
        exchange_filter_style["display"] = "none"
        currency_filter_style["display"] = "block"
        indicator_filter_style["display"] = "none"
        arbitrage_filter_style["display"] = "block"
    return (
        grid_style,
        arbitrage_style,
        exchange_filter_style,
        currency_filter_style,
        indicator_filter_style,
        arbitrage_filter_style,
    )


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

    return price_chart.create_ohlc_chart(
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
    # print(currency, exchange)
    if not (currency or exchange):
        return {}

    prices = data_manager.get_live_prices(exchange, currency)
    if not prices:
        return {}

    return price_chart.create_ohlc_chart(prices, mark_limit=20, title="Live Price")


@app.callback(Output("news-table", "children"), [Input("currency-selector", "value")])
def update_news_chart(currency):
    if not currency:
        return {}

    news = news_fetcher.get_news_data(currency)
    if not news:
        return {}

    return news_chart.create_table(news)


@app.callback(
    Output("arbitrage_main_view", "figure"),
    [
        Input("arbitrage-selector", "value"),
        Input("currency-selector", "value"),
        Input("interval-component", "n_intervals"),
    ],
)
def update_main_arbitrage_chart(arbitrage, currency, n_intervals):
    if not currency:
        return {}

    if arbitrage == "simple":
        prices = data_manager.get_live_prices_for_all_exchanges(currency)
        fees = data_manager.get_fees_for_all_exchanges(currency)
        # print("live prices", prices)
        # print("fees", fees)
        if not prices:
            return {}

        return price_chart.create_line_charts(
            prices, mark_limit=20, title="Live Exchange Prices"
        )
    elif arbitrage == "triangular":
        return {}
    elif arbitrage == "statistical":
        return {}

    return {}


@app.callback(
    Output("arbitrage_plots_container", "children"),
    [
        Input("arbitrage-selector", "value"),
        Input("currency-selector", "value"),
        Input("interval-component", "n_intervals"),
    ],
)
def update_arbitrage_instructions(arbitrage, currency, n_intervals):

    if arbitrage == "simple":
        prices = data_manager.get_live_prices_for_all_exchanges(currency)
        fees = data_manager.get_fees_for_all_exchanges(currency)
        if prices and fees:
            arbitrage_handler.return_simple_arbitrage(prices, fees)

        # print(arbitrage)
        # print("live prices", prices)
        # print("fees", fees)

    return {}
    # return plots


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
