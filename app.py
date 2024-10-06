from dash import Dash, dcc

import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from src.layout.AppLayout import AppLayout
from src.layout.FilterComponents import FilterComponent

from src.prices.TechnicalIndicators import TechnicalIndicators
from src.prices.PriceChart import PriceChart
from src.prices.DataManager import DataManager
from src.arbitrage.ArbitrageHandler import ArbitrageHandler
from src.news.NewsFetcher import NewsFetcher
from src.news.NewsChart import NewsChart
from src.prices.NetworkGraph import create_network_graph
from src.arbitrage.CointegrationCalculator import CointegrationCalculator

from dash_bootstrap_templates import load_figure_template
from time import time
import ast

import yaml

# load_figure_template("DARKLY")

app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "Crypto Dashboard"

with open("./src/config/exchange_config.yaml", "r") as f:
    exchange_config = yaml.safe_load(f)

with open("./src/config/news_config.yaml", "r") as f:
    news_config = yaml.safe_load(f)

with open("./src/config/network_fees.yaml", "r") as f:
    network_fees_config = yaml.safe_load(f)

filter_component = FilterComponent()
technical_indicators = TechnicalIndicators()
arbitrage_handler = ArbitrageHandler()

app_layout = AppLayout(filter_component, technical_indicators, 10)
app.layout = app_layout.generate_layout()
start_time = time()
data_manager = DataManager(exchange_config, network_fees_config)
end_time = time()

# news_fetcher = NewsFetcher(news_config)
# news_chart = NewsChart()
price_chart = PriceChart()


@app.callback(
    [
        Output("grid-container", "style"),
        Output("arbitrage-container", "style"),
        Output("exchange-filter-container", "style"),
        Output("currency-filter-container", "style"),
        Output("indicator-selector-container", "style"),
        Output("arbitrage-filter-container", "style"),
        Output("cointegration-selector-container", "style"),
        Output("p-value-slider-container", "style"),
        Output("funds-input-container", "style"),
    ],
    [Input("tabs", "value")],
    [
        State("grid-container", "style"),
        State("arbitrage-container", "style"),
        State("exchange-filter-container", "style"),
        State("currency-filter-container", "style"),
        State("indicator-selector-container", "style"),
        State("arbitrage-filter-container", "style"),
        State("cointegration-selector-container", "style"),
        State("p-value-slider-container", "style"),
        State("funds-input-container", "style"),
        Input("arbitrage-selector", "value"),
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
    cointegration_filter_style,
    p_value_slider_style,
    funds_input_style,
    arbitrage_filter_value,
):
    if active_tab == "tab-1":
        grid_style["display"] = "flex"
        arbitrage_style["display"] = "none"
        exchange_filter_style["display"] = "block"
        currency_filter_style["display"] = "block"
        indicator_filter_style["display"] = "block"
        arbitrage_filter_style["display"] = "none"
        cointegration_filter_style["display"] = "none"
        p_value_slider_style["display"] = "none"
        funds_input_style["display"] = "none"
    elif active_tab == "tab-2":
        grid_style["display"] = "none"
        arbitrage_style["display"] = "flex"
        if arbitrage_filter_value == "triangular":
            exchange_filter_style["display"] = "block"
            currency_filter_style["display"] = "none"
            cointegration_filter_style["display"] = "none"
            p_value_slider_style["display"] = "none"
        elif arbitrage_filter_value == "statistical":
            exchange_filter_style["display"] = "block"
            currency_filter_style["display"] = "none"
            cointegration_filter_style["display"] = "block"
            p_value_slider_style["display"] = "block"
        else:
            exchange_filter_style["display"] = "none"
            currency_filter_style["display"] = "block"
            cointegration_filter_style["display"] = "none"
            p_value_slider_style["display"] = "none"

        indicator_filter_style["display"] = "none"
        arbitrage_filter_style["display"] = "block"
        funds_input_style["display"] = "block"

    return (
        grid_style,
        arbitrage_style,
        exchange_filter_style,
        currency_filter_style,
        indicator_filter_style,
        arbitrage_filter_style,
        cointegration_filter_style,
        p_value_slider_style,
        funds_input_style,
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
def update_live_price_chart(currency, exchange_name, n_intervals, indicator):
    # print(currency, exchange)
    if not (currency or exchange_name):
        return {}

    prices = data_manager.get_live_prices(exchange_name, currency)
    if not prices:
        return {}

    return price_chart.create_ohlc_chart(prices, mark_limit=20, title="Live Price")


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


@app.callback(
    Output("depth-chart", "figure"),
    [
        Input("exchange-selector", "value"),
        Input("currency-selector", "value"),
        Input("interval-component", "n_intervals"),
    ],
)
def update_news_chart(exchange, currency, n_intervals):
    if not currency or not exchange:
        return {}

    order_book = data_manager.get_order_book(exchange, currency)
    if not order_book:
        return {}

    return price_chart.plot_depth_chart(order_book)


@app.callback(
    [
        Output("arbitrage_main_view", "children"),
        Output("arbitrage_instructions_container", "children"),
    ],
    [
        Input("arbitrage-selector", "value"),
        Input("exchange-selector", "value"),
        Input("currency-selector", "value"),
        Input("cointegration-pairs-input", "value"),
        Input("funds-input", "value"),
        Input("interval-component", "n_intervals"),
    ],
)
def update_arbitrage_graphs(
    arbitrage, exchange, currency, cointegration_pair_str, funds, n_intervals
):
    if not funds:
        funds = 0.1

    # start_time = time()
    if arbitrage == "simple":
        main_chart, instructions = simple_arbitrage_graphs(currency, funds)
    elif arbitrage == "triangular":
        main_chart, instructions = triangular_arbitrage_graphs(exchange, funds)
    elif arbitrage == "statistical":
        main_chart, instructions = statistical_arbitrage_graphs(
            exchange, funds, cointegration_pair_str
        )
    else:
        main_chart, instructions = {}, {}

    # end_time = time()
    # print(end_time - start_time)
    return main_chart, instructions


def simple_arbitrage_graphs(currency, funds):
    if currency is None:
        return {}, {}

    prices = data_manager.get_live_prices_for_all_exchanges(currency)
    currency_fees = data_manager.get_maker_taker_fees_for_all_exchanges(currency)
    exchange_fees = data_manager.get_withdrawal_deposit_fees_for_all_exchanges()
    network_fees = data_manager.get_network_fees(currency)

    if not prices:
        return {}, {}

    prices = {
        exchange: price_list
        for exchange, price_list in prices.items()
        if len(price_list.close) > 0
    }

    price_charts = price_chart.create_line_charts(
        prices, mark_limit=20, title="Live Exchange Prices"
    )
    arbitrage_instructions = {}
    if prices and currency_fees and exchange_fees and network_fees:
        arbitrage_instructions = arbitrage_handler.return_simple_arbitrage_instructions(
            currency, prices, currency_fees, exchange_fees, network_fees, funds
        )

    return (
        dcc.Graph(figure=price_charts, style={"height": "100%"}),
        arbitrage_instructions,
    )


def triangular_arbitrage_graphs(exchange, funds):
    if exchange is None:
        return {}, {}

    prices, currency_fees = data_manager.get_live_prices_and_fees_for_single_exchange(
        exchange
    )
    prices = {
        currency: price.close[-1]
        for currency, price in prices.items()
        if len(price.close) > 0
    }

    arbitrage_opportunities = None
    if prices and currency_fees:
        arbitrage_opportunities = arbitrage_handler.identify_triangle_arbitrage(
            prices, currency_fees, exchange, funds
        )

    exchange_network_graph = create_network_graph(prices, arbitrage_opportunities)

    arbitrage_instructions = {}
    if arbitrage_opportunities:
        arbitrage_instructions = (
            arbitrage_handler.return_triangle_arbitrage_instructions(
                arbitrage_opportunities
            )
        )
    return (
        dcc.Graph(figure=exchange_network_graph, style={"height": "100%"}),
        arbitrage_instructions,
    )


def statistical_arbitrage_graphs(exchange, funds, cointegration_pair_str):
    if exchange is None or cointegration_pair_str is None:
        return {}, {}

    cointegration_pair = ast.literal_eval(cointegration_pair_str)
    start_time = time()
    prices = data_manager.get_historical_prices_for_all_currencies(exchange)

    spread = CointegrationCalculator.calculate_spread(
        prices, cointegration_pair[0], cointegration_pair[1]
    )
    cointegration_pair = (cointegration_pair[0], cointegration_pair[1])

    _, currency_fees = data_manager.get_live_prices_and_fees_for_single_exchange(
        exchange
    )
    got_prices = time()
    # print("got prices", got_prices - start_time)
    arbitrage_opportunities = None

    if spread:
        arbitrage_opportunities = ArbitrageHandler.identify_all_statistical_arbitrage(
            prices,
            cointegration_pair,
            spread,
            currency_fees,
            exchange,
            funds,
            window=30,
        )
    identify_arbitrage = time()
    # print("identify arbitrage", identify_arbitrage - got_prices)

    if not arbitrage_opportunities:
        return {}, {}

    arbitrage_instructions = (
        arbitrage_handler.return_statistical_arbitrage_instructions(
            arbitrage_opportunities
        )
    )
    returned_instructions = time()
    # print("returned_instructions", returned_instructions - identify_arbitrage)

    spread_chart, entry_dates, exit_dates = price_chart.plot_spread(
        spread["spread"], cointegration_pair, 30
    )
    plot_spread = time()
    # print("plot_spread", plot_spread - returned_instructions)
    statistical_arbitrage_chart = PriceChart.plot_prices_and_spread(
        prices,
        cointegration_pair,
        spread["hedge_ratio"],
        entry_dates,
        exit_dates,
    )
    plot_prices = time()
    # print("plot_prices", plot_prices - plot_spread)

    return (
        [
            dcc.Graph(figure=statistical_arbitrage_chart, style={"height": "50%"}),
            dcc.Graph(figure=spread_chart, style={"height": "50%"}),
        ],
        arbitrage_instructions,
    )


@app.callback(
    [
        Output("exchange-selector", "options"),
        Output("exchange-selector", "value"),
        Output("currency-selector", "options"),
        Output("currency-selector", "value"),
        Output("cointegration-pairs-input", "options"),
        Output("cointegration-pairs-input", "value"),
    ],
    [
        Input("exchange-selector", "value"),
        Input("currency-selector", "value"),
        Input("cointegration-pairs-input", "value"),
        Input("p-value-slider", "value"),
        Input("interval-component", "n_intervals"),
    ],
)
def update_filter_values(
    exchange_value, currency_value, cointegration_value, p_value, n_intervals
):
    exchange_options = data_manager.get_exchanges()
    if not exchange_options:
        return (
            [],  # Empty exchange options
            None,  # No value selected
            [],  # Empty currency options
            None,  # No value selected
            [],  # Empty cointegration pairs options
            None,  # No value selected
        )

    if exchange_value is None:
        exchange_value = exchange_options[0]

    currency_data = data_manager.get_historical_prices_for_all_currencies(
        exchange_value
    )
    if currency_data is None:
        currency_options = []
    else:
        currency_options = list(currency_data.keys())

    if not currency_options:
        currency_value = None
    elif currency_value is None:
        currency_value = currency_options[0]

    pairs = data_manager.get_exchanges_cointegration_pairs(exchange_value)
    significant_pairs = {}
    if pairs:
        significant_pairs = [
            (pair[0], pair[1], "p_val " + str(round(value, 2)))
            for pair, value in pairs.items()
            if value <= p_value
        ]

    if not significant_pairs:
        cointegration_pairs_str_options = []
    else:
        cointegration_pairs_str_options = sorted(
            [str(tup) for tup in significant_pairs]
        )

    if cointegration_value is None:
        if not cointegration_pairs_str_options:
            cointegration_value = None
        else:
            cointegration_value = cointegration_pairs_str_options[0]

    return (
        exchange_options,
        exchange_value,
        currency_options,
        currency_value,
        cointegration_pairs_str_options,
        cointegration_value,
    )


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
