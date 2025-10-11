import asyncio
import json
import logging
import os
import sys

from dash import Dash, dcc, html, exceptions
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from time import time
import ast
import yaml

from cryptopy import (
    AppLayout,
    FilterComponent,
    TechnicalIndicators,
    PriceChart,
    DataManager,
    ArbitrageHandler,
    NewsFetcher,
    NewsChart,
    CointegrationCalculator,
    PortfolioManager,
)
from cryptopy.src.arbitrage.SimpleArbitrage import SimpleArbitrage
from cryptopy.src.arbitrage.StatisticalArbitrage import StatisticalArbitrage
from cryptopy.src.arbitrage.TriangularArbitrage import TriangularArbitrage
from cryptopy.src.arbitrage.ArbitrageTracker import ArbitrageTracker
from cryptopy.src.trading.SimulationCharts import SimulationCharts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,  # Ensures it applies even if something already configured logging
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Change to DEBUG for verbose logs

app = Dash(
    __name__, external_stylesheets=[dbc.themes.SLATE], assets_folder="../../assets"
)

app.title = "Crypto Dashboard"

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

with open("cryptopy/config/exchange_config.yaml", "r") as f:
    exchange_config = yaml.safe_load(f)

with open("cryptopy/config/news_config.yaml", "r") as f:
    news_config = yaml.safe_load(f)

with open("cryptopy/config/network_fees.yaml", "r") as f:
    network_fees_config = yaml.safe_load(f)

SIMULATION_FOLDER = "data/simulations/portfolio_sim"

filter_component = FilterComponent()
technical_indicators = TechnicalIndicators()
arbitrage_handler = ArbitrageHandler()

chart_refresh_rate = 2
price_refresh_delay = 0
app_layout = AppLayout(filter_component, technical_indicators, chart_refresh_rate)
app.layout = app_layout.generate_layout()
start_time = time()
data_manager = DataManager(
    exchange_config,
    network_fees_config,
    live_trades=True,
    # use_cache=True,
    price_refresh_delay=price_refresh_delay,
)
end_time = time()

news_fetcher = NewsFetcher(news_config)
news_chart = NewsChart()
price_chart = PriceChart()
trades_path = r"data/portfolio_data/Kraken/trades.json"
portfolio_manager = PortfolioManager(trades_path=trades_path)
portfolio_manager.read_open_events()

arbitrage_tracker = ArbitrageTracker(r"data/arbitrage/arbitrage_lifetimes.jsonl")

order_book_path = r"C:\Users\thoma\PycharmProjects\CryptoDashboard\data\order_book\order_book_btc_2.json"


def save_order_book(order_book, filename):
    with open(filename, "w") as f:
        json.dump(order_book, f)


@app.callback(
    [
        Output("grid-container", "style"),
        Output("arbitrage-container", "style"),
        Output("simulation-container", "style"),
        Output("exchange-filter-container", "style"),
        Output("currency-filter-container", "style"),
        Output("indicator-selector-container", "style"),
        Output("arbitrage-filter-container", "style"),
        Output("cointegration-selector-container", "style"),
        Output("p-value-slider-container", "style"),
        Output("funds-input-container", "style"),
        Output("simulation-filter-container", "style"),
        Output("simulation-column-filter-container", "style"),
    ],
    [Input("tabs", "value")],
    [
        State("grid-container", "style"),
        State("arbitrage-container", "style"),
        State("simulation-container", "style"),
        State("exchange-filter-container", "style"),
        State("currency-filter-container", "style"),
        State("indicator-selector-container", "style"),
        State("arbitrage-filter-container", "style"),
        State("cointegration-selector-container", "style"),
        State("p-value-slider-container", "style"),
        State("funds-input-container", "style"),
        State("simulation-filter-container", "style"),
        State("simulation-column-filter-container", "style"),
        Input("arbitrage-selector", "value"),
    ],
)
def render_tab_content(
    active_tab,
    grid_style,
    arbitrage_style,
    simulation_style,
    exchange_filter_style,
    currency_filter_style,
    indicator_filter_style,
    arbitrage_filter_style,
    cointegration_filter_style,
    p_value_slider_style,
    funds_input_style,
    simulation_selector_style,
    simulation_split_selector_style,
    arbitrage_filter_value,
):
    if active_tab == "tab-1":
        grid_style["display"] = "flex"
        arbitrage_style["display"] = "none"

        # simulation_style["visibility"] = "hidden"
        # simulation_style["height"] = "0"
        simulation_style["display"] = "none"

        exchange_filter_style["display"] = "block"
        currency_filter_style["display"] = "block"
        indicator_filter_style["display"] = "block"
        arbitrage_filter_style["display"] = "none"
        cointegration_filter_style["display"] = "none"
        p_value_slider_style["display"] = "none"
        funds_input_style["display"] = "none"
        simulation_selector_style["display"] = "none"
        simulation_split_selector_style["display"] = "none"
    elif active_tab == "tab-2":
        grid_style["display"] = "none"
        arbitrage_style["display"] = "flex"

        # simulation_style["visibility"] = "hidden"
        # simulation_style["height"] = "0"
        simulation_style["display"] = "none"

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
        simulation_selector_style["display"] = "none"
        simulation_split_selector_style["display"] = "none"

    elif active_tab == "tab-3":
        grid_style["display"] = "none"
        arbitrage_style["display"] = "none"

        # simulation_style["visibility"] = "visible"
        simulation_style["height"] = "100%"
        simulation_style["display"] = "flex"

        exchange_filter_style["display"] = "none"
        currency_filter_style["display"] = "none"
        indicator_filter_style["display"] = "none"
        arbitrage_filter_style["display"] = "none"
        cointegration_filter_style["display"] = "none"
        p_value_slider_style["display"] = "none"
        funds_input_style["display"] = "none"
        simulation_selector_style["display"] = "block"
        simulation_split_selector_style["display"] = "block"

    return (
        grid_style,
        arbitrage_style,
        simulation_style,
        exchange_filter_style,
        currency_filter_style,
        indicator_filter_style,
        arbitrage_filter_style,
        cointegration_filter_style,
        p_value_slider_style,
        funds_input_style,
        simulation_selector_style,
        simulation_split_selector_style,
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
        return app_layout.default_figure

    prices = data_manager.get_historical_prices(exchange, currency)
    if not prices:
        return app_layout.default_figure

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
    if not (currency or exchange_name):
        return app_layout.default_figure

    prices = data_manager.get_live_prices(exchange_name, currency)
    if not prices:
        return app_layout.default_figure

    return price_chart.create_ohlc_chart(prices, mark_limit=20, title="Live Price")


@app.callback(Output("news-table", "children"), [Input("currency-selector", "value")])
def update_news_chart(currency):
    if not currency:
        return [dcc.Graph(id="news-default", figure=app_layout.default_figure)]

    news = news_fetcher.get_news_data(currency)
    if not news:
        return [dcc.Graph(id="news-default", figure=app_layout.default_figure)]

    return news_chart.create_table(news)


@app.callback(
    Output("depth-chart", "figure"),
    [
        Input("exchange-selector", "value"),
        Input("currency-selector", "value"),
        Input("interval-component", "n_intervals"),
    ],
)
def update_depth_chart(exchange, currency, n_intervals):
    if not currency or not exchange:
        return app_layout.default_figure

    order_book = data_manager.get_order_book(exchange, currency)
    save_order_book(order_book, order_book_path)

    if not order_book:
        return app_layout.default_figure

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
        main_chart, instructions = (
            html.Div("No arbitrage strategy selected"),
            html.Div(),
        )

    # end_time = time()
    # print(end_time - start_time)
    return main_chart, instructions


def simple_arbitrage_graphs(currency, funds):
    if currency is None:
        return html.Div(), html.Div()

    prices = data_manager.get_live_prices_for_all_exchanges(currency)
    currency_fees = data_manager.get_maker_taker_fees_for_all_exchanges(currency)
    exchange_fees = data_manager.get_withdrawal_deposit_fees_for_all_exchanges()
    network_fees = data_manager.get_network_fees(currency)

    if not prices:
        return html.Div(), html.Div()

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
        arbitrages = SimpleArbitrage.identify_arbitrage(
            currency, prices, currency_fees, exchange_fees, network_fees, funds
        )
        # arbitrage_tracker.update(arbitrages)
        arbitrage_instructions = arbitrage_handler.return_simple_arbitrage_instructions(
            arbitrages
        )

    return (
        dcc.Graph(figure=price_charts, style={"height": "100%"}),
        arbitrage_instructions,
    )


def triangular_arbitrage_graphs(exchange, funds):
    if exchange is None:
        return html.Div(), html.Div()

    prices, currency_fees = data_manager.get_live_prices_and_fees_for_single_exchange(
        exchange
    )
    prices = {
        currency: price.close[-1]
        for currency, price in prices.items()
        if len(price.close) > 0
    }

    if not prices or not currency_fees:
        return html.Div(), html.Div()

    arbitrages = TriangularArbitrage.identify_triangle_arbitrage(
        prices, currency_fees, exchange, funds
    )
    # arbitrage_tracker.update(arbitrages)
    exchange_network_graph, arbitrage_instructions = (
        arbitrage_handler.return_triangle_arbitrage_instructions(prices, arbitrages)
    )
    return (
        dcc.Graph(figure=exchange_network_graph, style={"height": "100%"}),
        arbitrage_instructions,
    )


def statistical_arbitrage_graphs(exchange, funds, cointegration_pair_str):
    if exchange is None or cointegration_pair_str is None:
        return html.Div(), html.Div()

    cointegration_pair = tuple(ast.literal_eval(cointegration_pair_str))

    prices = data_manager.get_df_of_historical_prices_pairs(
        exchange, cointegration_pair
    )

    cointegration_data = data_manager.get_cointegration_pair_from_exchange(
        exchange, cointegration_pair
    )
    if cointegration_data.spread is None:
        cointegration_data.spread, cointegration_data.hedge_ratio = (
            CointegrationCalculator.calculate_spread(prices, cointegration_pair)
        )

    _, currency_fees = data_manager.get_live_prices_and_fees_for_single_exchange(
        exchange
    )
    arbitrage_instructions = html.Div()
    if cointegration_data.spread is not None:
        arbitrages = StatisticalArbitrage.identify_all_statistical_arbitrage(
            prices,
            cointegration_data,
            currency_fees,
            exchange,
            funds,
            window=30,
        )
        arbitrage_instructions = (
            ArbitrageHandler.return_statistical_arbitrage_instructions(arbitrages)
        )

    spread_chart, entry_dates, exit_dates = price_chart.plot_spread(
        cointegration_data.spread, cointegration_pair, 30
    )

    statistical_arbitrage_chart = PriceChart.plot_prices_and_spread(
        prices,
        cointegration_pair,
        cointegration_data.hedge_ratio,
        entry_dates,
        exit_dates,
    )

    if arbitrage_instructions == {}:
        arbitrage_instructions = html.Div()

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

    currency_options = data_manager.get_historical_price_options_from_exchange(
        exchange_value
    )

    if not currency_options:
        currency_value = None
        currency_options = []
    elif currency_value is None:
        currency_value = currency_options[0]

    cointegration_pairs = data_manager.get_cointegration_pairs_from_exchange(
        exchange_value
    ).copy()

    # print(f"time to update all trade details {(end_time - start_time):.4f}")

    coins_in_portfolio = portfolio_manager.get_traded_pairs()
    color_order = {"red": 0, "green": 1, "black": 2}
    # Modified to only call create_filter_label once and store the color for sorting
    cointegration_pairs_str_options = sorted(
        [
            {
                # Run create_filter_label once and store both the label and color
                "label": create_filter_label(cointegration_data, coins_in_portfolio)[
                    0
                ],  # HTML label with color
                "value": str(pair),
                "color": create_filter_label(cointegration_data, coins_in_portfolio)[
                    1
                ],  # Extracted color for sorting
            }
            for pair, cointegration_data in cointegration_pairs.items()
            if cointegration_data.p_value <= p_value
        ],
        # Sort first by 'value', then by 'color'
        key=lambda x: (color_order[x["color"]], x["value"]),
    )

    # Remove the 'color' field after sorting to ensure the output is clean
    cointegration_pairs_str_options = [
        {"label": x["label"], "value": x["value"]}
        for x in cointegration_pairs_str_options
    ]

    if cointegration_value is None:
        if not cointegration_pairs_str_options:
            cointegration_value = None
        else:
            cointegration_value = cointegration_pairs_str_options[0]["value"]

    return (
        exchange_options,
        exchange_value,
        currency_options,
        currency_value,
        cointegration_pairs_str_options,
        cointegration_value,
    )


def create_filter_label(cointegration_data, coins_in_portfolio):
    trade_status = cointegration_data.trade_details.get("trade_status")
    pair = cointegration_data.pair

    is_open_opportunity = cointegration_data.is_open_opportunity()
    # print("is opportunity: ", is_open_opportunity)
    is_in_portfolio = (
        pair in coins_in_portfolio or (pair[1], pair[0]) in coins_in_portfolio
    )

    color = (
        "green"
        if is_open_opportunity and not is_in_portfolio
        else "red"
        if trade_status == "closed" and is_in_portfolio
        else "black"
    )

    return (
        html.Span(
            [
                # html.Img(
                #     src="/assets/images/language_icons/r-lang_50px.svg", height=20
                # ),
                html.Span(
                    f"{pair[0]}, {pair[1]}",
                    style={"font-size": 15, "padding-left": 10, "color": color},
                ),
            ],
            style={"align-items": "center", "justify-content": "center"},
        ),
        color,
    )


@app.callback(
    [
        Output("simulation-chart-1", "figure"),
        Output("simulation-chart-2", "figure"),
        Output("simulation-chart-3", "figure"),
        Output("simulation-chart-4", "figure"),
    ],
    [
        Input("simulation-selector", "value"),
        Input("tabs", "value"),
        Input("column-selector", "value"),
    ],
)
def display_selected_file(file_name, tabs, column_to_split):
    if tabs != "tab-3":
        raise exceptions.PreventUpdate
    if not file_name:
        # Return empty figures
        return [app_layout.default_figure] * 4

    df = SimulationCharts.convert_json_to_df(SIMULATION_FOLDER, file_name)

    if column_to_split is None:
        column_to_split = "open_direction"

    # Build the four figures
    fig1 = SimulationCharts.build_profit_per_open_day(df)
    fig2 = SimulationCharts.build_profit_histogram(df, column_to_split)
    fig3 = SimulationCharts.build_cumulative_profit(df, column_to_split)
    fig4 = SimulationCharts.build_expected_vs_actual_profit(df, column_to_split)

    return fig1, fig2, fig3, fig4


def get_json_files():
    return [f for f in os.listdir(SIMULATION_FOLDER) if f.endswith(".json")]


@app.callback(
    Output("simulation-selector", "options"),
    Output("simulation-selector", "value"),
    # Input("interval-component", "n_intervals"),
    Input("tabs", "value"),
    State("simulation-selector", "value"),
)
def update_dropdown(tabs, value):
    options = [{"label": f, "value": f} for f in get_json_files()]
    if options and value is None:
        value = options[0]["value"]
    return options, value


def main():
    app.run(debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
