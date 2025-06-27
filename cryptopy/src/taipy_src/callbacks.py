import ast
import time

import pandas as pd
from taipy.gui import State, invoke_long_callback

from cryptopy.src.arbitrage.ArbitrageHandler import ArbitrageHandler
from cryptopy.src.arbitrage.CointegrationCalculator import CointegrationCalculator
from cryptopy.src.prices.PriceChart import PriceChart
from cryptopy.src.taipy_src.helper import (
    extract_selector_value,
    create_filter_label,
    default_figure,
)


def on_summary_init(state):
    print("running on summary init")
    invoke_long_callback(
        state,
        idle_fn,  # Keeps the callback alive
        [],
        update_summary_charts,  # Called every interval
        [],
        5000,  # ms interval
    )


def on_arbitrage_init(state):
    print("running on arbitrage init")
    invoke_long_callback(
        state,
        idle_fn,  # Keeps the callback alive
        [],
        update_arbitrage_charts,  # Called every interval
        [],
        5000,  # ms interval
    )


def on_simulation_init(state):
    print("running on simulation init")
    invoke_long_callback(
        state,
        idle_fn,  # Keeps the callback alive
        [],
        update_simulation_charts,  # Called every interval
        [],
        5000,  # ms interval
    )


def idle_fn():
    while True:
        time.sleep(1)


def update_summary_charts(state):
    update_live_price_chart(state)
    update_historic_price_chart(state)
    update_depth_chart(state)
    update_news_chart(state)


def update_arbitrage_charts(state):
    update_arbitrage_graphs(state)


def update_simulation_charts(state):
    pass


def on_exchange_change_summary_page(state):
    update_historic_price_chart(state)
    update_live_price_chart(state)
    update_depth_chart(state)
    # update_news_chart(state)


def on_currency_change_summary_page(state):
    update_historic_price_chart(state)
    update_live_price_chart(state)
    update_depth_chart(state)


def on_technical_indicator_change_summary_page(state):
    update_historic_price_chart(state)


def render_tab_content(state):
    active_tab = state.selected_page  # You control this from a menu
    arbitrage = getattr(state, "arbitrage_selector", {}).get("value", "simple")

    state.grid_visible = active_tab == "Summary"
    state.arbitrage_visible = active_tab == "Arbitrage"

    state.exchange_filter_visible = arbitrage in ["triangular", "statistical"]
    state.currency_filter_visible = arbitrage == "simple"
    state.indicator_filter_visible = active_tab == "Summary"
    state.arbitrage_filter_visible = active_tab == "Arbitrage"
    state.cointegration_filter_visible = arbitrage == "statistical"
    state.p_value_slider_visible = arbitrage == "statistical"
    state.funds_input_visible = active_tab == "Arbitrage"


def update_historic_price_chart(state):
    currency = extract_selector_value(state.currency_selector)
    exchange = extract_selector_value(state.exchange_selector)
    indicators = extract_selector_value(state.indicator_selector, multi=True)

    if not (currency and exchange):
        state.historic_price_chart_data = default_figure
        return

    prices = state.data_manager.get_historical_prices(exchange, currency)
    if not prices:
        state.historic_price_chart_data = default_figure
        return

    # state.historic_price_chart_data = prices.to_dataframe()
    state.historic_price_chart_data = state.price_chart.create_ohlc_chart(
        prices, indicators, title="Historic Price", mark_limit=60
    )


def update_live_price_chart(state):
    currency = extract_selector_value(state.currency_selector)
    exchange = extract_selector_value(state.exchange_selector)

    if not (currency and exchange):
        state.live_price_chart_data = default_figure
        return

    prices = state.data_manager.get_live_prices(exchange, currency)
    if not prices:
        state.live_price_chart_data = default_figure
        return

    # state.live_price_chart_data = prices.to_dataframe()

    state.live_price_chart_data = state.price_chart.create_ohlc_chart(
        prices, mark_limit=20, title="Live Price"
    )


def update_depth_chart(state: State):
    exchange = extract_selector_value(state.exchange_selector)
    currency = extract_selector_value(state.currency_selector)

    if not currency or not exchange:
        state.depth_chart_data = default_figure
        return

    order_book = state.data_manager.get_order_book(exchange, currency)
    if not order_book:
        state.depth_chart_data = default_figure
    else:
        state.depth_chart_data = state.price_chart.plot_depth_chart(order_book)


def update_news_chart(state: State):
    currency = extract_selector_value(state.currency_selector)
    no_data_default = pd.DataFrame(
        [{"Source": "N/A", "Title": "No news data found", "URL": "", "Published": ""}]
    )

    if not currency:
        state.news_table_data = no_data_default
        return

    news = state.news_fetcher.get_news_data(currency)
    if not news:
        state.news_table_data = no_data_default
    else:
        news_data, _ = state.news_chart.get_table_data(news)
        EXPECTED_NEWS_COLUMNS = ["Source", "Title", "URL", "Published"]
        filtered_data = [
            {key: item[key] for key in EXPECTED_NEWS_COLUMNS if key in item}
            for item in news_data
        ]
        state.news_table_data = pd.DataFrame(filtered_data)


def update_arbitrage_graphs(state: State):
    arbitrage = extract_selector_value(state.arbitrage_selector)

    if arbitrage == "simple":
        main_chart, instructions = simple_arbitrage_graphs(state)
    elif arbitrage == "triangular":
        main_chart, instructions = triangular_arbitrage_graphs(state)
    elif arbitrage == "statistical":
        main_chart, instructions = statistical_arbitrage_graphs(state)
    else:
        main_chart, instructions = "No arbitrage strategy selected", ""

    state.arbitrage_main_view = main_chart
    state.arbitrage_instructions = instructions

    return main_chart, instructions


def simple_arbitrage_graphs(state: State):
    currency = extract_selector_value(state.currency_selector)
    funds = state.funds_input or 0.1

    if not currency:
        return default_figure, default_figure

    prices = state.data_manager.get_live_prices_for_all_exchanges(currency)
    if not prices:
        return default_figure, default_figure

    currency_fees = state.data_manager.get_maker_taker_fees_for_all_exchanges(currency)
    exchange_fees = state.data_manager.get_withdrawal_deposit_fees_for_all_exchanges()
    network_fees = state.data_manager.get_network_fees(currency)

    prices = {
        exchange: price_list
        for exchange, price_list in prices.items()
        if len(price_list.close) > 0
    }

    price_charts = state.price_chart.create_line_charts(
        prices, mark_limit=20, title="Live Exchange Prices"
    )
    arbitrage_instructions = default_figure

    if prices and currency_fees and exchange_fees and network_fees:
        arbitrage_instructions = ArbitrageHandler.return_simple_arbitrage_instructions(
            currency, prices, currency_fees, exchange_fees, network_fees, funds
        )

    return price_charts, arbitrage_instructions


def triangular_arbitrage_graphs(state):
    exchange = extract_selector_value(state.exchange_selector)
    funds = state.funds_input or 0.1

    if not exchange:
        return default_figure, default_figure

    prices, currency_fees = (
        state.data_manager.get_live_prices_and_fees_for_single_exchange(exchange)
    )
    prices = {
        currency: price.close[-1]
        for currency, price in prices.items()
        if len(price.close) > 0
    }

    if not prices or not currency_fees:
        return default_figure, default_figure

    exchange_network_graph, arbitrage_instructions = (
        state.arbitrage_handler.return_triangle_arbitrage_instructions(
            prices, currency_fees, exchange, funds
        )
    )

    return exchange_network_graph, arbitrage_instructions


def statistical_arbitrage_graphs(state):
    exchange = extract_selector_value(state.exchange_selector)
    pair_str = state.cointegration_pairs_input
    funds = state.funds_input or 0.1

    if not exchange or not pair_str:
        return default_figure, default_figure

    try:
        cointegration_pair = tuple(ast.literal_eval(pair_str))
    except Exception:
        return default_figure, default_figure

    prices = state.data_manager.get_df_of_historical_prices_pairs(
        exchange, cointegration_pair
    )
    cointegration_data = state.data_manager.get_cointegration_pair_from_exchange(
        exchange, cointegration_pair
    )

    if cointegration_data.spread is None:
        cointegration_data.spread, cointegration_data.hedge_ratio = (
            CointegrationCalculator.calculate_spread(prices, cointegration_pair)
        )

    _, currency_fees = state.data_manager.get_live_prices_and_fees_for_single_exchange(
        exchange
    )

    arbitrage_instructions = default_figure
    if cointegration_data.spread is not None:
        arbitrage_instructions = (
            ArbitrageHandler.return_statistical_arbitrage_instructions(
                prices, cointegration_data, currency_fees, exchange, funds, window=30
            )
        )

    spread_chart, entry_dates, exit_dates = state.price_chart.plot_spread(
        cointegration_data.spread, cointegration_pair, 30
    )

    statistical_arbitrage_chart = PriceChart.plot_prices_and_spread(
        prices,
        cointegration_pair,
        cointegration_data.hedge_ratio,
        entry_dates,
        exit_dates,
    )

    return [statistical_arbitrage_chart, spread_chart], arbitrage_instructions


def update_filter_values(state: State):
    exchange_value = extract_selector_value(state.exchange_selector)
    currency_value = extract_selector_value(state.currency_selector)
    cointegration_value = state.cointegration_pairs_input
    p_value = state.p_value_slider

    exchange_options = state.data_manager.get_exchanges()
    state.exchange_selector = (
        exchange_options[0]
        if not exchange_value and exchange_options
        else exchange_value
    )

    currency_options = state.data_manager.get_historical_price_options_from_exchange(
        state.exchange_selector
    )
    state.currency_selector = (
        currency_options[0]
        if not currency_value and currency_options
        else currency_value
    )

    cointegration_pairs = state.data_manager.get_cointegration_pairs_from_exchange(
        state.exchange_selector
    ).copy()
    coins_in_portfolio = state.portfolio_manager.get_traded_pairs()

    color_order = {"red": 0, "green": 1, "black": 2}
    cointegration_pairs_str_options = sorted(
        [
            {
                "label": create_filter_label(cointegration_data, coins_in_portfolio)[0],
                "value": str(pair),
                "color": create_filter_label(cointegration_data, coins_in_portfolio)[1],
            }
            for pair, cointegration_data in cointegration_pairs.items()
            if cointegration_data.p_value <= p_value
        ],
        key=lambda x: (color_order[x["color"]], x["value"]),
    )
    cointegration_pairs_str_options = [
        {"label": x["label"], "value": x["value"]}
        for x in cointegration_pairs_str_options
    ]

    if not cointegration_value and cointegration_pairs_str_options:
        cointegration_value = cointegration_pairs_str_options[0]["value"]

    state.exchange_selector_lov = exchange_options
    state.currency_selector_lov = currency_options
    state.cointegration_pairs_input_lov = cointegration_pairs_str_options
    state.cointegration_pairs_input = cointegration_value
