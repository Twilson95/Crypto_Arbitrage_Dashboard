from dash import Dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from src.layout.AppLayout import AppLayout
from src.layout.FilterComponents import FilterComponent
from src.prices.TechnicalIndicators import TechnicalIndicators
from src.prices.PriceChart import PriceChart
from src.news.NewsChart import NewsChart
from dummy_data import news, live_data, historical_data

import configparser

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE],
    # external_stylesheets=[dbc.themes.BOOTSTRAP],
)
app.title = "Crypto Dashboard"

config = configparser.ConfigParser()
config.read("./src/Config.ini")

filter_component = FilterComponent()
technical_indicators = TechnicalIndicators()
app_layout = AppLayout(filter_component, technical_indicators)
app.layout = app_layout.generate_layout()

price_chart = PriceChart()
news_chart = NewsChart()


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
    indicators = None

    return price_chart.create_chart(
        historical_data, indicators, title="Historic Price", mark_limit=60
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

    return price_chart.create_chart(live_data, mark_limit=20, title="Live Price")


@app.callback(Output("news-table", "children"), [Input("currency-selector", "value")])
def update_news_chart(currency):

    return news_chart.create_table_layout(news)


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
