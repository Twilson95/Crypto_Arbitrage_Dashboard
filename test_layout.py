from dash import Dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

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

print("data enabled")

price_chart = PriceChart()
news_chart = NewsChart()


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
