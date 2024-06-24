from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from src.AppLayout import AppLayout
from src.FilterComponents import FilterComponent
from src.TechnicalIndicators import TechnicalIndicators
from src.PriceChart import PriceChart

# from src.NewsChart import NewsChart
from src.DataManager import DataManager

app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "Crypto Dashboard"

filter_component = FilterComponent()
technical_indicators = TechnicalIndicators()
app_layout = AppLayout(filter_component, technical_indicators)
app.layout = app_layout.generate_layout()
data_manager = DataManager()
price_chart = PriceChart()
# news_chart = NewsChart()


@app.callback(
    [
        Input("interval-component", "n_intervals"),
    ]
)
def fetch_all_live_prices(n_intervals):
    data_manager.fetch_all_live_prices()


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
    if not prices:
        return {}

    return price_chart.create_chart(prices, mark_limit=20, title="Live Price")


# @app.callback(Output("news-chart", "figure"), [Input("currency-selector", "value")])
# def update_news_chart(currency):
#     if currency:
#         news = news_chart.get_news(currency)
#         return news_chart.create_chart(news)
#     return {}


if __name__ == "__main__":
    app.run_server(debug=True)
