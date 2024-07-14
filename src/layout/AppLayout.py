from dash import html, dcc
import dash_bootstrap_components as dbc
from src.layout.layout_styles import (
    container_style,
    header_style,
    filter_container_style,
    filter_style,
    grid_container_style,
    grid_row_style,
    grid_element_style,
)


class AppLayout:
    def __init__(self, filter_component, technical_indicators):
        self.filter_component = filter_component
        self.technical_indicators = technical_indicators

    def create_filters(self):
        exchange_filter = dcc.Dropdown(
            id="exchange-selector",
            options=self.filter_component.get_exchange_options(),
            placeholder="Select an exchange",
            value="Bitmex",
            style=filter_style,
        )

        currency_filter = dcc.Dropdown(
            id="currency-selector",
            options=self.filter_component.get_currency_options(),
            placeholder="Select a currency",
            value="BTC/USD",
            style=filter_style,
        )

        indicator_filter = dcc.Dropdown(
            id="indicator-selector",
            options=self.technical_indicators.get_indicator_options(),
            placeholder="Select a Technical Indicator",
            value=[],
            multi=True,
            style=filter_style,
        )
        return [exchange_filter, currency_filter, indicator_filter]

    @staticmethod
    def create_grid_elements():
        return [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(
                            id="historic-price-chart",
                            style={"height": "100%", "width": "100%"},
                        ),
                        style=grid_element_style,
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id="live-price-chart",
                            style={"height": "100%", "width": "100%"},
                        ),
                        style=grid_element_style,
                    ),
                ],
                style=grid_row_style,
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Container(
                            id="news-table",
                            style={
                                "height": "100%",
                                "width": "100%",
                                "padding-left": "0",
                                "padding-right": "0",
                                "display": "flex",
                                "flex-direction": "column",
                            },
                        ),
                        style=grid_element_style,
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id="fourth-chart",
                            style={
                                "height": "100%",
                                "width": "100%",
                                "padding-left": "0",
                                "padding-right": "0",
                                "display": "flex",
                                "flex-direction": "column",
                            },
                        ),
                        style=grid_element_style,
                    ),
                ],
                style=grid_row_style,
            ),
        ]

    def generate_layout(self):
        return dbc.Col(
            children=[
                self.create_title_and_tabs(),
                dcc.Interval(
                    id="interval-component", interval=10 * 1000, n_intervals=0
                ),
                html.Div(
                    self.create_filters(),
                    id="filter-container",
                    style=filter_container_style,
                ),
                # html.Div(id="tabs-content"),
                self.tab_1_elements(),
                self.tab_2_elements(),
            ],
            style=container_style,
        )

    @staticmethod
    def create_title_and_tabs():
        return dbc.Row(
            children=[
                dbc.Col(
                    dcc.Tabs(
                        id="tabs",
                        value="tab-1",
                        children=[
                            dcc.Tab(label="Summary", value="tab-1"),
                            dcc.Tab(label="Arbitrage", value="tab-2"),
                        ],
                        style={"width": "100%", "padding": "10px"},
                    ),
                    width={"size": 3, "order": 1},
                    style={"display": "flex", "align-items": "center"},
                ),
                dbc.Col(
                    html.Div(
                        html.H1("Crypto Dashboard"),
                        style={"textAlign": "center", **header_style},
                    ),
                    width={"size": 6, "order": 2},
                ),
                dbc.Col(
                    # Empty column for spacing or other elements if needed
                    width={"size": 3, "order": 3},
                ),
            ],
            style={
                "padding": "5px",
                "align-items": "center",
                "justify-content": "center",
            },
        )

    def tab_1_elements(self):
        return html.Div(
            id="grid-container",
            style=grid_container_style,
            children=self.create_grid_elements(),
        )

    def tab_2_elements(self):
        return html.Div(
            id="arbitrage-container",
            style=grid_container_style,
            children=self.create_arbitrage_elements(),
        )

    @staticmethod
    def create_arbitrage_elements():
        return [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(
                            id="arbitrage_main_view",
                            style={"height": "100%", "width": "100%"},
                        ),
                        style=grid_element_style,
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id="arbitrage_instructions",
                            style={"height": "100%", "width": "100%"},
                        ),
                        style=grid_element_style,
                    ),
                ],
                style=grid_row_style,
            ),
        ]
