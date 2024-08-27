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
from src.prices.helper_functions import generate_log_marks


class AppLayout:
    def __init__(self, filter_component, technical_indicators, interval_rate_sec=10):
        self.filter_component = filter_component
        self.technical_indicators = technical_indicators
        self.interval_rate = interval_rate_sec

    def create_filters(self):

        arbitrage_filter = html.Div(
            [
                html.Label("Arbitrage"),
                dcc.Dropdown(
                    id="arbitrage-selector",
                    options=self.filter_component.get_arbitrage_options(),
                    placeholder="Select an Arbitrage technique",
                    value="simple",
                    multi=False,
                ),
            ],
            title="Arbitrage",
            id="arbitrage-filter-container",
            style=filter_style,
        )

        exchange_filter = html.Div(
            [
                html.Label("Exchange"),
                dcc.Dropdown(
                    id="exchange-selector",
                    options=self.filter_component.get_exchange_options(),
                    placeholder="Select an exchange",
                    value="Bitmex",
                ),
            ],
            title="Exchange",
            id="exchange-filter-container",
            style=filter_style,
        )

        currency_filter = html.Div(
            [
                html.Label("Currency"),
                dcc.Dropdown(
                    id="currency-selector",
                    options=self.filter_component.get_currency_options(),
                    placeholder="Select a currency",
                    value="BTC/USD",
                ),
            ],
            title="Currency",
            id="currency-filter-container",
            style=filter_style,
        )

        indicator_filter = html.Div(
            [
                html.Label("Technical Indicators"),
                dcc.Dropdown(
                    id="indicator-selector",
                    options=self.technical_indicators.get_indicator_options(),
                    placeholder="Select a Technical Indicator",
                    value=[],
                    multi=True,
                ),
            ],
            title="Select Indicators",
            id="indicator-selector-container",
            style=filter_style,
        )

        # log_marks = generate_log_marks()
        funds_slider = html.Div(
            [
                html.Label("Funds"),
                dcc.Slider(
                    id="funds-slider",
                    min=0,
                    max=100_000,
                    step=0.1,
                    value=1,
                    marks=None,
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                    },  # Tooltip to show value
                ),
            ],
            title="Funds",
            id="funds-slider-container",  # New container div for the slider
            style=filter_style,  # Initialize with display set to 'none'
        )

        funds_input = html.Div(
            [
                html.Label("Funds", style={"display": "block"}),  # Label for the input
                html.Div(
                    [
                        html.Span(
                            "$", style={"margin-right": "5px", "font-size": "16px"}
                        ),  # Dollar symbol
                        dcc.Input(
                            id="funds-input",
                            type="number",
                            value=1,  # Default value
                            min=0,
                            # max=1_000_000,
                            step=1,
                            style={
                                "display": "inline-block",
                                "width": "90%",
                            },  # Input style
                        ),
                    ],
                    style={
                        "display": "flex",
                        "align-items": "center",
                        "padding-top": "5px",
                    },  # Container style for symbol and input
                ),
            ],
            title="Funds",
            id="funds-input-container",  # New container div for the slider
            style=filter_style,  # Initialize with display set to 'none'
        )

        return [
            arbitrage_filter,
            exchange_filter,
            currency_filter,
            indicator_filter,
            # funds_slider,
            funds_input,
        ]

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
                        dcc.Graph(
                            id="depth-chart",
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
                ],
                style=grid_row_style,
            ),
        ]

    def generate_layout(self):
        return dbc.Col(
            children=[
                self.create_title_and_tabs(),
                dcc.Interval(
                    id="interval-component",
                    interval=self.interval_rate * 1000,
                    n_intervals=0,
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
                        style={"width": "100%", "padding": "0px", "height": "60px"},
                    ),
                    width={"size": 3, "order": 1, "height": "100%"},
                    style={
                        "display": "flex",
                        "align-items": "center",
                        "height": "100%",
                    },
                ),
                dbc.Col(
                    html.Div(
                        html.H1("Crypto Dashboard"),
                        style={"textAlign": "center", **header_style},
                    ),
                    width={"size": 6, "order": 2, "height": "100%"},
                ),
                dbc.Col(
                    # Empty column for spacing or other elements if needed
                    width={"size": 3, "order": 3, "height": "100%"},
                ),
            ],
            style=header_style,
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
                        width=9,
                        style={
                            "height": "100%",  # Set a fixed height to prevent shrinking
                            "padding": "2.5px",
                            "overflow": "hidden",
                            "size": 9,
                            # "outline": "2px solid yellow",
                        },
                    ),
                    dbc.Col(
                        html.Div(
                            id="arbitrage_instructions_container",
                            style={
                                "height": "570px",
                                "width": "100%",
                                "overflowY": "scroll",
                            },
                        ),
                        width=3,
                        style={
                            "height": "100%",  # Set a fixed height to prevent shrinking
                            "padding": "2.5px",
                            "size": 1,
                            # "outline": "2px solid yellow",
                        },
                    ),
                ],
                style=grid_row_style,
            ),
        ]
