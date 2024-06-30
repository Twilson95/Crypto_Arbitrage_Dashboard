from dash import Dash, html, dcc, dash_table
import dash_bootstrap_components as dbc
from src.layout_styles import (
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
            value="bitmex",
            style=filter_style,
        )

        currency_filter = dcc.Dropdown(
            id="currency-selector",
            options=self.filter_component.get_currency_options(),
            placeholder="Select a currency",
            value="bitcoin",
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
        return html.Div(
            children=[
                html.Div(
                    html.H1("Crypto Dashboard"),
                    style=header_style,
                ),
                html.Div(
                    self.create_filters(),
                    style=filter_container_style,
                ),
                html.Div(
                    id="grid-container",
                    style=grid_container_style,
                    children=self.create_grid_elements(),
                ),
                dcc.Interval(
                    id="interval-component", interval=10 * 1000, n_intervals=0
                ),
            ],
            style=container_style,
        )

    # import dash_ag_grid as dag
    # from dash import Dash, html
    #
    # app = Dash(__name__)
    #
    # columnDefs = [
    #     {"headerName": "Employee", "field": "employee"},
    #     {
    #         "headerName": "Number Sick Days (Editable)",
    #         "field": "sickDays",
    #         "editable": True,
    #     },
    # ]
    #
    # rowData = [
    #     {"employee": "Josh Finch", "sickDays": 4},
    #     {"employee": "Flavia Mccloskey", "sickDays": 1},
    # ]
    #
    # getRowStyle = {
    #     "styleConditions": [
    #         {
    #             "condition": "params.data.sickDays > 5 && params.data.sickDays <= 7",
    #             "style": {"backgroundColor": "sandybrown"},
    #         },
    #         {
    #             "condition": "params.data.sickDays >= 8",
    #             "style": {"backgroundColor": "lightcoral"},
    #         },
    #     ],
    #     "defaultStyle": {"backgroundColor": "grey", "color": "white"},
    # }
    #
    # app.layout = html.Div(
    #     [
    #         dag.AgGrid(
    #             id="styling-rows-conditional-style1",
    #             columnDefs=columnDefs,
    #             rowData=rowData,
    #             columnSize="sizeToFit",
    #             getRowStyle=getRowStyle,
    #             dashGridOptions={"animateRows": False},
    #         ),
    #     ],
    # )
