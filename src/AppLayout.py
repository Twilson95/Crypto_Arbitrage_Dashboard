from dash import Dash, html, dcc, dash_table
import dash_bootstrap_components as dbc


class AppLayout:
    def __init__(self, filter_component, technical_indicators):
        self.filter_component = filter_component
        self.technical_indicators = technical_indicators
        self.container_style = {
            "display": "flex",
            "flex-direction": "column",
            "height": "100vh",  # Use full viewport height
            "margin": "0px",
        }
        self.header_style = {
            "flex": "0 1 auto",  # Allow header to take only necessary space
            "text-align": "center",
            "padding": "10px",
        }
        self.filter_container_style = {
            "display": "flex",
            "justify-content": "space-between",
            "align-items": "center",
            "width": "100%",
            "padding": "10px",
            "outline": "2px solid red",
            "flex": "0 1 auto",  # Allow filter container to take only necessary space
        }
        self.filter_style = {
            "width": "100%",
            "margin-left": "1%",
            "margin-right": "1%",
        }
        self.grid_container_style = {
            "flex": "1",  # Take the remaining space
            "display": "flex",
            "flex-direction": "column",
            # "justify-content": "center",
            # "align-items": "center",
            "width": "100%",
            # "height": "80vh",
            "outline": "2px solid blue",
        }
        self.grid_row_style = {
            "width": "100%",
            "display": "flex",
            "flex": "1",
            "height": "50%",
            "justify-content": "center",
            "align-items": "center",
            "outline": "2px solid orange",
        }
        self.grid_element_style = {
            "flex": "0 1 49%",  # flex-grow, flex-shrink, flex-basis
            "min-width": "49%",  # Ensure elements don't shrink below 45% of the container width
            "height": "99%",  # Set a fixed height to prevent shrinking
            "max-width": "49%",
            # "margin": "1%",
            "display": "flex",
            # "overflow": "hidden",
        }

    def create_filters(self):
        exchange_filter = dcc.Dropdown(
            id="exchange-selector",
            options=self.filter_component.get_exchange_options(),
            placeholder="Select an exchange",
            value="bitmex",
            style=self.filter_style,
        )

        currency_filter = dcc.Dropdown(
            id="currency-selector",
            options=self.filter_component.get_currency_options(),
            placeholder="Select a currency",
            value="bitcoin",
            style=self.filter_style,
        )

        indicator_filter = dcc.Dropdown(
            id="indicator-selector",
            options=self.technical_indicators.get_indicator_options(),
            placeholder="Select a Technical Indicator",
            value=[],
            multi=True,
            style=self.filter_style,
        )
        return [exchange_filter, currency_filter, indicator_filter]

    def create_grid_elements(self):
        return [
            dbc.Row(
                [
                    dcc.Graph(id="historic-price-chart", style=self.grid_element_style),
                    dcc.Graph(id="live-price-chart", style=self.grid_element_style),
                ],
                style=self.grid_row_style,
            ),
            dbc.Row(
                [
                    dbc.Container(id="news-table", style=self.grid_element_style),
                    # dcc.Graph(id="dummy-chart", style=self.grid_element_style),
                    dcc.Graph(id="fourth-chart", style=self.grid_element_style),
                ],
                style=self.grid_row_style,
            ),
        ]

    def generate_layout(self):
        return html.Div(
            children=[
                html.Div(
                    html.H1("Crypto Dashboard"),
                    style=self.header_style,
                ),
                html.Div(
                    self.create_filters(),
                    style=self.filter_container_style,
                ),
                html.Div(
                    id="grid-container",
                    style=self.grid_container_style,
                    children=self.create_grid_elements(),
                ),
                dcc.Interval(
                    id="interval-component", interval=10 * 1000, n_intervals=0
                ),
            ],
            style=self.container_style,
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
