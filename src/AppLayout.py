from dash import Dash, html, dcc


class AppLayout:
    def __init__(self, filter_component, technical_indicators):
        self.filter_component = filter_component
        self.technical_indicators = technical_indicators
        self.filter_container_style = {
            "display": "flex",
            "justify-content": "space-between",
            "width": "100%",
        }
        self.filter_style = {
            "width": "98%",
            "display": "inline-block",
            "margin-left": "1%",
            "margin-right": "1%",
        }
        self.grid_container_style = {
            "width": "100%",
            "height": "50%",
            "verticalAlign": "bottom",
            "margin": "0px",
        }
        self.grid_element_style = {
            "width": "49.25%",
            "height": "290px",
            "display": "inline-block",
            "margin-left": "0.5%",
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
            dcc.Graph(id="historic-price-chart", style=self.grid_element_style),
            dcc.Graph(id="live-price-chart", style=self.grid_element_style),
            dcc.Graph(id="news-chart", style=self.grid_element_style),
            dcc.Graph(id="fourth-chart", style=self.grid_element_style),
        ]

    def generate_layout(self):
        return html.Div(
            children=[
                html.H1("Crypto Dashboard"),
                html.Div(
                    self.create_filters(),
                    style=self.filter_container_style,
                ),
                html.Div(
                    id="grid-container",
                    style=self.grid_container_style,
                    children=html.Div(self.create_grid_elements()),
                ),
                dcc.Interval(
                    id="interval-component", interval=10 * 1000, n_intervals=0
                ),
            ],
            style={"overflow": "hidden"},  # Ensure no scrollbars
        )
