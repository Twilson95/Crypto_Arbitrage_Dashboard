import ast
import plotly.graph_objs as go
import taipy.gui.builder as tgb
from taipy.gui.utils._map_dict import _MapDict


def dashboard_header():
    with tgb.layout(columns="1 1 1") as header:
        with tgb.part():
            tgb.navbar()
        with tgb.part("text-center"):
            tgb.text("#Crypto Dashboard", mode="md")
        with tgb.part():
            tgb.text("")
            tgb.text("Selected: {arbitrage_selector}")
    tgb.part(height="15px")
    return header


def extract_selector_value(selector_option, multi=False):
    def get_value(opt):
        # Convert Taipy _MapDict to real dict
        if isinstance(opt, _MapDict):
            opt = dict(opt)

        # Handle stringified dict
        if isinstance(opt, str):
            try:
                opt = ast.literal_eval(opt)
            except Exception:
                return opt

        # Handle dict
        if isinstance(opt, dict) and "value" in opt:
            return opt["value"]

        # Handle tuple
        if isinstance(opt, tuple) and len(opt) == 2:
            return opt[1]

        return opt

    if selector_option is None:
        return None

    if multi and isinstance(selector_option, (list, tuple)):
        return [get_value(opt) for opt in selector_option]
    else:
        return get_value(selector_option)


# arbitrage_options = [
#     {"label": "Simple", "value": "simple"},
#     {"label": "Triangular", "value": "triangular"},
# ]

arbitrage_options = [
    ("simple", "Simple"),
    ("triangular", "Triangular"),
    ("statistical", "Statistical"),
]

exchange_options = [
    ("Bitmex", "Bitmex"),
    ("Kraken", "Kraken"),
    ("Coinbase", "Coinbase"),
]

currency_options = [
    ("BTC/USD", "BTC/USD"),
    ("ETH/USD", "ETH/USD"),
]

indicator_options = [
    ("SMA", "sma"),
    ("EMA", "ema"),
]

page_options = [
    {"label": "Summary", "value": "Summary"},
    {"label": "Arbitrage", "value": "Arbitrage"},
    {"label": "Simulation", "value": "Simulation"},
]

# arbitrage_options = [
#     {"label": "Simple", "value": "simple"},
#     {"label": "Triangular", "value": "triangular"},
# ]

# exchange_options = [
#     {"Label": "Bitmex", "value": "Bitmex"},
#     {"label": "Kraken", "value": "Kraken"},
#     {"label": "Coinbase", "value": "Coinbase"},
# ]

# currency_options = [
#     {"label": "BTC/USD", "value": "BTC/USD"},
#     {"label": "ETH/USD", "value": "ETH/USD"},
# ]
#
# indicator_options = [
#     {"label": "SMA", "value": "sma"},
#     {"label": "EMA", "value": "ema"},
# ]

page_options = [
    {"label": "Summary", "value": "Summary"},
    {"label": "Arbitrage", "value": "Arbitrage"},
    {"label": "Simulation", "value": "Simulation"},
]

# Default placeholder figure
default_figure = go.Figure(
    layout=dict(
        template="plotly_dark",
        annotations=[
            {
                "text": "Waiting for data...",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 24, "color": "white"},
                "x": 0.5,
                "y": 0.5,
                "xanchor": "center",
                "yanchor": "middle",
            }
        ],
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
)


def on_arbitrage_selector_change(state, var_name, value):
    print("Selector changed (raw):", value)
    try:
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if isinstance(value, dict) and "value" in value:
            state.arbitrage_value = value["value"]
    except Exception as e:
        print("Error in selector callback:", e)


def general_dict_adapter(value):
    print("Selector changed (raw):", value)
    try:
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if isinstance(value, dict):
            return value["label"]
    except Exception as e:
        print("Error in selector callback:", e)


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
        else "red" if trade_status == "closed" and is_in_portfolio else "black"
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
