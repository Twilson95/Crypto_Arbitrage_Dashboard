import ast
import plotly.graph_objs as go
import taipy.gui.builder as tgb


def dashboard_header():
    with tgb.layout(columns="1 1 1") as header:
        with tgb.part():
            tgb.navbar()
        with tgb.part("text-center"):
            tgb.text("#Crypto Dashboard", mode="md")
        with tgb.part():
            tgb.text("")
            tgb.text("Selected: {arbitrage_selector}")
    tgb.part(height="20px")
    return header


def extract_selector_value(selector_option, multi=False):
    if not isinstance(selector_option, tuple):
        return None
    if multi:
        return [option[1] for option in selector_option]
    else:
        return selector_option[1]


# arbitrage_options = [
#     {"label": "Simple", "value": "simple"},
#     {"label": "Triangular", "value": "triangular"},
# ]

arbitrage_options = [("simple", "Simple"), ("triangular", "Triangular")]

exchange_options = [
    {"label": "Binance", "value": "binance"},
    {"label": "Kraken", "value": "kraken"},
]

currency_options = [
    {"label": "BTC/USD", "value": "BTC/USD"},
    {"label": "ETH/USD", "value": "ETH/USD"},
]

indicator_options = [
    {"label": "SMA", "value": "sma"},
    {"label": "EMA", "value": "ema"},
]

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
