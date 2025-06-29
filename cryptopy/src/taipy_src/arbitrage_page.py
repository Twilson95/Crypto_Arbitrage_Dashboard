import taipy.gui.builder as tgb

from cryptopy.src.taipy_src.callbacks import on_arbitrage_init
from cryptopy.src.taipy_src.helper import (
    arbitrage_options,
    exchange_options,
    currency_options,
    dashboard_header,
)

# --- PAGE 2: Arbitrage ---
with tgb.Page(on_init=on_arbitrage_init) as arbitrage_page:
    dashboard_header()

    with tgb.layout(columns="1 1 1 1 1 1 1"):
        tgb.selector(
            label="Arbitrage",
            lov=arbitrage_options,
            value="{arbitrage_selector}",
            dropdown=True,
            multiple=False,
            width="200px",
        )
        tgb.selector(
            label="Exchange",
            lov=exchange_options,
            value="{exchange_selector}",
            dropdown=True,
            width="200px",
        )
        tgb.selector(
            label="Currency",
            lov=currency_options,
            value="{currency_selector}",
            dropdown=True,
            width="200px",
        )
        tgb.selector(
            label="Cointegration Pairs",
            lov={},  # You can dynamically update this later
            value="{cointegration_pairs_input}",
            dropdown=True,
            width="200px",
        )

        with tgb.part():
            tgb.text("P-Value Threshold")
            tgb.slider(
                min=0,
                max=1,
                step=0.01,
                value="{p_value_slider}",
                width="200px",
            )
        tgb.number(label="Funds", value="{funds_input}", width="200px")

    with tgb.layout(columns="4 1"):
        with tgb.part():
            tgb.chart(
                # data="{historic_price_chart_data}",
                height="500px",
                figure="{arbitrage_main_view}",
            )
        with tgb.part():
            tgb.chart(
                # data="{historic_price_chart_data}",
                height="500px",
                figure="{arbitrage_instructions}",
            )
