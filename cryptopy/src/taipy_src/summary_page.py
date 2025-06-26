import taipy.gui.builder as tgb

from cryptopy.src.taipy_src.helper import (
    exchange_options,
    currency_options,
    indicator_options,
    dashboard_header,
)
from cryptopy.src.taipy_src.callbacks import (
    on_exchange_change_summary_page,
    on_currency_change_summary_page,
    on_technical_indicator_change_summary_page,
)

# --- PAGE 1: Summary ---
with tgb.Page() as summary_page:
    dashboard_header()

    with tgb.layout(columns="1 1 1 1 1 1 1"):
        tgb.selector(
            label="Exchange",
            lov=exchange_options,
            value="{exchange_selector}",
            dropdown=True,
            width="200px",
            on_change=on_exchange_change_summary_page,
        )
        tgb.selector(
            label="Currency",
            lov=currency_options,
            value="{currency_selector}",
            dropdown=True,
            width="200px",
            on_change=on_currency_change_summary_page,
        )
        tgb.selector(
            label="Technical Indicators",
            lov=indicator_options,
            value="{indicator_selector}",
            multiple=True,
            dropdown=True,
            width="200px",
            on_change=on_technical_indicator_change_summary_page,
        )

    with tgb.layout(columns="1 1"):
        with tgb.part():
            tgb.chart(
                data="{historic_price_chart_data}",
                height="500px",
                type="candlestick",
                x="datetime",
                open="open",
                close="close",
                low="low",
                high="high",
            )
            # tgb.chart(data="{depth_chart_data}", height="300px")

        with tgb.part():
            tgb.chart(
                data="{live_price_chart_data}",
                height="300px",
                type="candlestick",
                x="datetime",
                open="open",
                close="close",
                low="low",
                high="high",
            )
            tgb.table(
                data="{news_table_data}",
                columns=["Source", "Title", "URL", "Published"],
                height="400px",
                width="100%",
                markdown=True,  # Enable markdown (optional for basic formatting)
            )
