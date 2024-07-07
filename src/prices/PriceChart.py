from plotly import graph_objs as go


class PriceChart:
    @staticmethod
    def create_chart(prices, indicators=None, mark_limit=30, title="Price Chart"):
        fig = go.Figure()

        x = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in prices.datetime[-mark_limit:]]
        # print("dates: ", input_from, x)
        close_prices = prices.close[-mark_limit:]
        open_prices = prices.open[-mark_limit:]
        high_prices = prices.high[-mark_limit:]
        low_prices = prices.low[-mark_limit:]

        # Add the candlestick trace for prices
        fig.add_trace(
            go.Candlestick(
                x=x,
                open=open_prices,
                high=high_prices,
                low=low_prices,
                close=close_prices,
                name="Price",
            )
        )

        # Add traces for each indicator
        if indicators:
            for indicator_name, indicator_data in indicators.items():
                if indicator_name == "datetime":
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=prices.datetime[-mark_limit:],
                        y=indicator_data[-mark_limit:],
                        mode="lines",
                        name=indicator_name,
                    )
                )

        # Layout for the chart
        fig.update_layout(
            title=title,
            xaxis=dict(
                title="Time",
                #  type="date",
                #  tickformat="%Y-%m-%d %H:%M:%S",
                tickangle=0,
                rangeslider=dict(visible=False),
            ),  # Disable the range slider
            yaxis=dict(title="Price (USD)"),
            template="plotly_dark",  # Optional: add a template for better visualization
            margin=dict(l=10, r=10, t=40, b=10),  # Minimize the margins
            # height=400,  # Set a fixed height for the chart
        )

        return fig
