from plotly import graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd


class PriceChart:

    @staticmethod
    def create_ohlc_chart(prices, indicators=None, mark_limit=30, title="Price Chart"):
        fig = go.Figure()
        # fig = make_subplots(
        #     rows=3,
        #     cols=1,
        #     shared_xaxes=True,
        #     row_heights=[0.5, 0.25, 0.25],
        #     vertical_spacing=0.05,
        #     subplot_titles=("Price", "MACD", "RSI"),
        # )

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
                # title="Time",
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

    @staticmethod
    def create_line_charts(prices, indicators=None, mark_limit=30, title="Price Chart"):
        fig = go.Figure()

        for exchange, dataset in prices.items():
            x = [
                dt.strftime("%Y-%m-%d %H:%M:%S")
                for dt in dataset.datetime[-mark_limit:]
            ]
            close_prices = dataset.close[-mark_limit:]

            # Add the candlestick trace for prices
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=close_prices,
                    mode="lines",
                    name=exchange,
                )
            )

        fig.update_layout(
            title=title,
            xaxis=dict(
                tickangle=0,
                rangeslider=dict(visible=False),
            ),
            yaxis=dict(title="Price (USD)"),
            template="plotly_dark",  # Optional: add a template for better visualization
            margin=dict(l=10, r=10, t=40, b=10),  # Minimize the margins
            # height=400,  # Set a fixed height for the chart
        )
        return fig

    @staticmethod
    def format_order_book_for_plotly(order_book, percentage_range=0.05):

        # Extract bids and asks
        bids = order_book["bids"]
        asks = order_book["asks"]

        # Convert to DataFrame for easier manipulation
        df_bids = pd.DataFrame(bids, columns=["price", "quantity"])
        df_asks = pd.DataFrame(asks, columns=["price", "quantity"])

        # Calculate the current price (midpoint of highest bid and lowest ask)
        current_price = (df_bids["price"].max() + df_asks["price"].min()) / 2

        # Calculate the price range limits
        lower_bound = current_price * (1 - percentage_range / 100)
        upper_bound = current_price * (1 + percentage_range / 100)

        # Filter bids and asks to be within the specified range
        df_bids = df_bids[df_bids["price"] >= lower_bound]
        df_asks = df_asks[df_asks["price"] <= upper_bound]

        # Sort by price
        df_bids = df_bids.sort_values(by="price", ascending=False)  # High to low
        df_asks = df_asks.sort_values(by="price")  # Low to high

        # Calculate cumulative quantity for depth chart
        df_bids["cumulative_quantity"] = df_bids["quantity"].cumsum()
        df_asks["cumulative_quantity"] = df_asks["quantity"].cumsum()

        return df_bids, df_asks

    @staticmethod
    def plot_depth_chart(order_book):
        df_bids, df_asks = PriceChart.format_order_book_for_plotly(order_book)

        fig = go.Figure()

        # Add bids trace (buy orders)
        fig.add_trace(
            go.Scatter(
                x=df_bids["price"],
                y=df_bids["cumulative_quantity"],
                mode="lines",
                name="Bids",
                fill="tozeroy",  # Fill to the zero Y-axis
                line=dict(color="green"),
            )
        )

        # Add asks trace (sell orders)
        fig.add_trace(
            go.Scatter(
                x=df_asks["price"],
                y=df_asks["cumulative_quantity"],
                mode="lines",
                name="Asks",
                fill="tozeroy",  # Fill to the zero Y-axis
                line=dict(color="red"),
            )
        )

        # Customize the layout
        fig.update_layout(
            title="Order Book Depth Chart",
            xaxis_title="Price",
            yaxis_title="Cumulative Quantity",
            template="plotly_dark",
            margin=dict(l=10, r=10, t=40, b=10),
        )

        return fig
