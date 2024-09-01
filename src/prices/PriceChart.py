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
                # increasing=dict(
                #     line=dict(color="green")
                # ),  # Green for increasing prices
                # decreasing=dict(line=dict(color="red")),  # Red for decreasing prices
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
    def format_order_book_for_plotly(order_book, percentage_range=0.1):

        # Extract bids and asks
        bids = order_book["bids"]
        asks = order_book["asks"]

        norm_bids = PriceChart.normalize_data(bids)
        norm_ask = PriceChart.normalize_data(asks)

        # Convert to DataFrame for easier manipulation
        df_bids = pd.DataFrame(norm_bids, columns=["price", "quantity"])
        df_asks = pd.DataFrame(norm_ask, columns=["price", "quantity"])

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
    def normalize_data(data):
        normalized_data = [item[:2] for item in data]
        return normalized_data

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
                # line=dict(color="green"),
                line=dict(color="rgba(0, 204, 150, 0.6)"),
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
                # line=dict(color="red"),
                line=dict(color="rgba(239, 85, 59, 0.6)"),
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

    @staticmethod
    def plot_spread(spread, pair, window=30):
        """Plot the spread and indicate potential arbitrage opportunities using Plotly."""
        # Calculate rolling mean and standard deviation for the spread
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()

        # Define thresholds for entry and exit signals
        upper_threshold = spread_mean + 2 * spread_std
        lower_threshold = spread_mean - 2 * spread_std

        # Align all data
        spread = spread.align(upper_threshold)[0]
        spread = spread.align(lower_threshold)[0]

        # Mask the initial period where thresholds are not yet available
        spread = spread[spread_mean.notna()]
        spread_mean = spread_mean[spread_mean.notna()]
        upper_threshold = upper_threshold[upper_threshold.notna()]
        lower_threshold = lower_threshold[lower_threshold.notna()]

        # Determine entry and exit points
        entry_points = []
        exit_points = []
        above_upper = False
        below_lower = False

        # Track entry and exit points based on the threshold conditions
        for i in range(1, len(spread)):
            if spread.iloc[i] > upper_threshold.iloc[i] and not above_upper:
                entry_points.append((spread.index[i], spread.iloc[i]))
                above_upper = True
            elif spread.iloc[i] < lower_threshold.iloc[i] and not below_lower:
                entry_points.append((spread.index[i], spread.iloc[i]))
                below_lower = True

            if spread.iloc[i] < spread_mean.iloc[i] and above_upper:
                exit_points.append((spread.index[i], spread.iloc[i]))
                above_upper = False
            elif spread.iloc[i] > spread_mean.iloc[i] and below_lower:
                exit_points.append((spread.index[i], spread.iloc[i]))
                below_lower = False

        # Create traces for the plot
        spread_trace = go.Scatter(
            x=spread.index,
            y=spread,
            mode="lines",
            name="Spread",
            line=dict(color="blue"),
        )

        mean_trace = go.Scatter(
            x=spread_mean.index,
            y=spread_mean,
            mode="lines",
            name="Mean",
            line=dict(color="white", dash="dash"),
        )

        upper_trace = go.Scatter(
            x=upper_threshold.index,
            y=upper_threshold,
            mode="lines",
            name="Upper Threshold",
            line=dict(color="red", dash="dot"),
        )

        lower_trace = go.Scatter(
            x=lower_threshold.index,
            y=lower_threshold,
            mode="lines",
            name="Lower Threshold",
            line=dict(color="green", dash="dot"),
        )

        entry_trace = go.Scatter(
            x=[point[0] for point in entry_points],
            y=[point[1] for point in entry_points],
            mode="markers",
            name="Entry Point",
            marker=dict(color="red", size=10, symbol="triangle-up"),
        )

        exit_trace = go.Scatter(
            x=[point[0] for point in exit_points],
            y=[point[1] for point in exit_points],
            mode="markers",
            name="Exit Point",
            marker=dict(color="green", size=10, symbol="triangle-down"),
        )

        # Create the figure and add all traces
        fig = go.Figure(
            data=[
                spread_trace,
                mean_trace,
                upper_trace,
                lower_trace,
                entry_trace,
                exit_trace,
            ]
        )

        # Update layout
        fig.update_layout(
            title=f"Spread and Arbitrage Opportunities, {pair[0]} and {pair[1]}",
            xaxis_title=None,
            yaxis_title="Spread",
            template="plotly_dark",
        )
        return fig
