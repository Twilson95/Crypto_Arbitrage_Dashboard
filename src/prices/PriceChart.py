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
    def plot_spread(spread, pair, avg_window=30, view_window=70):
        """Plot the spread and indicate potential arbitrage opportunities using Plotly."""
        # Calculate rolling mean and standard deviation for the spread
        spread_mean = spread.rolling(window=avg_window).mean()
        spread_std = spread.rolling(window=avg_window).std()

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

        spread = spread[-view_window:]
        spread_mean = spread_mean[-view_window:]
        upper_threshold = upper_threshold[-view_window:]
        lower_threshold = lower_threshold[-view_window:]

        # Determine entry and exit points
        entry_points = []
        exit_points = []
        above_upper = False
        below_lower = False

        # Track entry and exit points based on the threshold conditions
        for i in range(1, len(spread)):
            if spread.iloc[i] > upper_threshold.iloc[i] and not above_upper:
                # Spread is above upper threshold -> Sell expensive, Buy cheaper
                entry_points.append(
                    (spread.index[i], spread.iloc[i], "sell_expensive", "buy_cheaper")
                )
                above_upper = True
            elif spread.iloc[i] < lower_threshold.iloc[i] and not below_lower:
                # Spread is below lower threshold -> Buy expensive, Sell cheaper
                entry_points.append(
                    (spread.index[i], spread.iloc[i], "buy_expensive", "sell_cheaper")
                )
                below_lower = True

            if spread.iloc[i] < spread_mean.iloc[i] and above_upper:
                exit_points.append(
                    (spread.index[i], spread.iloc[i], "buy_expensive", "sell_cheaper")
                )
                above_upper = False
            elif spread.iloc[i] > spread_mean.iloc[i] and below_lower:
                exit_points.append(
                    (spread.index[i], spread.iloc[i], "sell_expensive", "buy_cheaper")
                )
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
            margin=dict(l=10, r=10, t=40, b=10),
        )

        # Return the figure, entry and exit points along with action indicators
        return fig, entry_points, exit_points

    @staticmethod
    def plot_prices_and_spread(
        df, pair, hedge_ratio, entry_points, exit_points, window=70
    ):
        """
        Plot the prices of two coins with hedge ratio adjustment and arbitrage signals.
        Args:
        df: DataFrame containing price data for both coins, with coin names as column labels.
        pair: Tuple containing the coin names, with pair[0] being the more expensive coin and pair[1] the cheaper coin.
        hedge_ratio: The hedge ratio used to adjust the cheaper coin's price.
        entry_points: List of entry points returned from plot_spread, including buy/sell indicators.
        exit_points: List of exit points returned from plot_spread, including buy/sell indicators.
        window: The rolling window size for calculating mean and standard deviation (default is 60).
        """

        # Extract price data from the DataFrame
        more_expensive_price = df[pair[0]]
        cheaper_price = df[pair[1]]

        # Adjust the cheaper coin's price using the hedge ratio
        adjusted_cheaper_price = cheaper_price * hedge_ratio

        # Use the datetime index from the DataFrame as the x-axis
        x_axis = df.index

        # Get the latest 'window' number of dates
        filtered_x_axis = x_axis[-window:]
        filtered_more_expensive_price = more_expensive_price[-window:]
        filtered_adjusted_cheaper_price = adjusted_cheaper_price[-window:]

        # Initialize lists for plotting buy/sell points for both coins
        buy_expensive_points = []
        sell_expensive_points = []
        buy_cheaper_points = []
        sell_cheaper_points = []

        # Filter and plot entry points based on buy/sell actions
        for entry_date, spread_value, action_expensive, action_cheaper in entry_points:
            if entry_date in filtered_x_axis:
                index = filtered_x_axis.get_loc(entry_date)
                more_expensive_price = filtered_more_expensive_price.iloc[index]
                cheaper_price = filtered_adjusted_cheaper_price.iloc[index]
                if action_expensive == "buy_expensive":
                    buy_expensive_points.append((entry_date, more_expensive_price))
                else:
                    sell_expensive_points.append((entry_date, more_expensive_price))

                if action_cheaper == "buy_cheaper":
                    buy_cheaper_points.append((entry_date, cheaper_price))
                else:
                    sell_cheaper_points.append((entry_date, cheaper_price))

        # Handle exit points based on buy/sell actions
        for exit_date, spread_value, action_expensive, action_cheaper in exit_points:
            if exit_date in filtered_x_axis:
                index = filtered_x_axis.get_loc(exit_date)
                more_expensive_price = filtered_more_expensive_price.iloc[index]
                cheaper_price = filtered_adjusted_cheaper_price.iloc[index]
                if action_expensive == "buy_expensive":
                    buy_expensive_points.append((exit_date, more_expensive_price))
                else:
                    sell_expensive_points.append((exit_date, more_expensive_price))

                if action_cheaper == "buy_cheaper":
                    buy_cheaper_points.append((exit_date, cheaper_price))
                else:
                    sell_cheaper_points.append((exit_date, cheaper_price))

        # Create traces for the plot, using only the latest 'window' dates
        more_expensive_trace = go.Scatter(
            x=filtered_x_axis,
            y=filtered_more_expensive_price,
            mode="lines",
            name=f"{pair[0].split('/')[0]} Price",
            line=dict(color="blue"),
        )

        adjusted_cheaper_trace = go.Scatter(
            x=filtered_x_axis,
            y=filtered_adjusted_cheaper_price,
            mode="lines",
            name=f"{pair[1].split('/')[0]} Adj Price",
            line=dict(color="orange"),
        )

        # Entry points for buying and selling both coins
        buy_expensive_trace = go.Scatter(
            x=[point[0] for point in buy_expensive_points],
            y=[point[1] for point in buy_expensive_points],  # Expensive coin buy
            mode="markers",
            name="Buy Expensive",
            marker=dict(color="green", size=10, symbol="triangle-up"),
        )

        sell_expensive_trace = go.Scatter(
            x=[point[0] for point in sell_expensive_points],
            y=[point[1] for point in sell_expensive_points],  # Expensive coin sell
            mode="markers",
            name="Sell Expensive",
            marker=dict(color="red", size=10, symbol="triangle-down"),
        )

        buy_cheaper_trace = go.Scatter(
            x=[point[0] for point in buy_cheaper_points],
            y=[point[1] for point in buy_cheaper_points],  # Cheaper coin buy
            mode="markers",
            name="Buy Cheaper",
            marker=dict(color="green", size=10, symbol="triangle-up"),
        )

        sell_cheaper_trace = go.Scatter(
            x=[point[0] for point in sell_cheaper_points],
            y=[point[1] for point in sell_cheaper_points],  # Cheaper coin sell
            mode="markers",
            name="Sell Cheaper",
            marker=dict(color="red", size=10, symbol="triangle-down"),
        )

        # Create the figure and add all traces
        fig = go.Figure(
            data=[
                more_expensive_trace,
                adjusted_cheaper_trace,
                buy_expensive_trace,
                sell_expensive_trace,
                buy_cheaper_trace,
                sell_cheaper_trace,
            ]
        )

        # Update layout
        fig.update_layout(
            title=f"Prices and Arbitrage Opportunities: {pair[0]} and {pair[1]}",
            xaxis_title=None,
            yaxis_title="Price",
            template="plotly_dark",
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(
                type="date"
            ),  # Ensure datetime index is properly formatted on x-axis
        )

        return fig
