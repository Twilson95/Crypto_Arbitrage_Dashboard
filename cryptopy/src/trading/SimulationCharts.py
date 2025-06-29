import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import plotly.io as pio

from cryptopy.src.helpers.json_helper import JsonHelper


class SimulationCharts:

    @staticmethod
    def build_profit_per_open_day(df):
        df["open_days"] = (df["close_date"] - df["open_date"]).dt.days
        profits_per_day = df.groupby("open_days")["profit"].mean().reset_index()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=profits_per_day["open_days"],
                y=profits_per_day["profit"],
                mode="lines",
                name="Mean Profit",
            )
        )

        # Add horizontal line at y=0
        fig.add_shape(
            type="line",
            x0=profits_per_day["open_days"].min(),
            x1=profits_per_day["open_days"].max(),
            y0=0,
            y1=0,
            line=dict(color="red", dash="dash"),
        )

        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=40, b=10),
            # height="300px",
            xaxis_title="Open Days",
            yaxis_title="Average Profit",
            title="Profit per Open Day",
        )
        return fig

    @staticmethod
    def build_profit_histogram(df):
        fig = go.Figure()

        # For each direction, add a separate histogram
        directions = df["open_direction"].unique()
        for direction in directions:
            subset = df[df["open_direction"] == direction]
            fig.add_trace(
                go.Histogram(
                    x=subset["profit"],
                    name=direction,
                    opacity=0.6,
                    nbinsx=30,
                )
            )

        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=40, b=10),
            barmode="overlay",
            xaxis_title="Profit",
            yaxis_title="Count",
            title="Histogram of Profits by Open Direction",
        )
        return fig

    @staticmethod
    def build_cumulative_profit(df):
        df_sorted = df.sort_values(by="close_date").copy()
        df_sorted["net_profit"] = df_sorted["profit"].apply(lambda x: max(x, -30))
        df_sorted["cumulative_profit"] = df_sorted["net_profit"].cumsum()

        fig = go.Figure()

        # Main line
        fig.add_trace(
            go.Scatter(
                x=df_sorted["close_date"],
                y=df_sorted["cumulative_profit"],
                mode="lines",
                name="Cumulative Profit",
                line=dict(color="blue"),
            )
        )

        # Scatter points colored by open_direction
        colors = {
            k: v
            for k, v in zip(
                df_sorted["open_direction"].unique(),
                ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"],
            )
        }
        for direction in df_sorted["open_direction"].unique():
            subset = df_sorted[df_sorted["open_direction"] == direction]
            fig.add_trace(
                go.Scatter(
                    x=subset["close_date"],
                    y=subset["cumulative_profit"],
                    mode="markers",
                    name=direction,
                    marker=dict(size=6, color=colors[direction]),
                )
            )

        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Close Date",
            yaxis_title="Cumulative Profit",
            title="Cumulative Profit Over Time",
        )
        return fig

    @staticmethod
    def build_expected_vs_actual_profit(df):
        fig = go.Figure()

        colors = {
            k: v
            for k, v in zip(
                df["open_direction"].unique(),
                ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"],
            )
        }

        for direction in df["open_direction"].unique():
            subset = df[df["open_direction"] == direction]
            fig.add_trace(
                go.Scatter(
                    x=subset["open_expected_profit"],
                    y=subset["profit"],
                    mode="markers",
                    name=direction,
                    marker=dict(size=6, color=colors[direction]),
                )
            )

        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Expected Profit",
            yaxis_title="Actual Profit",
            title="Expected vs Actual Profit",
        )
        return fig

    @staticmethod
    def convert_json_to_df(folder, file_name):
        json_data = JsonHelper.read_from_json(f"{folder}/{file_name}")
        flattened_data = []
        results = json_data.get("trade_events", [])

        for entry in results:
            open_event = entry.get("open_event", {})
            close_event = entry.get("close_event", {})
            open_spread_data = open_event.get("spread_data", {})
            close_spread_data = close_event.get("spread_data", {})

            flattened_entry = {
                "pair": entry.get("pair"),
                "open_date": open_event.get("date"),
                "open_spread": open_spread_data.get("spread"),
                "open_direction": open_event.get("direction"),
                "open_avg_price_ratio": open_event.get("avg_price_ratio"),
                "volume_ratio": open_event.get("volume_ratio"),
                "volatility_ratio": open_event.get("volatility_ratio"),
                "open_stop_loss": open_event.get("stop_loss"),
                "open_expected_profit": open_event.get("expected_profit"),
                "close_date": close_event.get("date"),
                "close_spread": close_spread_data.get("spread"),
                "close_reason": close_event.get("reason"),
                "profit": entry.get("profit"),
            }
            flattened_data.append(flattened_entry)

        df = pd.DataFrame(flattened_data)

        if "pair" in df.columns:
            df[["coin_1", "coin_2"]] = df["pair"].apply(pd.Series)

        for date_col in ["open_date", "close_date"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])

        return df
