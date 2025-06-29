import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from cryptopy.src.helpers.json_helper import JsonHelper


class SimulationCharts:

    @staticmethod
    def build_profit_per_open_day(df):
        df["open_days"] = (df["close_date"] - df["open_date"]).dt.days
        profits_per_day = df.groupby("open_days")["profit"].mean().reset_index()

        fig = px.line(
            profits_per_day,
            x="open_days",
            y="profit",
            title="Profit per Open Day",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(xaxis_title="Open Days", yaxis_title="Average Profit")
        return fig

    @staticmethod
    def build_profit_histogram(df):
        fig = px.histogram(
            df,
            x="profit",
            color="open_direction",
            nbins=30,
            title="Histogram of Profits by Open Direction",
            barmode="overlay",
        )
        fig.update_layout(
            xaxis_title="Profit",
            yaxis_title="Count",
        )
        return fig

    @staticmethod
    def build_cumulative_profit(df):
        df_sorted = df.sort_values(by="close_date").copy()
        df_sorted["net_profit"] = df_sorted["profit"].apply(lambda x: max(x, -30))
        df_sorted["cumulative_profit"] = df_sorted["net_profit"].cumsum()

        fig = px.line(
            df_sorted,
            x="close_date",
            y="cumulative_profit",
            title="Cumulative Profit Over Time",
        )

        fig.update_traces(line=dict(color="blue"))

        # Add points colored by open_direction
        scatter = px.scatter(
            df_sorted,
            x="close_date",
            y="cumulative_profit",
            color="open_direction",
        )
        for trace in scatter.data:
            fig.add_trace(trace)

        fig.update_layout(
            xaxis_title="Close Date",
            yaxis_title="Cumulative Profit",
        )

        return fig

    @staticmethod
    def build_expected_vs_actual_profit(df):
        fig = px.scatter(
            df,
            x="open_expected_profit",
            y="profit",
            color="open_direction",
            trendline="ols",
            title="Expected Profit vs Actual Profit",
        )
        fig.update_layout(
            xaxis_title="Expected Profit",
            yaxis_title="Actual Profit",
        )
        return fig

    @staticmethod
    def convert_json_to_df(folder, file_name):
        json_data = JsonHelper.read_from_json(f"{folder}/{file_name}")
        flattened_data = []
        results = json_data["trade_events"]

        for entry in results:
            flattened_entry = {
                "pair": entry["pair"],
                "open_date": entry["open_event"]["date"],
                "open_spread": entry["open_event"]["spread_data"]["spread"],
                "open_direction": entry["open_event"]["direction"],
                "open_avg_price_ratio": entry["open_event"]["avg_price_ratio"],
                # "volume_ratio": entry["open_event"]["volume_ratio"],
                # "volatility_ratio": entry["open_event"]["volatility_ratio"],
                "open_stop_loss": entry["open_event"]["stop_loss"],
                "open_expected_profit": entry["open_event"]["expected_profit"],
                "close_date": entry["close_event"]["date"],
                "close_spread": entry["close_event"]["spread_data"]["spread"],
                "close_reason": entry["close_event"]["reason"],
                "profit": entry["profit"],
            }
            flattened_data.append(flattened_entry)

        df = pd.DataFrame(flattened_data)
        df[["coin_1", "coin_2"]] = df["pair"].apply(pd.Series)
        df["open_date"] = pd.to_datetime(df["open_date"])
        df["close_date"] = pd.to_datetime(df["close_date"])

        return df
