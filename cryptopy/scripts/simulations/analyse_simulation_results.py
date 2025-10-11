import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
from cryptopy.scripts.simulations.analysis_helpers import (
    box_plot_coins,
    scatter_plot_with_trend,
)
from cryptopy import JsonHelper

matplotlib.use("TkAgg")  # Or another backend like 'Qt5Agg' depending on your system

simulation_name = "long_history_forecast_exit_price_min_exp_0.005"
simulation_path = f"../../../data/simulations/portfolio_sim/{simulation_name}.json"
# simulation_path = f"../../../data/simulations/all_trades/{simulation_name}.json"

json_data = JsonHelper.read_from_json(simulation_path)
flattened_data = []
results = json_data["trade_events"]
print(results)
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

# Convert to DataFrame
df = pd.DataFrame(flattened_data)

df[["coin_1", "coin_2"]] = df["pair"].apply(pd.Series)

print(f"number of trades {len(df)}")

min_index = df["profit"].idxmin()
print(f"lowest_profit {df.loc[min_index]}")

print(df.groupby("close_reason")["profit"].count())
print(df.groupby("close_reason")["profit"].sum())

df["open_date"] = pd.to_datetime(df["open_date"])
df["close_date"] = pd.to_datetime(df["close_date"])
df["open_days"] = (df["close_date"] - df["open_date"]).dt.days

# df_success = df.loc[df["close_reason"] == "crossed_mean"]
# print("mean days of successful position", df_success["open_days"].mean())
# df_success["open_days"].hist(bins=30)
# plt.xlabel("open_days")
# plt.ylabel("Frequency")
# plt.title("Histogram of Profits by Pairs")
# plt.show()
#
profits_per_day = df.groupby(["open_days"])["profit"].mean()
fig, ax1 = plt.subplots(figsize=(12, 6))
profits_per_day.plot()
plt.axhline(y=0, color="red", linestyle="--", linewidth=1)
# ax1.set_ylim(-50, 200)
plt.xlabel("open_days")
plt.ylabel("profit")
plt.title("profit per open day")
plt.show()

# ----------- histogram of profit per trade --------------
sns.histplot(
    data=df,
    x="profit",
    hue="open_direction",
    bins=30,
    kde=False,  # Set to True if you want a kernel density estimate
)

plt.xlabel("Profit")
plt.ylabel("Frequency")
plt.title("Histogram of Profits by Pairs (Colored by Open_Direction)")
plt.tight_layout()
plt.show()

scatter_plot_with_trend(df)
print(df.groupby(["open_direction"])["profit"].sum())


# ------------- box plot of profits per coin ----------
df["coin"] = df["coin_1"]
df_copy = df.copy()
df_copy["coin"] = df["coin_2"]
df_combined = pd.concat([df, df_copy])
box_plot_coins(df_combined, "coin")


# ----------- Cumulative profit line chart -------------
df["pair_str"] = df["pair"].astype(str)
# box_plot_coins(df, "pair_str")
df.sort_values(by="close_date", inplace=True)
df["extra_fees"] = 0
df["net_profit"] = df["profit"] - df["extra_fees"]
df["net_profit"] = df["profit"].apply(lambda x: max(x, -30))
df["cumulative_profit"] = df["net_profit"].cumsum()

fig, ax = plt.subplots(figsize=(12, 6))

sns.lineplot(
    x="close_date",
    y="cumulative_profit",
    data=df,
    ax=ax,
    color="blue",
    legend=False,  # We'll handle the legend/colorbar manually
)

points = sns.scatterplot(
    x="close_date",
    y="cumulative_profit",
    data=df,
    hue="open_direction",
    palette="viridis",
    edgecolor="black",
    s=60,
    ax=ax,
    legend=True,
)

plt.title("Cumulative Profit Over Each Row")
plt.xlabel("Row Index")
plt.ylabel("Cumulative Profit")
plt.grid(True)
plt.show()


df["expected_profit_error"] = df["open_expected_profit"] - df["profit"]
# df_stop_loss = df.loc[df["close_reason"] == "stop_loss"]
df_sucess = df.loc[df["close_reason"] == "crossed_mean"]

profit_error_by_open_direction = df_sucess.groupby("open_direction")[
    "expected_profit_error"
].mean()
profit_error_by_open_direction.plot(kind="bar")
plt.xlabel("Open Direction")
plt.ylabel("Expected Profit Error")
plt.title("Expected Profit Error by Open Direction")
plt.show()

# unprofitable_df = df.loc[df["profit"] <= -20]
# df = df.loc[df["close_reason"] == "crossed_mean"]
sns.lmplot(
    data=df,
    x="open_expected_profit",
    y="profit",
    hue="open_direction",
    height=6,
    aspect=1.5,
    scatter_kws={"s": 10},
)
plt.title("Scatter Plot of Price Ratio vs Profit with Separate Trend Lines")
plt.xlabel("Expected")
plt.ylabel("Profit")
plt.show()

sns.lmplot(
    data=df,
    x="volume_ratio",
    y="profit",
    hue="open_direction",
    height=6,
    aspect=1.5,
    scatter_kws={"s": 10},
)
plt.title("Scatter Plot of Price Ratio vs Profit with Separate Trend Lines")
plt.xlabel("volume_ratio")
plt.ylabel("Profit")
plt.show()

sns.lmplot(
    data=df,
    x="volatility_ratio",
    y="profit",
    hue="open_direction",
    height=6,
    aspect=1.5,
    scatter_kws={"s": 10},
)
plt.title("Scatter Plot of Price Ratio vs Profit with Separate Trend Lines")
plt.xlabel("volatility_ratio")
plt.ylabel("Profit")
plt.show()

df["multi_volatility"] = df["volume_ratio"] * abs(df["volatility_ratio"])
df["div_volatility"] = df["volume_ratio"] / abs(df["volatility_ratio"])

sns.lmplot(
    data=df,
    x="multi_volatility",
    y="profit",
    hue="open_direction",
    height=6,
    aspect=1.5,
    scatter_kws={"s": 10},
)
plt.title("Scatter Plot of Price Ratio vs Profit with Separate Trend Lines")
plt.xlabel("multi_volatility")
plt.ylabel("Profit")
plt.show()

sns.lmplot(
    data=df,
    x="div_volatility",
    y="profit",
    hue="open_direction",
    height=6,
    aspect=1.5,
    scatter_kws={"s": 10},
)
plt.title("Scatter Plot of Price Ratio vs Profit with Separate Trend Lines")
plt.xlabel("div_volatility")
plt.ylabel("Profit")
plt.show()
