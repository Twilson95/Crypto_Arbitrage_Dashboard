import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from cryptopy.scripts.simulations.analysis_helpers import (
    box_plot_coins,
    scatter_plot_with_trend,
)
from cryptopy.scripts.simulations.simulation_helpers import read_from_json

matplotlib.use("TkAgg")  # Or another backend like 'Qt5Agg' depending on your system

simulation_name = "portfolio_sim_dynamic_trade_amount"
simulation_path = f"../../../data/simulations/{simulation_name}.json"

json_data = read_from_json(simulation_path)
df = pd.DataFrame(json_data["trade_events"])

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
profits_per_day = df.groupby(["open_days"])["profit"].sum()
profits_per_day.plot()
plt.xlabel("open_days")
plt.ylabel("profit")
plt.title("profit per open day")
plt.show()
#
profits_by_pair = df.groupby(["coin_1", "coin_2"])["profit"].sum()
print(profits_by_pair)
profits_by_pair.hist(bins=30)
plt.xlabel("Profit")
plt.ylabel("Frequency")
plt.title("Histogram of Profits by Pairs")
plt.show()

# scatter_plot_with_trend(df)
print(df.groupby(["open_direction"])["profit"].sum())

df["coin"] = df["coin_1"]
df_copy = df.copy()
df_copy["coin"] = df["coin_2"]
df_combined = pd.concat([df, df_copy])
box_plot_coins(df_combined, "coin")

df["pair_str"] = df["pair"].astype(str)
# box_plot_coins(df, "pair_str")

df["extra_fees"] = 2
df["net_profit"] = df["profit"] - df["extra_fees"]
df["cumulative_profit"] = df["net_profit"].cumsum()
plt.figure(figsize=(12, 6))
plt.plot(
    df.index,
    df["cumulative_profit"],
    marker="o",
    linestyle="-",
    color="blue",
)
plt.title("Cumulative Profit Over Each Row")
plt.xlabel("Row Index")
plt.ylabel("Cumulative Profit")
plt.grid(True)
plt.show()
