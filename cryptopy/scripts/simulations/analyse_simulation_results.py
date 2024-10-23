import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from cryptopy.scripts.simulations.simulation_helpers import read_from_json

matplotlib.use("TkAgg")  # Or another backend like 'Qt5Agg' depending on your system


simulation_name = "default_parameters"
simulation_path = f"../../data/simulations/{simulation_name}.json"

json_data = read_from_json(simulation_path)

df = pd.DataFrame(json_data["trade_events"])
df["coin_1"] = df["pair"].loc[0]
df["coin_2"] = df["pair"].loc[1]

print(df)
print(df.groupby("close_reason")["profit"].count())
print(df.groupby("close_reason")["profit"].sum())
df_success = df.loc[df["close_reason"] == "crossed_mean"]
df_success["open_days"] = pd.to_datetime(df_success["close_date"]) - pd.to_datetime(
    df_success["open_date"]
)
print("mean days of successful position", df_success["open_days"].mean())

profits_by_pair = df.groupby(["coin1", "coin2"])["profit"].sum()
profits_by_pair.hist()
plt.xlabel("Profit")
plt.ylabel("Frequency")
plt.title("Histogram of Profits by Pairs")
plt.show()
