import pandas as pd
from matplotlib import pyplot as plt
from cryptopy.scripts.simulation_helpers import read_from_json

simulation_name = "default_parameters"
simulation_path = f"../../data/simulations/{simulation_name}.json"

json_data = read_from_json(simulation_path)

df = pd.DataFrame(json_data["trade_events"])
print(df)
print(df.groupby("close_reason")["profit"].count())
print(df.groupby("close_reason")["profit"].sum())
df_success = df.loc[df["close_reason"] == "crossed_mean"]
df_success["open_days"] = pd.to_datetime(df_success["close_date"]) - pd.to_datetime(
    df_success["open_date"]
)
print("mean days of successful position", df_success["open_days"].mean())
