import pandas as pd
import os

exchange_name = "Kraken"

input_folder_1 = f"../../../data/historical_data/{exchange_name}_300_days/"
input_folder_2 = f"../../../data/historical_data/{exchange_name}/"
save_to_folder = f"../../../data/historical_data/{exchange_name}_long_history/"

cointegration_pairs_path = f"../../../data/historical_data/cointegration_pairs.csv"

pair_combinations_df = pd.read_csv(cointegration_pairs_path)
pair_combinations = list(pair_combinations_df.itertuples(index=False, name=None))
coin_set = set(
    [pair[0] for pair in pair_combinations] + [pair[1] for pair in pair_combinations]
)
coin_set = {coin.replace("/", "_") for coin in coin_set}

print(coin_set)

for coin in coin_set:
    df_1 = pd.read_csv(os.path.join(input_folder_1, coin + ".csv"), index_col=0)
    df_2 = pd.read_csv(os.path.join(input_folder_2, coin + ".csv"), index_col=0)

    combined_df = pd.concat([df_1, df_2]).loc[
        ~pd.concat([df_1, df_2]).index.duplicated(keep="first")
    ]

    combined_df.to_csv(os.path.join(save_to_folder, coin + ".csv"))
