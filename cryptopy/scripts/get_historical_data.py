import os
import pandas as pd
import ccxt
import yaml

# Load exchange config from yaml file
with open("../config/exchange_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Setup exchange connection
exchange_name = "Kraken"
api_key = config[exchange_name]["api_key"]
api_secret = config[exchange_name]["api_secret"]
pairs_mapping = config[exchange_name]["pairs"]  # Map currency pairs to symbols

# Initialize exchange with credentials
api_secret = api_secret.replace("\\n", "\n").strip()
exchange_class = getattr(ccxt, exchange_name.lower())
exchange = exchange_class(
    {
        "apiKey": api_key,
        "secret": api_secret,
    }
)
markets = exchange.load_markets()

# Define the folder where historical data is stored
historic_data_folder = f"../../data/historical_data/{exchange_name}/"


# Helper function to read all files in a directory
def read_files_from(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ]


# Helper function to find the earliest date in a dataframe
def find_earliest_date(df):
    return pd.to_datetime(df["datetime"].min())


# Loop through all files in the historic data folder
files = read_files_from(historic_data_folder)
for file in files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)

    # Ensure 'datetime' column is treated as a datetime type
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Determine the earliest date in the file and calculate 100 days before that
    earliest_date = find_earliest_date(df)
    start_date = earliest_date - pd.Timedelta(days=100)

    # Convert start_date to a timestamp in milliseconds (required by ccxt)
    since_timestamp = int(start_date.timestamp() * 1000)

    # Extract the currency pair from the filename and replace '_' with '/' (e.g., BTC_USD -> BTC/USD)
    currency_pair = os.path.basename(file).replace(".csv", "").replace("_", "/")

    # Use the pairs_mapping dictionary to get the correct symbol for the query
    if currency_pair in pairs_mapping:
        symbol = pairs_mapping[currency_pair]
    else:
        print(
            f"Symbol not found for {currency_pair} in pairs_mapping. Skipping file: {file}"
        )
        continue

    # Specify the timeframe and fetch new data from the exchange
    timeframe = "1d"  # Adjust this to the appropriate timeframe (e.g., '1h', '1d')
    new_data = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=100)

    # Create a new DataFrame from the fetched data
    new_data_df = pd.DataFrame(
        new_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    # Convert timestamps to ISO 8601 datetime format with UTC timezone
    new_data_df["datetime"] = pd.to_datetime(
        new_data_df["timestamp"], unit="ms", utc=True
    )

    # Drop the 'timestamp' column since we already have 'datetime'
    new_data_df = new_data_df.drop(columns=["timestamp"])

    # Ensure that the column order matches the existing CSV (datetime, open, high, low, close, volume)
    new_data_df = new_data_df[["datetime", "open", "high", "low", "close", "volume"]]

    # Remove any potential duplicate rows based on the 'datetime' column
    combined_df = (
        pd.concat([df, new_data_df])
        .drop_duplicates(subset=["datetime"])
        .sort_values(by="datetime")
    )

    # Write the updated DataFrame back to the file
    combined_df.to_csv(file, index=False)

    print(f"Updated file: {file}")
