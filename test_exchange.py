import ccxt
import yaml

with open("./src/config/exchange_config.yaml", "r") as f:
    config = yaml.safe_load(f)

exchange_name = "Bitmex"
exchange_name = "Kraken"
api_key = config[exchange_name]["api_key"]
api_secret = config[exchange_name]["api_secret"]
pairs_mapping = config[exchange_name]["pairs"]

api_secret = api_secret.replace("\\n", "\n").strip()

exchange_class = getattr(ccxt, exchange_name.lower())
exchange = exchange_class(
    {
        "apiKey": api_key,
        "secret": api_secret,
    }
)
markets = exchange.load_markets()

# print(exchange.fetch_withdrawal_fees())


exchange_fees = exchange.fees
exchange_fees["withdrawal"]
