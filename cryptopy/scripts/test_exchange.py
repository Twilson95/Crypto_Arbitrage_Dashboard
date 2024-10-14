import ccxt
import yaml

with open("../config/exchange_config.yaml", "r") as f:
    config = yaml.safe_load(f)

exchange_name = "Bitmex"
exchange_name = "HTX"
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

for symbol, market_details in markets.items():
    if "tao" in symbol.lower():
        print(market_details)

# for currency, symbol in pairs_mapping.items():
#     try:
#         trading_fee = exchange.fetch_trading_fee(currency)
#         print(trading_fee.get("maker", None))
#         print(trading_fee.get("taker", None))
#         print(currency, symbol, "trading_fee", trading_fee)
#     except:
#         print("failed to find fees", currency, symbol)
