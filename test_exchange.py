import ccxt
import yaml

with open("./src/config/exchange_config.yaml", "r") as f:
    config = yaml.safe_load(f)

exchange_name = "Bitmex"
exchange_name = "Coinbase"
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

for currency, symbol in pairs_mapping.items():
    try:
        trading_fee = exchange.fetch_trading_fee(currency)
        print(trading_fee.get("maker", None))
        print(trading_fee.get("taker", None))
        print(currency, symbol, "trading_fee", trading_fee)
    except:
        print("failed to find fees", currency, symbol)


# Print fee details for each instrument
# for symbol, market in markets.items():
#     print(market)
# maker_fee = market.get("maker", "N/A")
# taker_fee = market.get("taker", "N/A")
# print(f"Symbol: {symbol}, Maker fee: {maker_fee}, Taker fee: {taker_fee}")


print(exchange.fees)

# import requests
#
# # BitMEX API endpoint for fee information
# url = "https://www.bitmex.com/api/v1/instrument/active"
#
# response = requests.get(url)
# instruments = response.json()
#
# # Print fee details for each instrument
# for instrument in instruments:
#     symbol = instrument["symbol"]
#     maker_fee = instrument.get("makerFee", "N/A")
#     taker_fee = instrument.get("takerFee", "N/A")
#     print(f"Symbol: {symbol}, Maker fee: {maker_fee}, Taker fee: {taker_fee}")
