import requests
import time
import hashlib
import hmac
import bitmex
import json

bitmex_api_key = "Hi7WUgzxyzCRY_1BJ0e7meab"
bitmex_api_secret = "A64KwkbRqURgFAmarfF758ceAVtFoIJWZIXe5lpRIeld1FxD"

coinbase_API_key = "cTgYvXpaksr5fFgr"
coinbase_API_secret = "Css2cMN9kTjPNh2XvuHLM9HrdVcX3ty5"


def fetch_bitcoin_price():
    client = bitmex.bitmex(
        test=False, api_key=bitmex_api_key, api_secret=bitmex_api_secret
    )

    positions = client.Instrument.Instrument_get(
        filter=json.dumps({"symbol": "XBTUSD"})
    ).result()[0][0]
    bitmex_btc = {}
    # print(positions['lastPrice'])
    return positions["lastPrice"]


def connect_apis():
    client = bitmex.bitmex(api_key=bitmex_api_key, api_secret=bitmex_api_secret)
    clientb = Client(coinbase_API_key, coinbase_API_secret)

    while True:
        positions = client.Position.Position_get(
            filter=json.dumps({"symbol": "XBTUSD"})
        ).result()[0][0]
        bitmex_btc = {}

        bitmex_btc["markPrice"] = positions["markPrice"]
        print("BitMex: ", bitmex_btc["markPrice"])

        coinbase_btc = clientb.get_spot_price(currency_pair="BTC-USD")
        print("Coinbase: ", coinbase_btc["amount"])

        percent = float(
            ((float(coinbase_btc["amount"]) - bitmex_btc["markPrice"]) * 100)
            / bitmex_btc["markPrice"]
        )

        sleep(1)

        if percent < 1.5:
            print("No arbitrage possibility")
            continue

        else:
            if percent == 1.5:
                print("ARBITRAGE TIME")
                break
        sleep(1)
