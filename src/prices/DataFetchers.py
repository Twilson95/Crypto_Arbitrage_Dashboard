from abc import abstractmethod
from datetime import datetime, timedelta
from dateutil.tz import tzutc
import time
import pandas as pd
from bitmex import bitmex

# import ccxt
import ccxt.async_support as ccxt
from ccxt.base.errors import RateLimitExceeded

from src.prices.OHLCData import OHLCData
import logging
import asyncio


class DataFetcher:
    def __init__(self, client):
        self.client = client
        self.currencies = {
            "bitcoin": "BTC/USD",
            "ethereum": "ETH/USD",
            "solana": "SOL/USD",
            "dogecoin": "DOGE/USD",
            "cardano": "ADA/USD",
            "ripple": "XRP/USD",
        }
        self.historical_data = {}
        self.live_data = {}

    async def async_init(self):
        await self.fetch_all_initial_live_prices(count=10)
        await self.update_all_historical_prices()

    def initialize_ohlc_data(self, currency):
        self.historical_data[currency] = OHLCData()
        self.live_data[currency] = OHLCData()

    def get_historical_prices(self, currency):
        return self.historical_data[currency]

    def get_live_prices(self, currency):
        return self.live_data[currency]

    async def fetch_all_live_prices(self):
        tasks = [self.fetch_live_price(currency) for currency in self.currencies.keys()]
        await asyncio.gather(*tasks)
        # time.sleep(0.1)

    async def fetch_all_initial_live_prices(self, count=10):
        tasks = [
            self.fetch_initial_live_prices(currency, count)
            for currency in self.currencies.keys()
        ]
        await asyncio.gather(*tasks)

    async def update_all_historical_prices(self):
        tasks = [
            self.update_historical_prices(currency)
            for currency in self.currencies.keys()
        ]
        await asyncio.gather(*tasks)

    async def fetch_initial_live_prices(self, currency, count):
        symbol = self.currencies[currency]
        since = int((datetime.now() - timedelta(seconds=1000)).timestamp() * 1000)
        trades_df = await self.fetch_trades_within_timeframe(symbol, since)
        # Bucket trades into 10-second intervals and aggregate into OHLCV
        ohlcv_df = self.bucket_trades_into_intervals(
            trades_df, interval="10s"
        ).reset_index()

        if currency not in self.historical_data.keys():
            self.initialize_ohlc_data(currency)

        self.historical_data[currency].update_from_dataframe(ohlcv_df)

    async def fetch_trades_within_timeframe(self, symbol, since):
        until = self.client.parse8601(datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
        trades = await self.client.fetch_trades(
            symbol, since=since, params={"until": until}
        )
        data = [
            {
                "timestamp": trade["timestamp"],
                "price": trade["price"],
                "amount": trade["amount"],
            }
            for trade in trades
        ]
        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    @staticmethod
    def bucket_trades_into_intervals(df, interval="10S"):

        # Resample trades into intervals and aggregate into OHLCV
        ohlcv = df.resample(interval, on="datetime").agg(
            {"price": ["first", "max", "min", "last"], "amount": "sum"}
        )
        # Rename columns
        ohlcv.columns = ["open", "high", "low", "close", "volume"]

        # Forward-fill missing data
        ohlcv["open"] = ohlcv["open"].ffill()
        ohlcv["high"] = ohlcv["high"].ffill()
        ohlcv["low"] = ohlcv["low"].ffill()
        ohlcv["close"] = ohlcv["close"].ffill()
        ohlcv["volume"] = ohlcv["volume"].fillna(0)

        return ohlcv

    async def fetch_live_price(self, currency):
        symbol = self.currencies[currency]
        ticker = await self.client.fetch_ticker(symbol)
        timestamp_ms = ticker["timestamp"]
        timestamp_s = timestamp_ms / 1000
        datetime_obj = datetime.fromtimestamp(timestamp_s)

        current_price = ticker["last"]
        self.live_data[currency].datetime.append(datetime_obj)
        self.live_data[currency].high.append(current_price)
        self.live_data[currency].low.append(current_price)
        self.live_data[currency].close.append(current_price)
        self.live_data[currency].open.append(current_price)

    async def update_historical_prices(self, currency):
        symbol = self.currencies[currency]
        timeframe = "1d"  # Daily data
        since = self.client.parse8601(
            (datetime.today() - timedelta(days=100)).strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        try:
            data = await self.client.fetch_ohlcv(
                symbol,
                timeframe,
                since,  # params={"until": 1719014400 * 1000}
            )
        except RateLimitExceeded as e:
            print(f"Rate limit exceeded: {e}")
            # Handle the rate limit case as needed, e.g., by waiting or retrying later
            return None

        df = pd.DataFrame(
            data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        if currency not in self.historical_data.keys():
            self.initialize_ohlc_data(currency)

        self.historical_data[currency].update_from_dataframe(df)
