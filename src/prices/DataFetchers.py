from abc import abstractmethod
from datetime import datetime, timedelta
from dateutil.tz import tzutc
import time
import pytz

import pandas as pd

import ccxt.async_support as ccxt
from ccxt.base.errors import RateLimitExceeded

from src.prices.OHLCData import OHLCData
import logging
import asyncio


class DataFetcher:
    def __init__(self, client, exchange_name, pairs_mapping, markets):
        self.client = client
        self.exchange_name = exchange_name
        self.currencies = pairs_mapping
        self.currency_fees = {}
        self.exchange_fees = {}
        self.historical_data = {}
        self.live_data = {}
        self.market_symbols = []
        self.timeout = 10
        self.markets = markets

    def extract_exchange_fees(self):
        exchange_fees = self.client.fees["funding"]
        for fee_type in ["withdrawal", "deposit"]:
            if exchange_fees.get(fee_type, 0) == {}:
                exchange_fees[fee_type] = 0
        self.exchange_fees = {
            "withdrawal": exchange_fees.get("withdrawal", 0),
            "deposit": exchange_fees.get("deposit", 0),
        }

    async def extract_currency_fees(self):
        for currency, symbol in self.currencies.items():
            options = [currency, symbol, currency + ":BTC", symbol + ":BTC"]
            trading_fee = self.extract_currency_fee_simple(options)

            if trading_fee is None:
                trading_fee = await self.extract_currency_fee_complex(options)

            self.update_currency_fee(currency, trading_fee)
        # print(self.exchange_name, self.currency_fees)

    def extract_currency_fee_simple(self, options):
        for option in options:
            trading_fee = self.markets.get(option, None)
            if trading_fee:
                return trading_fee
        return None

    async def extract_currency_fee_complex(self, options):
        trading_fee = None
        for option in options:
            try:
                trading_fee = await self.client.fetch_trading_fee(option)
                break
            except Exception as e:
                trading_fee = self.markets.get(option, None)
                if trading_fee:
                    break
        return trading_fee

    def update_currency_fee(self, currency, trading_fee):
        if trading_fee is not None:
            self.currency_fees[currency] = {
                "maker": trading_fee.get("maker", 0),
                "taker": trading_fee.get("taker", 0),
            }

    def get_currency_fee(self, currency):
        return self.currency_fees.get(currency, 0)

    def get_exchange_fees(self):
        return self.exchange_fees

    async def async_init(self):
        # await self.fetch_all_initial_live_prices(count=10)
        self.market_symbols = self.client.symbols
        self.extract_exchange_fees()
        await self.extract_currency_fees()
        await self.update_all_historical_prices()

    def initialize_ohlc_data(self, currency):
        self.historical_data[currency] = OHLCData()
        self.live_data[currency] = OHLCData()

    def get_historical_prices(self, currency):
        return self.historical_data.get(currency)

    def get_live_prices(self, currency):
        return self.live_data.get(currency)

    def get_all_live_prices(self):
        return {
            currency: prices.close[-1] for currency, prices in self.live_data.items()
        }

    def get_fees(self, currency):
        return self.currency_fees.get(currency)

    def get_all_currency_fees(self):
        return self.currency_fees

    def get_all_exchange_fees(self):
        return self.exchange_fees

    async def fetch_all_initial_live_prices(self, count=10):
        tasks = [
            self.fetch_initial_live_prices(currency, count)
            for currency in self.currencies.keys()
        ]
        await self._gather_with_timeout(tasks, "fetch_all_initial_live_prices")

    async def update_all_historical_prices(self):
        tasks = [
            self.update_historical_prices(currency)
            for currency in self.currencies.keys()
        ]
        await self._gather_with_timeout(tasks, "update_all_historical_prices")

    async def fetch_initial_live_prices(self, currency, count):
        symbol = self.currencies[currency]

        trades_df = await self.fetch_trades_within_timeframe(symbol)
        # Bucket trades into 10-second intervals and aggregate into OHLCV
        ohlcv_df = self.bucket_trades_into_intervals(
            trades_df, interval="10s"
        ).reset_index()

        if currency not in self.live_data.keys():
            self.initialize_ohlc_data(currency)

        self.live_data[currency].update_from_dataframe(ohlcv_df)

    async def fetch_trades_within_timeframe(self, symbol):
        now = datetime.now(pytz.timezone("Europe/London"))

        since = int((now - timedelta(minutes=30)).timestamp() * 1000)
        until = int(now.timestamp() * 1000)
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
        # print(data)
        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    @staticmethod
    def bucket_trades_into_intervals(df, interval="10S"):

        # Resample trades into intervals and aggregate into OHLCV
        ohlcv = df.resample(interval, on="datetime").agg(
            {"price": ["first", "max", "min", "last"], "amount": "sum"}
        )
        ohlcv.columns = ["open", "high", "low", "close", "volume"]

        # Forward-fill missing data
        ohlcv["open"] = ohlcv["open"].ffill()
        ohlcv["high"] = ohlcv["high"].ffill()
        ohlcv["low"] = ohlcv["low"].ffill()
        ohlcv["close"] = ohlcv["close"].ffill()
        ohlcv["volume"] = ohlcv["volume"].fillna(0)

        return ohlcv

    async def fetch_all_live_prices(self):
        tasks = [self.fetch_live_price(currency) for currency in self.currencies.keys()]
        await self._gather_with_timeout(tasks, "fetch_all_live_prices")

    async def fetch_live_price(self, currency):
        # print("fetching", currency)
        symbol = self.currencies[currency]

        try:
            ticker = await asyncio.wait_for(
                self.client.fetch_ticker(symbol), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            print(
                f"{self.exchange_name}: Timeout while fetching live price for {currency}"
            )
            return
        except Exception as e:
            print(
                f"{self.exchange_name}: Error while fetching live price for {currency}: {e}"
            )
            return

        timestamp_ms = ticker["timestamp"]
        if timestamp_ms is None:
            datetime_obj = datetime.now()
        else:
            timestamp_s = timestamp_ms / 1000
            datetime_obj = datetime.fromtimestamp(timestamp_s)

        if currency not in self.live_data.keys():
            self.initialize_ohlc_data(currency)

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

    async def _gather_with_timeout(self, tasks, task_name):
        await asyncio.wait_for(asyncio.gather(*tasks), self.timeout)

        # try:
        #     await asyncio.wait_for(asyncio.gather(*tasks), self.timeout)
        # except asyncio.TimeoutError:
        #     print(f"{self.exchange_name}: Timeout during {task_name}")
        # except Exception as e:
        #     print(f"{self.exchange_name}: Error during {task_name}: {e}")
