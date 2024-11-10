from datetime import datetime, timedelta
from itertools import permutations
import pytz
import re
from time import time
import pandas as pd
import asyncio
import os
import ccxt.async_support as ccxt_async

from cryptopy import OHLCData
from cryptopy import CointegrationCalculator


class DataFetcher:
    def __init__(self, client, exchange_name, pairs_mapping, markets, use_cache=True):
        self.client = client
        self.exchange_name = exchange_name
        self.currencies = pairs_mapping
        self.inter_coin_symbols = None
        self.currency_fees = {}
        self.exchange_fees = {}
        self.historical_data = dict()
        self.live_data = dict()
        self.use_cache = use_cache
        self.cointegration_pairs = {}
        self.order_books = {}
        self.market_symbols = []
        self.timeout = 60
        self.markets = markets
        self.caching_folder = f"data/historical_data/{self.exchange_name}_long_history/"
        self.balance = None
        self.open_orders = None

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
                print(self.exchange_name, "extract_complex_fees")
                trading_fee = await self.extract_currency_fee_complex(options)

            self.update_currency_fee(currency, trading_fee)

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

        self.inter_coin_symbols = self.generate_inter_coin_symbols(
            self.currencies, self.market_symbols
        )

        self.extract_exchange_fees()
        await self.extract_currency_fees()

        self.currency_fees = DataFetcher.generate_crypto_to_crypto_fees(
            self.currency_fees, self.inter_coin_symbols
        )
        await self.set_balance()
        await self.set_open_trades()

    def update_cointegration_pairs(self):
        start_time = time()
        df = self.get_df_of_all_historical_data()

        self.cointegration_pairs = CointegrationCalculator.find_cointegration_pairs(
            df, self.cointegration_pairs
        )

        end_time = time()
        print(f"{(end_time - start_time):.2f} seconds to generate cointegrity pairs")

    def initialize_live_data(self, currency):
        self.live_data[currency] = OHLCData()

    def initialize_historic_data(self, currency):
        self.historical_data[currency] = OHLCData()

    def get_historical_prices(self, currency):
        return self.historical_data.get(currency)

    def get_live_prices(self, currency):
        return self.live_data.get(currency)

    def get_live_data(self):
        return self.live_data

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

    async def update_all_historical_prices(self, batch_size=6, wait_time=10):
        currencies = list(self.currencies.keys())
        for i in range(0, len(currencies), batch_size):
            batch = currencies[i : i + batch_size]
            tasks = [self.update_historical_prices(currency) for currency in batch]
            await self._gather_with_timeout(tasks, "update_all_historical_prices")
            self.update_cointegration_pairs()

        # self.output_cointegration_pairs(
        #     f"data/historical_data/cointegration_pairs.csv"
        # )

        # Add a delay if there are more batches
        # if i + batch_size < len(currencies):
        #     print(f"Waiting for {wait_time} seconds before the next batch...")
        #     await asyncio.sleep(wait_time)

    def output_cointegration_pairs(self, path):
        pairs = self.cointegration_pairs.keys()
        df = pd.DataFrame(pairs, columns=["coin_1", "coin_2"])
        df.to_csv(path, index=False)

    async def fetch_initial_live_prices(self, currency, count):
        symbol = self.currencies[currency]

        trades_df = await self.fetch_trades_within_timeframe(symbol)
        # Bucket trades into 10-second intervals and aggregate into OHLCV
        ohlcv_df = self.bucket_trades_into_intervals(
            trades_df, interval="10s"
        ).reset_index()

        if currency not in self.live_data.keys():
            self.initialize_live_data(currency)

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

    @staticmethod
    def create_batches(data_dict, batch_size):
        items = list(data_dict.items())
        return [
            dict(items[i : i + batch_size]) for i in range(0, len(items), batch_size)
        ]

    async def fetch_all_live_prices(self, batch_size=20, delay_time=0):
        # Collect all currencies (union of dictionaries)
        all_currencies = {**self.currencies, **self.inter_coin_symbols}
        if not all_currencies:
            return

        # If the exchange is not Bybit and does not support fetchTickers, handle it separately
        if self.exchange_name != "Bybit" and not self.client.has.get("fetchTickers"):
            tasks = [
                self.fetch_live_price(currency, symbol)
                for currency, symbol in all_currencies.items()
            ]
            await self._gather_with_timeout(tasks, "fetch_all_live_prices_individual")
            return

        batches = []

        if self.exchange_name == "Bybit":
            # Batch self.currencies and self.inter_coin_symbols separately
            currency_batches = DataFetcher.create_batches(self.currencies, batch_size)
            intercoin_batches = DataFetcher.create_batches(
                self.inter_coin_symbols, batch_size
            )
            batches.extend(currency_batches)
            batches.extend(intercoin_batches)

        elif self.client.has.get("fetchTickers"):
            all_currency_batches = DataFetcher.create_batches(
                all_currencies, batch_size
            )
            batches.extend(all_currency_batches)

        for batch_number, batch in enumerate(batches):
            # print(
            # f"Running live data batch {batch_number + 1}/{len(batches)} with {len(batch)} items"
            # )

            tasks = [self.fetch_live_price_multiple(batch)]

            try:
                await self._gather_with_timeout(
                    tasks, f"fetch_all_live_prices_batch_{batch_number}"
                )
            except Exception as e:
                print(f"Error occurred while processing batch {batch_number + 1}: {e}")

            # Optionally, add a delay between batches to prevent overloading
            if batch_number + 1 < len(batches):
                # print(f"Waiting {delay_time} seconds before the next batch...")
                await asyncio.sleep(delay_time)

    async def fetch_live_price(self, currency, symbol):
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
            self.initialize_live_data(currency)

        current_price = ticker["last"]
        self.live_data[currency].datetime.append(datetime_obj)
        self.live_data[currency].high.append(current_price)
        self.live_data[currency].low.append(current_price)
        self.live_data[currency].close.append(current_price)
        self.live_data[currency].open.append(current_price)

        if currency in self.historical_data.keys():
            self.historical_data[currency].update_with_latest_value(
                current_price, datetime_obj
            )

    async def fetch_live_price_multiple(self, currencies):
        symbols = list(currencies.values())

        try:
            tickers = await asyncio.wait_for(
                self.client.fetch_tickers(symbols), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            print(f"{self.exchange_name}: Timeout while fetching live price multiple")
            return
        except Exception as e:
            print(
                f"{self.exchange_name}: Error while fetching live price multiple: {e}"
            )
            return

        for key, ticker in tickers.items():
            currency = key.split(":")[0]
            currency = currency.replace("USDT", "USD")

            timestamp_ms = ticker["timestamp"]
            if timestamp_ms is None:
                datetime_obj = datetime.now()
            else:
                timestamp_s = timestamp_ms / 1000
                datetime_obj = datetime.fromtimestamp(timestamp_s)

            if currency not in self.live_data.keys():
                self.initialize_live_data(currency)

            current_price = ticker["last"]
            self.live_data[currency].datetime.append(datetime_obj)
            self.live_data[currency].high.append(current_price)
            self.live_data[currency].low.append(current_price)
            self.live_data[currency].close.append(current_price)
            self.live_data[currency].open.append(current_price)

            if currency in self.historical_data.keys():
                self.historical_data[currency].close[-1] = current_price

    # async def update_historical_prices(self, currency):
    #     symbol = self.currencies[currency]
    #     timeframe = "1d"  # Daily data
    #     since = self.client.parse8601(
    #         (datetime.today() - timedelta(days=100)).strftime("%Y-%m-%dT%H:%M:%SZ")
    #     )
    #     try:
    #         data = await asyncio.wait_for(
    #             self.client.fetch_ohlcv(symbol, timeframe, since), timeout=self.timeout
    #         )
    #     except asyncio.TimeoutError:
    #         print(f"Timeout for fetching {currency}. Skipping.")
    #         return None
    #     except RateLimitExceeded as e:
    #         print(f"Rate limit exceeded: {e}")
    #         return None
    #     except Exception as e:
    #         print(f"Error fetching {currency}: {e}")
    #         return None
    #
    #     df = pd.DataFrame(
    #         data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    #     )
    #     df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    #
    #     if currency not in self.historical_data.keys():
    #         self.initialize_historic_data(currency)
    #
    #     self.historical_data[currency].update_from_dataframe(df)

    async def update_historical_prices(self, currency):
        symbol = self.currencies[currency]
        timeframe = "1d"
        days_to_fetch = 100

        if currency not in self.historical_data.keys():
            self.initialize_historic_data(currency)

        if self.use_cache:
            cached_df, since, missing_days = await self.check_for_cached_data(
                currency, days_to_fetch
            )
            self.historical_data[currency].update_from_dataframe(cached_df)
        else:
            missing_days = days_to_fetch
            since = self.client.parse8601(
                (datetime.today() - timedelta(days=days_to_fetch)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            )

        if missing_days <= 0:
            return

        new_data = await self.query_historical_data(symbol, timeframe, since)
        if new_data is None:
            return

        self.historical_data[currency].update_from_dataframe(new_data)
        self.cache_historical_data(currency)

    async def check_for_cached_data(self, currency, days_to_fetch):
        """
        Check if cached data exists and calculate how many days of data are missing.
        Returns cached dataframe, since timestamp, and missing days.
        """
        safe_currency = currency.replace("/", "_")
        file_name = f"{safe_currency}.csv"
        file_path = os.path.join(self.caching_folder, file_name)
        print(f"caching check for {currency}")
        if os.path.exists(file_path):
            cached_df = pd.read_csv(file_path, parse_dates=["datetime"])
            latest_cached_date = cached_df["datetime"].max().date()
            print(
                f"Cached data found for {currency}, latest date: {latest_cached_date}"
            )
        else:
            print(f"No cached data available for {currency}")
            cached_df = pd.DataFrame()
            latest_cached_date = None
            # print(f"No cached data for {currency}")

        if latest_cached_date is None:

            # No cached data, fetch the last `days_to_fetch` days
            since = self.client.parse8601(
                (datetime.today() - timedelta(days=days_to_fetch)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            )
            missing_days = days_to_fetch
        else:
            missing_days = (datetime.today().date() - latest_cached_date).days
            since = self.client.parse8601(
                (latest_cached_date + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            )

        return cached_df, since, missing_days

    async def query_historical_data(self, symbol, timeframe, since):
        try:
            data = await asyncio.wait_for(
                self.client.fetch_ohlcv(symbol, timeframe, since), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            print(f"Timeout for fetching {symbol}. Skipping.")
            return None
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None

        if data is None:
            return None

        df = pd.DataFrame(
            data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def cache_historical_data(self, currency):
        safe_currency = currency.replace("/", "_")
        file_name = f"{safe_currency}.csv"
        file_path = os.path.join(self.caching_folder, file_name)
        if not os.path.exists(self.caching_folder):
            os.makedirs(self.caching_folder, exist_ok=True)
        historical_ohlc = self.get_historical_prices(currency)
        historical_df = historical_ohlc.to_dataframe()
        if historical_df.empty:
            print(f"{currency} is empty")
            return

        historical_df.to_csv(file_path, index=False)

    async def _gather_with_timeout(self, tasks, task_name):
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=self.timeout)
        except asyncio.TimeoutError:
            print(f"Timeout occurred while gathering tasks for {task_name}")
            # Optionally handle the timeout, retry, or cancel tasks
        except Exception as e:
            print(f"Exception in {task_name}: {e}")
            # Optionally handle other exceptions

    @staticmethod
    def generate_inter_coin_symbols(currencies, market_symbols):
        # Determine if symbols in the dictionary use a delimiter (like "/")
        delimiter = "/" if any("/" in symbol for symbol in currencies.values()) else ""

        # Extract base currencies and the quote currency
        base_currencies = []
        quote_currency = None
        for key in currencies.keys():
            base, quote = key.split("/")
            base_currencies.append(base)
            quote_currency = quote  # Assuming all have the same quote currency

        # Generate all permutations of the base currencies for inter-coin pairs
        inter_coin_pairs = list(permutations(base_currencies, 2))

        # Create a new dictionary to hold the synthetic pairs and include the original pairs
        inter_coin_symbols = {}  # Start with the original pairs

        # Generate synthetic symbols based on permutations
        for base1, base2 in inter_coin_pairs:
            pair1 = f"{base1}/{base2}"
            pair2 = f"{base2}/{base1}"

            # Extract the exchange-specific symbols for base1 and base2
            symbol1_base = currencies.get(f"{base1}/{quote_currency}")
            symbol2_base = currencies.get(f"{base2}/{quote_currency}")

            # Construct the inter-coin symbols in the same style as the original dictionary
            if delimiter:
                # Use the delimiter style (e.g., "btc/usd")
                synthetic_symbol1 = f"{symbol1_base.split(delimiter)[0]}{delimiter}{symbol2_base.split(delimiter)[0]}"
                synthetic_symbol2 = f"{symbol2_base.split(delimiter)[0]}{delimiter}{symbol1_base.split(delimiter)[0]}"
            else:
                # No delimiter style (e.g., "btcusd")
                # Remove the 'usd' part from the symbols
                symbol1_base = re.sub(
                    r"usdt|usd", "", symbol1_base, flags=re.IGNORECASE
                )
                symbol2_base = re.sub(
                    r"usdt|usd", "", symbol2_base, flags=re.IGNORECASE
                )
                synthetic_symbol1 = f"{symbol1_base}{symbol2_base}"
                synthetic_symbol2 = f"{symbol2_base}{symbol1_base}"

            matched_symbol1 = DataFetcher.find_matching_symbol(
                market_symbols, synthetic_symbol1
            )
            matched_symbol2 = DataFetcher.find_matching_symbol(
                market_symbols, synthetic_symbol2
            )

            if matched_symbol1:
                if DataFetcher.get_inverse_pair(pair1) in inter_coin_symbols.keys():
                    continue
                inter_coin_symbols[pair1] = matched_symbol1
            if matched_symbol2:
                if DataFetcher.get_inverse_pair(pair2) in inter_coin_symbols.keys():
                    continue
                inter_coin_symbols[pair2] = matched_symbol2

            # inter_coin_symbols.pop("ETHXBT")
            # all_currencies.pop("XBTETH")

        return inter_coin_symbols

    def update_all_cointegration_spreads(self):
        print("start updating all cointegration pairs from live data")
        start_time = time()

        for cointegration_data in self.cointegration_pairs.values():
            # if cointegration_data.p_value > 0.05:
            #     continue
            coin1, coin2 = cointegration_data.pair
            price1 = self.get_historical_prices(coin1).close[-1]
            price2 = self.get_historical_prices(coin2).close[-1]
            if (
                price1 is None
                or price2 is None
                or cointegration_data.hedge_ratio is None
            ):
                continue
            try:
                cointegration_data.update_latest_spread(price1, price2)
                cointegration_data.update_trade_details()
            except Exception as e:
                print(f"An error occurred: {e}")

        end_time = time()
        print(
            f"time to update all cointegration with live prices {(end_time - start_time):.2f}"
        )

    @staticmethod
    def get_inverse_pair(pair):
        return pair.split("/")[1] + "/" + pair.split("/")[0]

    @staticmethod
    def find_matching_symbol(market_symbols, synthetic_symbol):
        """
        Find the matching symbol in the market symbols list using the format with a delimiter.
        Returns the version without the delimiter if a match is found.
        """
        # Loop through the market symbols to find a match
        normalized_synthetic_symbol = (
            synthetic_symbol.replace("/", "").lower().replace("xbt", "btc")
        )

        for market_symbol in market_symbols:
            # skip futures
            if "-" in market_symbol:
                continue
            # Remove delimiter for matching
            base_market_symbol = market_symbol.split(":")[0]
            normalized_market_symbol = (
                base_market_symbol.replace("/", "").lower().replace("xbt", "btc")
            )
            # Exact match without delimiter
            if normalized_market_symbol == normalized_synthetic_symbol:
                return market_symbol
                # return synthetic_symbol

            # Match with suffix, ignoring case
            if normalized_market_symbol.startswith(normalized_synthetic_symbol):
                return market_symbol
                # return synthetic_symbol

        return None

    @staticmethod
    def generate_crypto_to_crypto_fees(currency_fees, inter_coin_dict):
        """Generate synthetic crypto-to-crypto pairs' fees based on the inter_coin_dict."""
        # Loop through the inter_coin_dict to generate synthetic fees
        for pair, synthetic_symbol in inter_coin_dict.items():
            crypto1, crypto2 = pair.split("/")

            # Generate fees for the synthetic pairs
            # Assuming fees are similar to the corresponding USD pairs
            if f"{crypto1}/USD" in currency_fees:
                currency_fees[pair] = currency_fees[f"{crypto1}/USD"]
            if f"{crypto2}/USD" in currency_fees:
                currency_fees[pair] = currency_fees[f"{crypto2}/USD"]

        return currency_fees

    async def fetch_all_order_books(self):
        tasks = [
            self.fetch_order_book(currency, symbol)
            for currency, symbol in self.currencies.items()
        ]
        await self._gather_with_timeout(tasks, "fetch_all_live_prices")

    # async def fetch_all_order_books(self, batch_size):
    #     batches = DataFetcher.create_batches(self.currencies, batch_size)
    #
    #     for batch_number, batch in enumerate(batches):
    #         print(f"Processing order book batch {batch_number + 1}/{len(batches)}")
    #
    #         tasks = [
    #             self.fetch_order_book(currency, symbol)
    #             for currency, symbol in batch.items()
    #         ]
    #
    #         await self._gather_with_timeout(
    #             tasks, f"fetch_order_books_batch_{batch_number}"
    #         )

    async def fetch_order_book(self, currency, symbol):
        current_order_book = await self.client.fetch_order_book(symbol)
        self.order_books[currency] = current_order_book

    def get_order_book(self, currency):
        return self.order_books.get(currency)

    def get_df_of_all_historical_data(self, field="close"):
        start_time = time()
        dataframes = []

        # take copy to avoid errors of it being read and written to at same time
        historic_data_copy = self.historical_data.copy()

        for currency, historic_currency_data in historic_data_copy.items():
            datetime_values = historic_currency_data.datetime
            if field == "volume":
                data = historic_currency_data.volume
            else:
                data = historic_currency_data.close

            df_currency = pd.DataFrame(
                data={currency: data}, index=pd.to_datetime(datetime_values)
            )
            dataframes.append(df_currency)

        if not dataframes:
            return None

        # Concatenate all DataFrames along the datetime index
        df_combined = pd.concat(dataframes, axis=1, join="outer")
        end_time = time()
        print(f"time to get all historic prices {(end_time - start_time):.2f}")
        df_combined = df_combined.sort_index()
        return df_combined

    def get_df_of_historical_prices_pairs(self, pair):
        start_time = time()
        dataframes = []

        for currency in pair:
            historic_data = self.historical_data[currency]
            datetime_values = historic_data.datetime
            closing_prices = historic_data.close

            df_currency = pd.DataFrame(
                data={currency: closing_prices}, index=pd.to_datetime(datetime_values)
            )
            dataframes.append(df_currency)

        if not dataframes:
            return None

        df_combined = pd.concat(dataframes, axis=1, join="outer")
        end_time = time()
        # print(f"time to get pair of historic prices {(end_time - start_time):.2f}")
        df_combined = df_combined.sort_index()
        return df_combined

    def get_cointegration_pairs(self):
        return self.cointegration_pairs

    def get_cointegration_pair(self, pair):
        return self.cointegration_pairs[pair]

    def get_historical_price_options(self):
        return list(self.historical_data.keys())

    async def set_balance(self):
        try:
            balance = await self.client.fetch_balance()
            print("Balance:", balance)
            self.balance = balance
        except ccxt_async.BaseError as e:
            print("Error fetching balance:", e)

    async def set_open_trades(self):
        try:
            # self.client.options["defaultType"] = "spot"
            # spot_orders = await self.client.fetch_open_orders()

            self.client.options["defaultType"] = "margin"
            open_positions = await self.client.fetch_positions()

            print(f"open_positions: {open_positions}")
            self.open_orders = open_positions
        except ccxt_async.BaseError as e:
            print("Error fetching open trades:", e)

    def get_balance(self):
        balance_dict = self.balance["info"]["result"]
        balance_dict = {
            symbol + "/USD": float(sym_balance["balance"])
            for symbol, sym_balance in balance_dict.items()
            if float(sym_balance["balance"]) > 0
        }
        return balance_dict

    def get_open_trades(self):
        open_orders = {
            element["symbol"]: element["contracts"] for element in self.open_orders
        }
        return open_orders

    async def open_long_position(self, symbol, amount):
        """Open a long position by buying in the spot market."""
        try:
            print(f"Opening long position for {symbol}, amount: {amount}")
            # self.client.options["defaultType"] = "spot"
            self.client.options.update({"defaultType": "spot"})

            # order = None
            order = await self.client.create_market_buy_order(symbol, float(amount))

            print("Long position opened:", order)
            return order
        except ccxt_async.BaseError as e:
            print(f"Error opening long position: {e}")
            return None

    async def close_long_position(self, symbol, amount):
        """Close a long position by selling in the spot market."""
        try:
            print(f"Closing long position for {symbol}, amount: {amount}")
            self.client.options["defaultType"] = "spot"
            order = None
            # order = await self.client.create_market_sell_order(symbol, amount)

            print("Long position closed:", order)
            return order
        except ccxt_async.BaseError as e:
            print(f"Error closing long position: {e}")
            return None

    async def open_short_position(self, symbol, amount):
        """Open a short position using margin trading."""
        try:
            print(f"Opening short position for {symbol}, amount: {amount}")
            # self.client.options["defaultType"] = "margin"
            self.client.options.update({"defaultType": "margin"})
            # order = None
            order = await self.client.create_market_sell_order(symbol, float(amount))
            print("Short position opened:", order)
            return order
        except ccxt_async.BaseError as e:
            print(f"Error opening short position: {e}")
            return None

    async def close_short_position(self, symbol, amount):
        """Close a short position by buying in the margin market."""
        try:
            print(f"Closing short position for {symbol}, amount: {amount}")
            self.client.options["defaultType"] = "margin"
            order = None
            # order = await self.client.create_market_buy_order(symbol, amount)
            print("Short position closed:", order)
            return order
        except ccxt_async.BaseError as e:
            print(f"Error closing short position: {e}")
            return None

    async def open_arbitrage_positions(self, position_size):
        long_position = position_size["long_position"]
        short_position = position_size["short_position"]

        if not long_position or not short_position:
            print("Error: Both long and short position details are required.")
            return None

        long_coin, long_amount = long_position["coin"], long_position["amount"]
        short_coin, short_amount = short_position["coin"], short_position["amount"]

        try:
            # Open long position first with spot settings
            self.client.options["defaultType"] = "spot"
            long_trade = await self.open_long_position(long_coin, long_amount)

            # After the long position is open, switch to margin for the short
            self.client.options["defaultType"] = "margin"
            short_trade = await self.open_short_position(short_coin, short_amount)

            return {
                "long_trade": long_trade,
                "short_trade": short_trade,
            }
        except Exception as e:
            print(f"Error opening arbitrage positions: {e}")
            return None

    # async def open_arbitrage_positions(self, position_size):
    #     long_position = position_size["long_position"]
    #     short_position = position_size["short_position"]
    #
    #     if not long_position or not short_position:
    #         print("Error: Both long and short position details are required.")
    #         return None
    #
    #     long_coin, long_amount = long_position["coin"], long_position["amount"]
    #     short_coin, short_amount = short_position["coin"], short_position["amount"]
    #
    #     try:
    #         long_trade, short_trade = await asyncio.gather(
    #             self.open_long_position(long_coin, long_amount),
    #             self.open_short_position(short_coin, short_amount),
    #         )
    #         return {
    #             "long_trade": long_trade,
    #             "short_trade": short_trade,
    #         }
    #     except Exception as e:
    #         print(f"Error opening arbitrage positions: {e}")
    #         return None

    async def close_arbitrage_positions(self, position_size):
        long_position = position_size["long_position"]
        short_position = position_size["short_position"]

        if not long_position or not short_position:
            print("Error: Both long and short position details are required to close.")
            return None

        long_coin, long_amount = long_position["coin"], long_position["amount"]
        short_coin, short_amount = short_position["coin"], short_position["amount"]

        try:
            close_long_trade, close_short_trade = await asyncio.gather(
                self.close_long_position(long_coin, long_amount),
                self.close_short_position(short_coin, short_amount),
            )
            return {
                "close_long_trade": close_long_trade,
                "close_short_trade": close_short_trade,
            }

        except Exception as e:
            print(f"Error closing arbitrage positions: {e}")
            return None

    def open_arbitrage_positions_sync(self, position_size):
        return asyncio.run(self.open_arbitrage_positions(position_size))

    def close_arbitrage_positions_sync(self, position_size):
        return asyncio.run(self.close_arbitrage_positions(position_size))
