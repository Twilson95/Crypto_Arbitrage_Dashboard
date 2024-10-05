import asyncio
import ccxt.async_support as ccxt
from src.prices.DataFetchers import DataFetcher
from threading import Thread


class DataManager:
    def __init__(self, config, network_fees_config):
        self.sleep_time = 2
        self.config = config
        self.network_fees_config = network_fees_config
        self.exchanges = {}
        self.exchange_fees = {}
        self.network_fees = self.fetch_network_fees()
        self.initialized_event = asyncio.Event()

        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self.run_loop, args=(self.loop,))
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self.initialize_exchanges(), self.loop)
        self.schedule_periodic_fetching()

    @staticmethod
    def run_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def schedule_periodic_fetching(self):
        async def periodic_fetch():
            while True:
                await self.fetch_all_live_prices()
                await self.fetch_all_order_books()
                await asyncio.sleep(self.sleep_time)  # Fetch every 10 seconds

        asyncio.run_coroutine_threadsafe(periodic_fetch(), self.loop)

    async def initialize_exchanges(self):
        # initialise any exchange there exists api keys for
        tasks = [self.initialize_exchange(section) for section in self.config.keys()]
        await asyncio.gather(*tasks)
        self.initialized_event.set()  # Signal that initialization is complete

    async def initialize_exchange(self, exchange_name):
        exchange = None
        api_key = self.config[exchange_name]["api_key"]
        api_secret = self.config[exchange_name]["api_secret"]
        pairs_mapping = self.config[exchange_name]["pairs"]

        api_secret = api_secret.replace("\\n", "\n").strip()
        exchange_class = getattr(ccxt, exchange_name.lower())

        try:
            exchange = exchange_class(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                    # "enable_time_sync": True,
                    "options": {
                        "recvWindow": 10000,  # Increase recv_window to 10 seconds
                        # "defaultType": "spot",  # Ensure we're using the spot market
                        # "fetchMarkets": ["spot"],
                    },
                }
            )
        except ccxt.AuthenticationError as e:
            await exchange.close()
            print(f"Authentication failed for {exchange_name}: {e}")
        except ccxt.NetworkError as e:
            await exchange.close()
            print(f"Network error for {exchange_name}: {e}")
        except ccxt.BaseError as e:
            await exchange.close()
            print(f"Exchange error for {exchange_name}: {e}")
        except Exception as e:
            await exchange.close()
            print(f"Unexpected error for {exchange_name}: {e}")

        try:
            markets = await exchange.load_markets()
        except:
            markets = None

        data_fetcher = DataFetcher(exchange, exchange_name, pairs_mapping, markets)
        await data_fetcher.async_init()
        self.exchanges[exchange_name] = data_fetcher
        print(f"{exchange_name} initialized successfully")

        await data_fetcher.update_all_historical_prices()
        data_fetcher.update_cointegration_pairs()

    async def extract_currency_fees(self, exchange, exchange_name, currencies):
        for currency, symbol in currencies.items():
            trading_fee = await exchange.fetch_trading_fee(currency)
            self.exchange_fees[exchange_name][currency] = {
                "maker": trading_fee.get("maker", 0),
                "taker": trading_fee.get("taker", 0),
            }

    async def close_exchanges(self):
        tasks = [exchange.close() for exchange in self.exchanges.values()]
        await asyncio.gather(*tasks)

    def select_data_fetcher(self, exchange_name):
        return self.exchanges[exchange_name]

    async def fetch_all_live_prices(self):
        tasks = [
            exchange.fetch_all_live_prices() for exchange in self.exchanges.values()
        ]
        await asyncio.gather(*tasks)

    async def fetch_all_order_books(self):
        tasks = [
            exchange.fetch_all_order_books() for exchange in self.exchanges.values()
        ]
        await asyncio.gather(*tasks)

    def get_historical_prices(self, exchange_name, currency):
        future = asyncio.run_coroutine_threadsafe(
            self._get_historical_prices(exchange_name, currency), self.loop
        )
        return future.result()

    async def _get_historical_prices(self, exchange_name, currency):
        await self.initialized_event.wait()  # Wait for initialization to complete
        data_fetcher = self.exchanges.get(exchange_name)
        if not data_fetcher:
            return None
        prices = data_fetcher.get_historical_prices(currency)
        return prices

    def get_live_prices(self, exchange_name, currency):
        future = asyncio.run_coroutine_threadsafe(
            self._get_live_prices(exchange_name, currency), self.loop
        )
        return future.result()

    async def _get_live_prices(self, exchange_name, currency):
        await self.initialized_event.wait()  # Wait for initialization to complete
        data_fetcher = self.exchanges.get(exchange_name)
        if not data_fetcher:
            return None
        prices = data_fetcher.get_live_prices(currency)
        return prices

    def get_live_prices_for_all_exchanges(self, currency):
        exchange_prices = {}
        for exchange_name in self.exchanges.keys():
            live_prices = self.get_live_prices(exchange_name, currency)
            if live_prices:
                exchange_prices[exchange_name] = live_prices
        return exchange_prices

    def get_maker_taker_fees_for_all_exchanges(self, currency):
        future = asyncio.run_coroutine_threadsafe(
            self._get_maker_taker_fees_for_all_exchanges(currency), self.loop
        )
        return future.result()

    async def _get_maker_taker_fees_for_all_exchanges(self, currency):
        await self.initialized_event.wait()
        exchange_fees = {}
        for exchange_name, exchange in self.exchanges.items():
            fees = exchange.get_currency_fee(currency)
            if fees:
                exchange_fees[exchange_name] = fees
        return exchange_fees

    def get_withdrawal_deposit_fees_for_all_exchanges(self):
        future = asyncio.run_coroutine_threadsafe(
            self._get_withdrawal_deposit_fees_for_all_exchanges(), self.loop
        )
        return future.result()

    async def _get_withdrawal_deposit_fees_for_all_exchanges(self):
        await self.initialized_event.wait()
        exchange_fees = {}
        for exchange_name, exchange in self.exchanges.items():
            fees = exchange.get_exchange_fees()
            if fees:
                exchange_fees[exchange_name] = fees
        return exchange_fees

    def fetch_network_fees(self):
        # possible to make this dynamic in future
        return self.network_fees_config["default_network_fees"]

    def get_network_fees(self, currency):
        return self.network_fees[currency]

    def get_exchanges(self):
        return list(self.exchanges.keys())

    def get_live_prices_and_fees_for_single_exchange(self, exchange_name):
        exchange = self.exchanges[exchange_name]
        prices = exchange.get_live_data()
        # exchange_fees = exchange.get_all_exchange_fees()
        currency_fees = exchange.get_all_currency_fees()
        return prices, currency_fees

    def get_order_book(self, exchange_name, symbol):
        future = asyncio.run_coroutine_threadsafe(
            self._get_order_book(exchange_name, symbol), self.loop
        )
        return future.result()

    async def _get_order_book(self, exchange_name, symbol):
        await self.initialized_event.wait()
        exchange = self.exchanges[exchange_name]
        order_book = exchange.get_order_book(symbol)
        return order_book

    def get_historical_prices_for_all_currencies(self, exchange_name):
        future = asyncio.run_coroutine_threadsafe(
            self._get_historical_prices_for_all_currencies(exchange_name), self.loop
        )
        return future.result()

    async def _get_historical_prices_for_all_currencies(self, exchange_name):
        await self.initialized_event.wait()  # Wait for initialization to complete
        data_fetcher = self.exchanges.get(exchange_name)
        if not data_fetcher:
            return None
        prices = data_fetcher.get_df_of_all_historical_prices()
        return prices

    def get_cointegration_spreads(self, exchange_name):
        future = asyncio.run_coroutine_threadsafe(
            self._get_cointegration_spreads(exchange_name), self.loop
        )
        return future.result()

    async def _get_cointegration_spreads(self, exchange_name):
        await self.initialized_event.wait()  # Wait for initialization to complete
        data_fetcher = self.exchanges.get(exchange_name)
        if not data_fetcher:
            return None
        cointegration_pairs = data_fetcher.get_cointegration_spreads()
        return cointegration_pairs
