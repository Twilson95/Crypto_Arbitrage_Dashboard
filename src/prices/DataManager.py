import asyncio
import ccxt.async_support as ccxt
from src.prices.DataFetchers import DataFetcher
from threading import Thread


class DataManager:
    def __init__(self, config):
        self.config = config
        self.exchanges = {}
        self.exchange_fees = {}
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

                await asyncio.sleep(10)  # Fetch every 10 seconds
                # print("Scheduled fetch_all_live_prices executed")

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

        markets = await exchange.load_markets()
        # print(exchange_name, "markets loaded")
        # await self.extract_currency_fees(exchange, exchange_name, pairs_mapping)

        data_fetcher = DataFetcher(exchange, exchange_name, pairs_mapping, markets)
        # print(exchange_name, "data fetcher created")
        self.exchanges[exchange_name] = data_fetcher
        # print(exchange_name, "added to exchange dict")
        await data_fetcher.async_init()
        # print(exchange_name, "data fetcher initialized")

        print(f"{exchange_name} initialized successfully")

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

    def select_data_fetcher(self, exchange):
        return self.exchanges[exchange]

    async def fetch_all_live_prices(self):
        tasks = [
            exchange.fetch_all_live_prices() for exchange in self.exchanges.values()
        ]
        await asyncio.gather(*tasks)

    def get_historical_prices(self, exchange, currency):
        future = asyncio.run_coroutine_threadsafe(
            self._get_historical_prices(exchange, currency), self.loop
        )
        return future.result()

    async def _get_historical_prices(self, exchange, currency):
        await self.initialized_event.wait()  # Wait for initialization to complete
        data_fetcher = self.exchanges.get(exchange)
        if not data_fetcher:
            return None
        prices = data_fetcher.get_historical_prices(currency)
        return prices

    def get_live_prices(self, exchange, currency):
        future = asyncio.run_coroutine_threadsafe(
            self._get_live_prices(exchange, currency), self.loop
        )
        return future.result()

    async def _get_live_prices(self, exchange, currency):
        await self.initialized_event.wait()  # Wait for initialization to complete
        data_fetcher = self.exchanges.get(exchange)
        if not data_fetcher:
            return None
        prices = data_fetcher.get_live_prices(currency)
        return prices

    def get_live_prices_for_all_exchanges(self, currency):
        exchange_prices = {}
        for exchange in self.exchanges:
            live_prices = self.get_live_prices(exchange, currency)
            if live_prices:
                exchange_prices[exchange] = live_prices
        return exchange_prices

    def get_fees_for_all_exchanges(self, currency):
        future = asyncio.run_coroutine_threadsafe(
            self._get_fees_for_all_exchanges(currency), self.loop
        )
        return future.result()

    async def _get_fees_for_all_exchanges(self, currency):
        await self.initialized_event.wait()
        exchange_fees = {}
        for exchange in self.exchanges.keys():
            fees = self.get_fees(exchange, currency)
            if fees:
                exchange_fees[exchange] = fees
        return exchange_fees

    def get_fees(self, exchange_name, currency):
        exchange = self.exchanges.get(exchange_name, {})
        return exchange.get_currency_fee(currency)

    def fetch_deposit_withdrawal_rates(self):
        deposit = None
        withdrawal = None
        return deposit, withdrawal
