import asyncio
import ccxt.async_support as ccxt
from src.prices.DataFetchers import DataFetcher
from threading import Thread


class DataManager:
    def __init__(self, config):
        self.config = config
        self.exchanges = {}
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
        try:
            api_key = self.config[exchange_name]["api_key"]
            api_secret = self.config[exchange_name]["api_secret"]
            pairs_mapping = self.config[exchange_name]["pairs"]

            api_secret = api_secret.replace("\\n", "\n").strip()

            exchange_class = getattr(
                ccxt, exchange_name.lower()
            )  # Get the exchange class
            exchange = exchange_class(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                }
            )
            await exchange.load_markets()
            data_fetcher = DataFetcher(exchange, pairs_mapping)
            self.exchanges[exchange_name] = data_fetcher
            await data_fetcher.async_init()
            print(f"{exchange_name} initialized successfully")

        except Exception as e:
            print(f"Failed to initialize {exchange_name}: {e}")

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
