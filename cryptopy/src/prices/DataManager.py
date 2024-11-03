import asyncio
import ccxt.async_support as ccxt
from cryptopy import DataFetcher
from threading import Thread


class DataManager:
    def __init__(self, config, network_fees_config=None, live_trades=True):
        self.sleep_time = 2
        self.config = config
        self.exchanges = {}
        self.exchange_fees = {}
        self.network_fees = self.fetch_network_fees(network_fees_config)
        self.initialized_event = asyncio.Event()

        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self.run_loop, args=(self.loop,))
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self.initialize_exchanges(), self.loop)
        if live_trades:
            self.schedule_periodic_fetching()

    async def initialize_exchanges(self):
        # initialise any exchange there exists api keys for
        tasks = [self.initialize_exchange(section) for section in self.config.keys()]
        await asyncio.gather(*tasks)
        self.initialized_event.set()  # Signal that initialization is complete

    def get_exchange(self, exchange_name):
        async def get_exchange_async(exchange_name):
            await self.initialized_event.wait()
            return self.exchanges.get(exchange_name, None)

        future = asyncio.run_coroutine_threadsafe(
            get_exchange_async(exchange_name), self.loop
        )
        return future.result()

    @staticmethod
    def run_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def schedule_periodic_fetching(self):
        async def periodic_fetch():
            while True:
                await self.fetch_all_live_prices()
                self.update_all_cointegration_spreads()
                await self.fetch_all_order_books()
                await asyncio.sleep(self.sleep_time)

        asyncio.run_coroutine_threadsafe(periodic_fetch(), self.loop)

    def update_all_cointegration_spreads(self):
        for exchange in self.exchanges.values():
            exchange.update_all_cointegration_spreads()

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

    async def extract_currency_fees(self, exchange, exchange_name, currencies):
        for currency, symbol in currencies.items():
            trading_fee = await exchange.fetch_trading_fee(currency)
            self.exchange_fees[exchange_name][currency] = {
                "maker": trading_fee.get("maker", 0),
                "taker": trading_fee.get("taker", 0),
            }

    async def close_exchanges(self):
        tasks = [exchange.client.close() for exchange in self.exchanges.values()]
        await asyncio.gather(
            *tasks, return_exceptions=True
        )  # Ensure all close tasks finish

    async def shutdown(self):
        await self.close_exchanges()

        pending_tasks = [
            task for task in asyncio.all_tasks(self.loop) if not task.done()
        ]
        for task in pending_tasks:
            task.cancel()
            try:
                await asyncio.sleep(0)
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                # Handle other exceptions gracefully
                print(f"Exception during task cancellation: {e}")

        print("All pending tasks canceled.")
        self.loop.stop()

    def shutdown_sync(self):
        asyncio.run(self.shutdown())
        self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread.is_alive():
            print("Waiting for the event loop thread to finish...")
            self.thread.join()

        self.loop.close()
        print("Event loop has been closed and thread joined.")

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
        data_fetcher = self.exchanges[exchange_name]
        prices = data_fetcher.get_historical_prices(currency)
        return prices

    def get_live_prices(self, exchange_name, currency):
        data_fetcher = self.exchanges[exchange_name]
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
        exchange_fees = {}
        for exchange_name, exchange in self.exchanges.items():
            fees = exchange.get_currency_fee(currency)
            if fees:
                exchange_fees[exchange_name] = fees
        return exchange_fees

    def get_withdrawal_deposit_fees_for_all_exchanges(self):
        exchange_fees = {}
        for exchange_name, exchange in self.exchanges.items():
            fees = exchange.get_exchange_fees()
            if fees:
                exchange_fees[exchange_name] = fees
        return exchange_fees

    @staticmethod
    def fetch_network_fees(network_fees_config):
        if network_fees_config is None:
            return None
        else:
            return network_fees_config["default_network_fees"]

    def get_network_fees(self, currency):
        return self.network_fees.get(currency, 0)

    def get_exchanges(self):
        return list(self.exchanges.keys())

    def get_live_prices_and_fees_for_single_exchange(self, exchange_name):
        exchange = self.exchanges[exchange_name]
        prices = exchange.get_live_data()
        # exchange_fees = exchange.get_all_exchange_fees()
        currency_fees = exchange.get_all_currency_fees()
        return prices, currency_fees

    def get_order_book(self, exchange_name, symbol):
        exchange = self.exchanges[exchange_name]
        order_book = exchange.get_order_book(symbol)
        return order_book

    def get_historical_prices_for_all_currencies(self, exchange_name):
        data_fetcher = self.exchanges[exchange_name]
        prices = data_fetcher.get_df_of_all_historical_prices()
        return prices

    def get_df_of_historical_prices_pairs(self, exchange_name, pair):
        data_fetcher = self.exchanges[exchange_name]
        prices = data_fetcher.get_df_of_historical_prices_pairs(pair)
        return prices

    def get_historical_price_options_from_exchange(self, exchange_name):
        data_fetcher = self.exchanges[exchange_name]
        options = data_fetcher.get_historical_price_options()
        return options

    def get_cointegration_spreads_from_exchange(self, exchange_name):
        data_fetcher = self.exchanges[exchange_name]
        cointegration_spreads = data_fetcher.get_cointegration_spreads()
        return cointegration_spreads

    def get_cointegration_pairs_from_exchange(self, exchange_name):
        data_fetcher = self.exchanges[exchange_name]
        cointegration_pairs = data_fetcher.get_cointegration_pairs()
        return cointegration_pairs

    def get_cointegration_pair_from_exchange(self, exchange_name, pair):
        data_fetcher = self.exchanges[exchange_name]
        return data_fetcher.get_cointegration_pair(pair)
