from src.DataFetchers import DataFetcher, BitmexDataFetcher, CoinbaseDataFetcher
import configparser


class DataManager:
    def __init__(self):
        self.bitmex_data_fetcher = BitmexDataFetcher()
        self.coinbase_data_fetcher = CoinbaseDataFetcher()

    def select_data_fetcher(self, exchange):
        if exchange == "bitmex":
            return self.bitmex_data_fetcher
        elif exchange == "coinbase":
            return self.coinbase_data_fetcher
        else:
            return None

    def fetch_all_live_prices(self):
        self.bitmex_data_fetcher.fetch_all_live_prices()
        self.coinbase_data_fetcher.fetch_all_live_prices()

    def get_historical_prices(self, exchange, currency):
        data_fetcher = self.select_data_fetcher(exchange)
        if not data_fetcher:
            return None
        prices = data_fetcher.get_historical_prices(currency)
        return prices

    def get_live_prices(self, exchange, currency):
        data_fetcher = self.select_data_fetcher(exchange)
        if not data_fetcher:
            return None
        prices = data_fetcher.get_live_prices(currency)
        return prices
