from dash import dcc


class FilterComponent:
    def __init__(self):
        self.currency_options = [
            {"label": "Bitcoin", "value": "BTC/USD"},
            {"label": "Ethereum", "value": "ETH/USD"},
            {"label": "Solana", "value": "SOL/USD"},
            {"label": "Dogecoin", "value": "DOGE/USD"},
            {"label": "Cardano", "value": "ADA/USD"},
            {"label": "Ripple", "value": "XRP/USD"},
            {"label": "Litecoin", "value": "LTC/USD"},
        ]
        self.exchange_options = [
            {"label": "Bitmex", "value": "Bitmex"},
            {"label": "Coinbase", "value": "Coinbase"},
            # {"label": "Binance", "value": "Binance"},
            {"label": "Huobi", "value": "HTX"},
            {"label": "Kraken", "value": "Kraken"},
            {"label": "Bybit", "value": "Bybit"},
        ]
        self.arbitrage_options = [
            {"label": "Simple Arbitrage", "value": "simple"},
            {"label": "Triangular Arbitrage", "value": "triangular"},
            {"label": "Statistical Arbitrage", "value": "statistical"},
        ]

    def get_currency_options(self):
        # Fetch currency options
        return self.currency_options

    def get_currency_names(self):
        return [currency["label"].lower for currency in self.currency_options]

    def get_exchange_options(self):
        # Fetch exchange options
        return self.exchange_options

    def get_exchange_names(self):
        # return ["bitmex", "coinbase", "binance"]
        return [exchange["label"].lower for exchange in self.currency_options]

    def get_arbitrage_options(self):
        return self.arbitrage_options
