from dash import dcc


class FilterComponent:
    @staticmethod
    def get_currency_options():
        # Fetch currency options
        return [
            {"label": "Bitcoin", "value": "BTC/USD"},
            {"label": "Ethereum", "value": "ETH/USD"},
            {"label": "Solana", "value": "SOL/USD"},
            {"label": "Litecoin", "value": "LTC/USD"},
            {"label": "Dogecoin", "value": "DOGE/USD"},
            {"label": "Cardano", "value": "ADA/USD"},
            {"label": "Ripple", "value": "XRP/USD"},
        ]

    @staticmethod
    def get_currency_names():
        return ["bitcoin", "ethereum", "solana", "dogecoin", "cardano", "ripple"]

    @staticmethod
    def get_exchange_options():
        # Fetch exchange options
        return [
            {"label": "Bitmex", "value": "Bitmex"},
            {"label": "Coinbase", "value": "Coinbase"},
            # {"label": "Binance", "value": "Binance"},
            {"label": "Huobi", "value": "HTX"},
            {"label": "Kraken", "value": "Kraken"},
            {"label": "Bybit", "value": "Bybit"},
        ]

    @staticmethod
    def get_exchange_names():
        return ["bitmex", "coinbase", "binance"]

    @staticmethod
    def get_arbitrage_options():
        return [
            {"label": "Simple Arbitrage", "value": "simple"},
            {"label": "Triangular Arbitrage", "value": "triangular"},
            {"label": "Statistical Arbitrage", "value": "statistical"},
        ]
