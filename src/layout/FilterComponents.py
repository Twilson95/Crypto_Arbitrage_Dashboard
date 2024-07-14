from dash import dcc


class FilterComponent:
    @staticmethod
    def get_currency_options():
        # Fetch currency options
        return [
            {"label": "Bitcoin", "value": "BTC/USD"},
            {"label": "Ethereum", "value": "ETH/USD"},
            {"label": "Solana", "value": "SOL/USD"},
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
            {"label": "Binance", "value": "Binance"},
        ]

    @staticmethod
    def get_exchange_names():
        return ["bitmex", "coinbase", "binance"]

    @staticmethod
    def get_indicator_options():
        return dcc.Dropdown(
            id="filter-component",
            options=[
                {"label": "SMA", "value": "SMA"},
                {"label": "EMA", "value": "EMA"},
            ],
            multi=True,
            placeholder="Select technical indicators",
        )
