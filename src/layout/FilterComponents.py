from dash import dcc


class FilterComponent:
    @staticmethod
    def get_currency_options():
        # Fetch currency options
        return [
            {"label": "Bitcoin", "value": "bitcoin"},
            {"label": "Ethereum", "value": "ethereum"},
            {"label": "Solana", "value": "solana"},
            {"label": "Dogecoin", "value": "dogecoin"},
            {"label": "Cardano", "value": "cardano"},
            {"label": "Ripple", "value": "ripple"},
        ]

    @staticmethod
    def get_currency_names():
        return ["bitcoin", "ethereum", "solana", "dogecoin", "cardano", "ripple"]

    @staticmethod
    def get_exchange_options():
        # Fetch exchange options
        return [
            {"label": "Bitmex", "value": "bitmex"},
            {"label": "Coinbase", "value": "coinbase"},
            {"label": "Binance", "value": "binance"},
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
