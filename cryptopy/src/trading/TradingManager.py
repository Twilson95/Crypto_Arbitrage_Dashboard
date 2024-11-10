import ccxt


class TradeManager:
    def __init__(self, config, exchange_name, make_trades=False):
        self.config = config
        self.client = self.initialise(exchange_name)
        self.max_leverage_by_symbol = self._fetch_all_max_leverage()
        self.make_trades = make_trades

    def _fetch_all_max_leverage(self):
        max_leverage_by_symbol = {}
        self.client.load_markets()

        for symbol, market in self.client.markets.items():
            leverage_limits = market.get("limits", {}).get("leverage", {})
            max_leverage = leverage_limits.get("max")
            max_leverage_by_symbol[symbol] = max_leverage if max_leverage else None

        print("Max leverage by symbol:", max_leverage_by_symbol)
        return max_leverage_by_symbol

    def get_max_leverage(self, symbol):
        return float(self.max_leverage_by_symbol.get(symbol, None))

    def initialise(self, exchange_name):
        api_key = self.config[exchange_name]["api_key"]
        api_secret = self.config[exchange_name]["api_secret"]
        api_secret = api_secret.replace("\\n", "\n").strip()
        exchange_class = getattr(ccxt, exchange_name.lower())

        return exchange_class(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "options": {
                    "recvWindow": 10000,  # Increase recv_window to 10 seconds
                },
            }
        )

    def open_long_position(self, symbol, amount):
        """Open a long position by buying in the spot market."""
        try:
            print(f"Opening long position for {symbol}, amount: {amount}")
            self.client.options["defaultType"] = "spot"  # Ensure spot market
            if self.make_trades:
                order = self.client.create_market_buy_order(symbol, float(amount))
                print("Long position opened:", order)
            else:
                order = None
            return order
        except ccxt.BaseError as e:
            print(f"Error opening long position: {e}")
            return None

    def close_long_position(self, symbol, amount):
        """Close a long position by selling in the spot market."""
        try:
            print(f"Closing long position for {symbol}, amount: {amount}")
            self.client.options["defaultType"] = "spot"  # Ensure spot market
            if self.make_trades:
                order = self.client.create_market_sell_order(symbol, float(amount))
                print("Long position closed:", order)
            else:
                order = None
            return order
        except ccxt.BaseError as e:
            print(f"Error closing long position: {e}")
            return None

    def open_short_position(self, symbol, amount):
        """Open a short position using margin trading."""
        leverage = self.get_max_leverage(symbol)
        try:
            print(
                f"Opening short position for {symbol}, amount: {amount}, leverage: {leverage}"
            )
            params = {
                "leverage": leverage,
                "trading_agreement": "agree",
            }
            self.client.options["defaultType"] = "margin"
            if self.make_trades:
                order = self.client.create_market_sell_order(
                    symbol, float(amount), params=params
                )
                print("Short position opened:", order)
            else:
                order = None
            return order
        except ccxt.BaseError as e:
            print(f"Error opening short position: {e}")
            return None

    def close_short_position(self, symbol, amount):
        """Close a short position by buying in the margin market."""
        leverage = self.get_max_leverage(symbol)
        try:
            print(
                f"Closing short position for {symbol}, amount: {amount}, leverage: {leverage}"
            )
            params = {
                "leverage": leverage,
                "reduceOnly": True,
                "trading_agreement": "agree",
            }
            self.client.options["defaultType"] = "margin"
            if self.make_trades:
                order = self.client.create_market_buy_order(
                    symbol, float(amount), params=params
                )
                print("Short position closed:", order)
            else:
                order = None
            return order
        except ccxt.BaseError as e:
            print(f"Error closing short position: {e}")
            return None

    def open_arbitrage_positions(self, position_size):
        print("-----------------------------------------------------------------------")
        long_position = position_size["long_position"]
        short_position = position_size["short_position"]

        if not long_position or not short_position:
            print("Error: Both long and short position details are required.")
            return None

        long_coin, long_amount = long_position["coin"], long_position["amount"]
        short_coin, short_amount = short_position["coin"], short_position["amount"]

        try:
            self.client.options["defaultType"] = "spot"
            long_trade = self.open_long_position(long_coin, long_amount)

            self.client.options["defaultType"] = "margin"
            short_trade = self.open_short_position(short_coin, short_amount)

            return {
                "long_trade": long_trade,
                "short_trade": short_trade,
            }
        except Exception as e:
            print(f"Error opening arbitrage positions: {e}")
            return None

    def close_arbitrage_positions(self, position_size):
        print("-----------------------------------------------------------------------")

        long_position = position_size["long_position"]
        short_position = position_size["short_position"]

        if not long_position or not short_position:
            print("Error: Both long and short position details are required to close.")
            return None

        long_coin, long_amount = long_position["coin"], long_position["amount"]
        short_coin, short_amount = short_position["coin"], short_position["amount"]

        try:
            self.client.options["defaultType"] = "spot"
            long_trade = self.close_long_position(long_coin, long_amount)

            self.client.options["defaultType"] = "margin"
            short_trade = self.close_short_position(short_coin, short_amount)

            return {
                "long_trade": long_trade,
                "short_trade": short_trade,
            }
        except Exception as e:
            print(f"Error opening arbitrage positions: {e}")
            return None

    def fetch_max_leverage(self, symbol):
        self.client.load_markets()

        if symbol in self.client.markets:
            market = self.client.markets[symbol]

            leverage_levels = market.get("leverage", [])

            if leverage_levels:
                max_leverage = max(leverage_levels)
                print(f"Max leverage for {symbol}: {max_leverage}")
                return max_leverage
            else:
                print(f"No leverage available for {symbol}")
                return None
        else:
            print(f"Symbol {symbol} not found.")
            return None
