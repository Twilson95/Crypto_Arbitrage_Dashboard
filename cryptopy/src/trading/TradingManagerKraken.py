import krakenex
import time


class TradeManagerKraken:
    def __init__(self, config, exchange_name, make_trades=False):
        self.config = config
        self.client = self.initialise(exchange_name)
        self.markets = self._fetch_markets()  # Store asset pair info
        self.max_leverage_by_symbol = self._fetch_all_max_leverage()
        self.make_trades = make_trades

    def initialise(self, exchange_name):
        api_key = self.config[exchange_name]["api_key"]
        api_secret = self.config[exchange_name]["api_secret"]
        api_secret = api_secret.replace("\\n", "\n").strip()

        client = krakenex.API(key=api_key, secret=api_secret)
        return client

    def _fetch_markets(self):
        """Fetch asset pair info from Kraken and store it for later use."""
        resp = self.client.query_public("AssetPairs")
        if resp.get("error"):
            print("Error fetching asset pairs:", resp["error"])
            return {}

        return resp.get("result", {})

    def _fetch_all_max_leverage(self):
        max_leverage_by_symbol = {}
        for pair, info in self.markets.items():
            leverage_buy = info.get("leverage_buy", [])
            leverage_sell = info.get("leverage_sell", [])
            # Combine both to find a universal max
            all_lev = leverage_buy + leverage_sell
            max_lev = max(all_lev) if all_lev else None
            max_leverage_by_symbol[pair] = max_lev
        print("Max leverage by symbol:", max_leverage_by_symbol)
        return max_leverage_by_symbol

    def get_max_leverage(self, symbol):
        lev = self.max_leverage_by_symbol.get(symbol, None)
        return float(lev) if lev else None

    def _place_order(self, symbol, side, volume, leverage=None):
        """Helper method to place a market order on Kraken."""
        order_data = {
            "pair": symbol.replace(
                "/", ""
            ),  # Kraken expects e.g. 'XBTUSD' without slash
            "type": side,  # 'buy' or 'sell'
            "ordertype": "market",
            "volume": str(volume),
        }

        if leverage is not None and leverage > 1:
            # Kraken expects leverage as a string, e.g. '5'
            order_data["leverage"] = str(int(leverage))

        if self.make_trades:
            resp = self.client.query_private("AddOrder", order_data)
            if resp.get("error"):
                print(f"Error placing order on {symbol}:", resp["error"])
                return None
            return resp.get("result")
        else:
            print(f"Mock order (no trade executed): {order_data}")
            return None

    def open_long_position(self, symbol, amount):
        """Open a long position by buying in the spot market (no leverage)."""
        print(f"Opening long position for {symbol}, amount: {amount}")
        return self._place_order(symbol, "buy", amount, leverage=None)

    def close_long_position(self, symbol, amount):
        """Close a long position by selling in the spot market."""
        print(f"Closing long position for {symbol}, amount: {amount}")
        return self._place_order(symbol, "sell", amount, leverage=None)

    def open_short_position(self, symbol, amount):
        """Open a short position (leveraged sell)."""
        leverage = self.get_max_leverage(symbol)
        if leverage is None:
            print(f"No leverage info for {symbol}, can't open short.")
            return None

        print(
            f"Opening short position for {symbol}, amount: {amount}, leverage: {leverage}"
        )
        # A leveraged sell order on Kraken is effectively a short.
        return self._place_order(symbol, "sell", amount, leverage=leverage)

    def close_short_position(self, symbol, amount):
        """Close a short position by buying back with leverage."""
        leverage = self.get_max_leverage(symbol)
        if leverage is None:
            print(f"No leverage info for {symbol}, can't close short.")
            return None

        print(
            f"Closing short position for {symbol}, amount: {amount}, leverage: {leverage}"
        )
        # A leveraged buy order to close the short position.
        return self._place_order(symbol, "buy", amount, leverage=leverage)

    def open_arbitrage_positions(self, position_size):
        print("-----------------------------------------------------------------------")
        long_position = position_size.get("long_position")
        short_position = position_size.get("short_position")

        if not long_position or not short_position:
            print("Error: Both long and short position details are required.")
            return None

        long_coin, long_amount = long_position["coin"], long_position["amount"]
        short_coin, short_amount = short_position["coin"], short_position["amount"]

        try:
            long_trade = self.open_long_position(long_coin, long_amount)
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

        long_position = position_size.get("long_position")
        short_position = position_size.get("short_position")

        if not long_position or not short_position:
            print("Error: Both long and short position details are required to close.")
            return None

        long_coin, long_amount = long_position["coin"], long_position["amount"]
        short_coin, short_amount = short_position["coin"], short_position["amount"]

        try:
            long_trade = self.close_long_position(long_coin, long_amount)
            short_trade = self.close_short_position(short_coin, short_amount)
            return {
                "long_trade": long_trade,
                "short_trade": short_trade,
            }
        except Exception as e:
            print(f"Error closing arbitrage positions: {e}")
            return None

    def fetch_max_leverage(self, symbol):
        # Since we already fetched asset pairs, just return from stored data
        max_leverage = self.get_max_leverage(symbol)
        if max_leverage:
            print(f"Max leverage for {symbol}: {max_leverage}")
        else:
            print(f"No leverage available for {symbol}")
        return max_leverage
