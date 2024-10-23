import pandas as pd


class PortfolioManager:
    def __init__(self):
        self.portfolio = None
        self.traded_pairs = set()
        self.bought_coins = set()
        self.sold_coins = set()
        self.open_events = dict()
        self.max_trades = 6

    def read_portfolio(self, portfolio_path):
        self.portfolio = pd.read_csv(portfolio_path)
        self.traded_pairs = set(
            zip(self.portfolio["bought_coin"], self.portfolio["sold_coin"])
        )
        self.bought_coins = set(self.portfolio["bought_coin"])
        self.sold_coins = set(self.portfolio["sold_coin"])

    def get_traded_pairs(self):
        return self.traded_pairs

    def write_to_portfolio(self, portfolio_path):
        self.portfolio.to_csv(portfolio_path)

    def get_portfolio(self):
        return self.portfolio

    def is_pair_traded(self, pair):
        if pair in self.traded_pairs:
            return True
        if pair[0] in self.bought_coins or pair[1] in self.bought_coins:
            return True
        if pair[0] in self.sold_coins or pair[1] in self.sold_coins:
            return True
        return False

    def on_closing_trade(self, pair):
        self.traded_pairs.remove(pair)
        del self.open_events[pair]

    def on_opening_trade(self, pair, open_event):
        self.traded_pairs.add(pair)
        self.open_events[pair] = open_event

    def get_open_event(self, pair):
        return self.open_events.get(pair, None)

    def is_at_max_trades(self):
        return len(self.traded_pairs) == self.max_trades
