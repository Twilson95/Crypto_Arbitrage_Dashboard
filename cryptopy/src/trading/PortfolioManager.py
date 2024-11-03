import pandas as pd
from datetime import datetime
from cryptopy import JsonHelper


class PortfolioManager:
    def __init__(self, max_trades=6, funds=1000, trades_path=None):
        self.trades_path = trades_path
        self.funds = funds
        self.portfolio = None
        self.traded_pairs = set()
        self.traded_coins = set()
        self.bought_coins = set()
        self.sold_coins = set()
        self.open_events = dict()
        self.max_trades = max_trades
        self.all_trade_events = None

    def get_funds(self):
        return self.funds

    def read_portfolio(self, portfolio_path):
        self.portfolio = pd.read_csv(portfolio_path)
        self.traded_pairs = set(
            zip(self.portfolio["bought_coin"], self.portfolio["sold_coin"])
        )
        self.bought_coins = set(self.portfolio["bought_coin"])
        self.sold_coins = set(self.portfolio["sold_coin"])

    def read_open_events(self):
        trade_data = JsonHelper.read_from_json(self.trades_path)
        trade_events = trade_data["trade_events"]
        self.all_trade_events = trade_events
        open_events = [event for event in trade_events if "close_event" not in event]
        self.open_events = {
            (event["pair"][0], event["pair"][1]): event for event in open_events
        }
        for open_event in self.open_events.values():
            open_event["open_event"]["date"] = datetime.strptime(
                open_event["open_event"]["date"], "%Y-%m-%d"
            ).date()
        for pair in self.open_events.keys():
            self.traded_pairs.add(pair)

    def get_traded_pairs(self):
        return self.traded_pairs

    def write_to_portfolio(self, portfolio_path):
        self.portfolio.to_csv(portfolio_path)

    def get_portfolio(self):
        return self.portfolio

    def get_all_trade_events(self):
        return self.all_trade_events

    def is_pair_traded(self, pair):
        if pair in self.traded_pairs:
            return True
        if pair[0] in self.traded_coins or pair[1] in self.traded_coins:
            return True
        # if pair[0] in self.bought_coins or pair[1] in self.bought_coins:
        #     return True
        # if pair[0] in self.sold_coins or pair[1] in self.sold_coins:
        #     return True
        return False

    def on_closing_trade(self, pair, profit):
        self.traded_pairs.remove(pair)
        self.funds += profit
        # self.traded_coins.remove(pair[0])
        # self.traded_coins.remove(pair[1])
        del self.open_events[pair]

    def on_opening_trade(self, pair, open_event):
        self.traded_pairs.add(pair)
        # self.traded_coins.add(pair[0])
        # self.traded_coins.add(pair[1])
        self.open_events[pair] = open_event

    def get_open_trades(self, pair):
        return self.open_events.get(pair, None)

    def get_all_open_events(self):
        return self.open_events

    def is_at_max_trades(self):
        return len(self.traded_pairs) == self.max_trades
