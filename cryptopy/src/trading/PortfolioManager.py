import pandas as pd


class PortfolioManager:
    def __init__(self):
        self.portfolio_path = r"data/portfolio_data/Kraken/live_positions.csv"
        self.portfolio = self.read_portfolio()

    def read_portfolio(self):
        return pd.read_csv(self.portfolio_path)

    def get_traded_pairs(self):
        return list(zip(self.portfolio["bought_coin"], self.portfolio["sold_coin"]))

    def write_to_portfolio(self):
        self.portfolio.to_csv(self.portfolio_path)

    def get_portfolio(self):
        return self.portfolio
