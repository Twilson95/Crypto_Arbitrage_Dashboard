import pandas as pd


class PortfolioManager:
    def __init__(self):
        self.portfolio = pd.DataFrame()
        self.portfolio_path = r"data/portfolio_data/Kraken/live_positions.csv"

    def read_live_trades(self):
        self.portfolio = pd.read_csv(self.portfolio_path)

    def write_live_trades(self):
        self.portfolio.to_csv(self.portfolio_path)
        pass

    def get_portfolio(self):
        return self.portfolio
