from cryptopy import (
    ArbitrageInstructions,
    SimpleArbitrage,
    TriangularArbitrage,
    StatisticalArbitrage,
)
from dash import html

from cryptopy.src.prices.NetworkGraph import create_network_graph


class ArbitrageHandler:
    def __init__(self):
        self.simple_arbitrage_opportunities = []
        self.triangular_arbitrage_opportunities = []
        self.statistical_arbitrage_opportunities = []

    @staticmethod
    def return_simple_arbitrage_instructions(
        currency, exchange_prices, currency_fees, exchange_fees, network_fees, funds
    ):
        arbitrages = SimpleArbitrage.identify_arbitrage(
            currency, exchange_prices, currency_fees, exchange_fees, network_fees, funds
        )
        if not arbitrages:
            return html.Div()

        instruction_diagrams = []
        for arbitrage in arbitrages:
            arbitrage_instructions = ArbitrageInstructions(arbitrage)
            instructions = arbitrage_instructions.return_single_arbitrage_panels()
            instruction_diagrams.append(instructions)
        return instruction_diagrams

    @staticmethod
    def return_triangle_arbitrage_instructions(prices, currency_fees, exchange, funds):
        arbitrages = TriangularArbitrage.identify_triangle_arbitrage(
            prices, currency_fees, exchange, funds
        )

        exchange_network_graph = create_network_graph(prices, arbitrages)

        if not arbitrages:
            return exchange_network_graph, html.Div()

        instruction_diagrams = []

        for arbitrage in arbitrages:
            arbitrage_instructions = ArbitrageInstructions(arbitrage)
            instructions = arbitrage_instructions.return_triangle_arbitrage_panels()
            instruction_diagrams.append(instructions)
        return exchange_network_graph, instruction_diagrams

    @staticmethod
    def return_statistical_arbitrage_instructions(
        prices, cointegration_data, currency_fees, exchange, funds, window=30
    ):
        arbitrages = StatisticalArbitrage.identify_all_statistical_arbitrage(
            prices,
            cointegration_data,
            currency_fees,
            exchange,
            funds,
            window=30,
        )

        if not arbitrages:
            return html.Div(), html.Div()

        arbitrages.reverse()
        arbitrages = arbitrages[:3]

        instruction_diagrams = []

        for arbitrage in arbitrages:
            arbitrage_instructions = ArbitrageInstructions(arbitrage)
            instructions = arbitrage_instructions.return_statistical_arbitrage_panels()
            instruction_diagrams.append(instructions)
        return instruction_diagrams
