import plotly.graph_objects as go
from dash import html, dcc


class ArbitrageInstructions:
    def __init__(self, arbitrage):
        self.arbitrage = arbitrage
        self.funds = 100

    def return_simple_arbitrage_instructions(self):
        return html.Div(
            [
                self.create_summary_instruction(),
                self.create_buy_instruction(),
                self.create_transfer_instruction(),
                self.create_sell_instruction(),
            ],
        )

    def create_summary_instruction(self):
        self.funds = 100  # Reset the funds to $100 at the start
        profit = round(self.arbitrage["profit"], 2)
        summary_text = f"Arbitrage Opportunity: Buy from {self.arbitrage['buy_exchange']} and sell at {self.arbitrage['sell_exchange']}. Expected profit: ${profit} after fees."
        return html.P(summary_text, style={"margin-bottom": "2px"})

    def create_buy_instruction(self):
        # Calculate the total fees for buying
        fees_amount = (
            self.arbitrage["buy_taker_fee"] + self.arbitrage["buy_withdraw_fee"]
        ) * self.funds
        # Decrease the running total
        self.funds -= fees_amount
        return dcc.Graph(
            figure=self._generate_stacked_bar_chart(fees_amount, "Buy"),
            style={"height": "200px", "margin-bottom": "2px"},
        )

    def create_transfer_instruction(self):
        return dcc.Graph(
            figure=self._generate_exchange_flow(
                self.arbitrage["buy_exchange"], self.arbitrage["sell_exchange"]
            ),
            style={"height": "200px", "margin-bottom": "2px"},
        )

    def create_sell_instruction(self):
        # Calculate the total fees for selling
        fees_amount = (
            self.arbitrage["sell_taker_fee"] + self.arbitrage["sell_deposit_fee"]
        ) * self.funds
        # Decrease the running total
        self.funds -= fees_amount
        return dcc.Graph(
            figure=self._generate_stacked_bar_chart(fees_amount, "Sell"),
            style={"height": "200px", "margin-bottom": "2px"},
        )

    def _generate_stacked_bar_chart(self, fees_amount, action):
        remaining_money = self.funds  # Remaining money after fees

        fig = go.Figure(
            data=[
                go.Bar(
                    name="Remaining Money",
                    x=[remaining_money],
                    y=[f"{action} Fees"],
                    orientation="h",
                    marker=dict(color="green"),
                ),
                go.Bar(
                    name="Fees",
                    x=[fees_amount],
                    y=[f"{action} Fees"],
                    orientation="h",
                    marker=dict(color="red"),
                ),
            ]
        )

        fig.update_layout(
            title=f"{action} Impact on Funds",
            barmode="stack",
            xaxis_title="Amount ($)",
            yaxis_title="",
            xaxis=dict(range=[0, 100]),
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            height=200,
        )

        return fig

    def _generate_exchange_flow(self, exchange_from, exchange_to):
        fig = go.Figure()

        # Arrows indicating the flow of money
        fig.add_annotation(
            x=1,
            y=0.5,  # Position of the arrow's head
            ax=0,
            ay=0.5,  # Position of the arrow's tail
            xref="paper",
            yref="paper",  # Use 'paper' coordinates for the arrow's head
            axref="pixel",
            ayref="pixel",  # Corrected to 'paper' to maintain consistency
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="white",
        )

        # Exchange names
        fig.add_annotation(
            x=0, y=0.5, text=exchange_from, showarrow=False, xref="paper", yref="paper"
        )

        fig.add_annotation(
            x=1, y=0.5, text=exchange_to, showarrow=False, xref="paper", yref="paper"
        )

        fig.update_layout(
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20),  # Adjusted margins
            height=200,
            template="plotly_dark",
        )

        return fig
