import plotly.graph_objects as go
from dash import html, dcc


class ArbitrageInstructions:
    def __init__(self, arbitrage):
        self.arbitrage = arbitrage
        self.funds = 1000

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
        self.funds = 1000  # Reset the funds to $100 at the start
        profit = round(self.arbitrage["profit"], 2)
        summary_text = f"Arbitrage Opportunity: Buy from {self.arbitrage['buy_exchange']} and sell at {self.arbitrage['sell_exchange']}. Expected profit: ${profit} after fees."
        return html.P(summary_text, style={"margin-bottom": "2px"})

    def create_buy_instruction(self):
        # Calculate the total fees for buying
        fees_amount = (
            self.arbitrage["buy_taker_fee"] + self.arbitrage["buy_withdraw_fee"]
        ) * self.funds
        # print(self.arbitrage)
        # print("fees", fees_amount)

        # Decrease the running total
        self.funds -= fees_amount
        return dcc.Graph(
            figure=self._generate_stacked_bar_chart(fees_amount, "Buy"),
            style={"height": "200px", "margin-bottom": "2px"},
        )

    def create_transfer_instruction(self):
        return dcc.Graph(
            figure=self._generate_exchange_flow(
                self.arbitrage["buy_exchange"],
                self.arbitrage["sell_exchange"],
                self.funds,
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

    def _generate_exchange_flow(self, exchange_from, exchange_to, transfer_amount):
        fig = go.Figure()

        fig.add_annotation(
            x=3.5,
            y=1.5,
            xref="x",
            yref="y",
            text="",
            showarrow=True,
            axref="x",
            ayref="y",
            ax=1,
            ay=1.5,
            arrowhead=3,
            arrowwidth=1.5,
            arrowcolor="white",
        )

        # transfer amount
        fig.add_annotation(
            x=0.5,
            y=0.3,
            text=transfer_amount,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(color="white", size=14),
        )

        # Exchange names
        fig.add_annotation(
            x=0.1,
            y=0.5,
            text=exchange_from,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(color="white", size=14),
        )

        fig.add_annotation(
            x=0.9,
            y=0.5,
            text=exchange_to,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(color="white", size=14),
        )

        # Update layout to hide axes and grid lines
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=200,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template="plotly_dark",
        )

        return fig
