import plotly.graph_objects as go
from dash import html, dcc


class ArbitrageInstructions:
    def __init__(self, arbitrage):
        self.arbitrage = arbitrage
        self.funds = 1000
        self.instruction_height = 150

    def return_simple_arbitrage_instructions(self):
        return html.Div(
            [
                self.create_summary_instruction(),
                self.create_summary_plot(),
                self.create_buy_instruction(),
                self.create_transfer_instruction(),
                self.create_sell_instruction(),
            ],
        )

    def create_summary_instruction(self):
        self.funds = 1000  # Reset the funds to $100 at the start
        profit = round(self.arbitrage["profit"], 2)
        summary_text = (
            f"Arbitrage Opportunity: Buy from {self.arbitrage['buy_exchange']} and sell at "
            f"{self.arbitrage['sell_exchange']}. Expected profit: ${profit} after fees."
        )
        return html.P(summary_text, style={"margin-bottom": "2px"})

    def create_buy_instruction(self):
        total_fees = self.funds * self.arbitrage["buy_taker_fee"]
        from_amount = self.funds
        self.funds -= total_fees
        to_usd = self.funds
        to_amount = self.funds / self.arbitrage["buy_price"]

        return dcc.Graph(
            figure=self._generate_exchange_flow(
                "Wallet",
                self.arbitrage["currency"][1],
                from_amount,
                self.arbitrage["buy_exchange"],
                self.arbitrage["currency"][0],
                to_amount,
                total_fees,
                from_usd=None,
                to_usd=None,
            ),
            style={
                "height": str(self.instruction_height) + "px",
                "margin-bottom": "2px",
            },
        )

    def create_transfer_instruction(self):
        fees = [
            self.arbitrage["buy_withdraw_fee"],
            self.arbitrage["sell_deposit_fee"],
            self.arbitrage["network_fees_usd"],
        ]
        start_funds = self.funds
        total_fees = 0
        for fee in fees:
            total_fees += fee
            self.funds -= fee

        end_funds = self.funds

        from_amount = start_funds / self.arbitrage["buy_price"]
        to_amount = end_funds / self.arbitrage["buy_price"]

        return dcc.Graph(
            figure=self._generate_exchange_flow(
                self.arbitrage["buy_exchange"],
                self.arbitrage["currency"][0],
                from_amount,
                self.arbitrage["sell_exchange"],
                self.arbitrage["currency"][0],
                to_amount,
                total_fees,
                from_usd=start_funds,
                to_usd=end_funds,
            ),
            style={
                "height": str(self.instruction_height) + "px",
                "margin-bottom": "2px",
            },
        )

    def create_sell_instruction(self):
        total_fees = self.funds * self.arbitrage["sell_taker_fee"]
        start_funds = self.funds
        self.funds -= total_fees
        to_amount = self.funds
        from_amount = self.funds / self.arbitrage["sell_price"]

        return dcc.Graph(
            figure=self._generate_exchange_flow(
                self.arbitrage["sell_exchange"],
                self.arbitrage["currency"][0],
                from_amount,
                "Wallet",
                self.arbitrage["currency"][1],
                to_amount,
                total_fees,
                from_usd=start_funds,
            ),
            style={
                "height": str(self.instruction_height) + "px",
                "margin-bottom": "2px",
            },
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
            margin=dict(l=20, r=20, t=80, b=80),
            height=self.instruction_height,
            legend=False,
        )

        return fig

    def _generate_exchange_flow(
        self,
        from_exchange,
        from_unit,
        from_amount,
        to_exchange,
        to_unit,
        to_amount,
        total_fees,
        from_usd=None,
        to_usd=None,
    ):
        fig = go.Figure()

        # arrow
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

        # transfer fees
        fig.add_annotation(
            x=0.5,
            y=0.2,
            text=str(round(total_fees, 2)) + " USD",
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(color="red" if total_fees > 0 else "green", size=14),
        )

        # Exchange names
        fig.add_annotation(
            x=0.1,
            y=0.5,
            text=from_exchange,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(color="white", size=14),
        )

        fig.add_annotation(
            x=0.9,
            y=0.5,
            text=to_exchange,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(color="white", size=14),
        )

        fig.add_annotation(
            x=0.9,
            y=0.3,
            text=str(round(to_amount, 2)) + " " + to_unit,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(color="white", size=14),
        )

        if to_usd:
            fig.add_annotation(
                x=0.9,
                y=0.15,
                text="(" + str(round(to_usd, 2)) + " USD)",
                showarrow=False,
                xref="paper",
                yref="paper",
                font=dict(color="white", size=14),
            )

        if from_usd:
            fig.add_annotation(
                x=0.1,
                y=0.15,
                text="(" + str(round(from_usd, 2)) + " USD)",
                showarrow=False,
                xref="paper",
                yref="paper",
                font=dict(color="white", size=14),
            )

        fig.add_annotation(
            x=0.1,
            y=0.3,
            text=str(round(from_amount, 2)) + " " + from_unit,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(color="white", size=14),
        )

        # Update layout to hide axes and grid lines
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=self.instruction_height,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template="plotly_dark",
        )

        return fig

    def create_summary_plot(self):
        return dcc.Graph(
            figure=self.create_waterfall_plot(),
            style={
                "height": str(self.instruction_height * 2) + "px",
                "margin-bottom": "2px",
            },
        )

    def create_waterfall_plot(
        self,
    ):
        fees = [
            -self.arbitrage["buy_taker_fee"] * self.arbitrage["buy_price"],
            -self.arbitrage["buy_withdraw_fee"] * self.arbitrage["buy_price"],
            -self.arbitrage["network_fees_usd"],
            -self.arbitrage["sell_deposit_fee"] * self.arbitrage["sell_price"],
            -self.arbitrage["sell_taker_fee"] * self.arbitrage["sell_price"],
        ]
        # fees = [fee for fee in fees if fee != 0]

        categories = [
            "Delta Price",
            "Buy Fee",
            "Withdrawal Fee",
            "Network Fee",
            "Deposit Fee",
            "Sell Fee",
            "Final Profit",
        ]
        delta_price = self.arbitrage["sell_price"] - self.arbitrage["buy_price"]
        values = [delta_price] + fees

        values = [round(value, 2) for value in values]

        # Create a Waterfall chart
        fig = go.Figure()

        fig.add_waterfall(
            name="Profit Calculation",
            orientation="v",
            measure=[
                "relative",
                "relative",
                "relative",
                "relative",
                "relative",
                "relative",
                "total",
            ],  # 'total' for the final profit
            x=categories,
            cliponaxis=False,
            text=[f"${val}" for val in values],
            y=values,
            connector=dict(line=dict(color="rgba(63, 63, 63, 0.5)")),
        )

        # Update layout for better visualization
        fig.update_layout(
            title="Cryptocurrency Profit Waterfall Chart",
            showlegend=False,
            margin=dict(l=10, r=0, b=10, t=50),
            # xaxis_title="Components",
            yaxis_title="Amount (USD)",
            yaxis=dict(tickprefix="$"),
            template="plotly_dark",
        )

        return fig
