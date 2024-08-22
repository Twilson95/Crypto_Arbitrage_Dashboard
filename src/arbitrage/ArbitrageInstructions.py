import plotly.graph_objects as go
from dash import html, dcc


class ArbitrageInstructions:
    instruction_height = 130

    def __init__(self, arbitrage):
        self.arbitrage = arbitrage
        self.funds = 1000

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

    def return_single_arbitrage_panels(self):
        panels = []
        if "summary_header" in self.arbitrage.keys():
            header_panel = self.build_simple_summary_panel(
                self.arbitrage["summary_header"]
            )
            panels.append(header_panel)
        if "waterfall_data" in self.arbitrage.keys():
            waterfall_panel = self.build_waterfall_panel(
                self.arbitrage["waterfall_data"]
            )
            panels.append(waterfall_panel)
        if "instructions" in self.arbitrage.keys():
            instruction_panels = self.build_all_instruction_panels(
                self.arbitrage["instructions"]
            )
            panels += instruction_panels
        return html.Div(panels)

    def return_triangle_arbitrage_panels(self):
        panels = []
        if "summary_header" in self.arbitrage.keys():
            header_panel = ArbitrageInstructions.build_triangular_summary_panel(
                self.arbitrage["summary_header"]
            )
            panels.append(header_panel)
        if "waterfall_data" in self.arbitrage.keys():
            waterfall_panel = ArbitrageInstructions.build_waterfall_panel(
                self.arbitrage["waterfall_data"]
            )
            panels.append(waterfall_panel)
        if "instructions" in self.arbitrage.keys():
            instruction_panels = ArbitrageInstructions.build_all_instruction_panels(
                self.arbitrage["instructions"]
            )
            panels += instruction_panels
        return html.Div(panels)

    @staticmethod
    def build_simple_summary_panel(summary_data):
        exchanges = summary_data["exchanges_used"]
        total_profit = summary_data["total_profit"]
        summary_text = (
            f"Arbitrage Opportunity: Buy from {exchanges[0]} and sell at "
            f"{exchanges[1]}. Expected profit: ${total_profit:.2f} after fees."
        )
        return html.P(summary_text, style={"margin-bottom": "2px"})

    @staticmethod
    def build_triangular_summary_panel(summary_data):
        coins = summary_data["coins_used"]
        total_profit = summary_data["total_profit"]
        summary_text = (
            f"Arbitrage Opportunity: Buy {coins[0]}, convert to {coins[1]} and sell for an "
            f"expected profit of ${total_profit:.2f} after fees."
        )
        return html.P(summary_text, style={"margin-bottom": "2px"})

    @staticmethod
    def build_all_instruction_panels(instructions):
        return [
            ArbitrageInstructions.build_instruction_panel(instructions)
            for instructions in instructions
        ]

    @staticmethod
    def build_instruction_panel(instruction):
        return dcc.Graph(
            figure=ArbitrageInstructions._generate_exchange_flow(
                instruction["from_exchange"],
                instruction["from_currency"],
                instruction["from_amount"],
                instruction["to_exchange"],
                instruction["to_currency"],
                instruction["to_amount"],
                instruction["total_fees"],
                from_usd=instruction["from_usd"],
                to_usd=instruction["to_usd"],
                instruction=instruction["instruction"],
            ),
            style={
                "height": str(ArbitrageInstructions.instruction_height) + "px",
                "margin-bottom": "2px",
            },
        )

    @staticmethod
    def build_waterfall_panel(waterfall_data):
        categories = list(waterfall_data.keys())
        values = [round(value, 2) for value in waterfall_data.values()]

        # Create a Waterfall chart
        fig = go.Figure()

        fig.add_waterfall(
            name="Profit Calculation",
            orientation="v",
            measure=["relative" for _ in values] + ["total"],
            x=categories,
            cliponaxis=False,
            text=[f"${val}" for val in values],
            y=values,
            connector=dict(line=dict(color="rgba(63, 63, 63, 0.5)")),
        )

        # Update layout for better visualization
        fig.update_layout(
            title="Profit and Fees",
            showlegend=False,
            margin=dict(l=10, r=0, b=10, t=50),
            # xaxis_title="Components",
            yaxis_title="Amount (USD)",
            yaxis=dict(tickprefix="$"),
            template="plotly_dark",
        )

        return dcc.Graph(
            figure=fig,
            style={
                "height": str(ArbitrageInstructions.instruction_height * 2.2) + "px",
                "margin-bottom": "2px",
            },
        )

    def create_summary_instruction(self):
        self.funds = self.arbitrage["buy_price"]  # Reset the funds to $100 at the start
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
                to_usd=to_usd,
                instruction="buy",
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
                instruction="transfer",
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
        to_amount = (self.funds / self.arbitrage["buy_price"]) * self.arbitrage[
            "sell_price"
        ]
        from_amount = start_funds / self.arbitrage["buy_price"]
        funds_change = start_funds - to_amount

        return dcc.Graph(
            figure=self._generate_exchange_flow(
                self.arbitrage["sell_exchange"],
                self.arbitrage["currency"][0],
                from_amount,
                "Wallet",
                self.arbitrage["currency"][1],
                to_amount,
                funds_change,
                from_usd=start_funds,
                instruction="sell",
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

    @staticmethod
    def _generate_exchange_flow(
        from_exchange,
        from_unit,
        from_amount,
        to_exchange,
        to_unit,
        to_amount,
        total_fees,
        from_usd=None,
        to_usd=None,
        instruction=None,
    ):
        fig = go.Figure()

        # arrow
        fig.add_annotation(
            x=3.5,
            y=2,
            xref="x",
            yref="y",
            text="",
            showarrow=True,
            axref="x",
            ayref="y",
            ax=1.5,
            ay=2,
            arrowhead=3,
            arrowwidth=1.5,
            arrowcolor="white",
        )

        # transfer fees
        fig.add_annotation(
            x=0.5,
            y=0.4,
            text=ArbitrageInstructions.format_amount(total_fees, "USD"),
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(color="red" if total_fees > 0 else "green", size=14),
        )

        # Exchange names
        fig.add_annotation(
            x=0.05,
            y=0.8,
            text=from_exchange,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(color="white", size=14),
        )

        fig.add_annotation(
            x=0.95,
            y=0.8,
            text=to_exchange,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(color="white", size=14),
        )

        # from amount
        fig.add_annotation(
            x=0.05,
            y=0.5,
            text=ArbitrageInstructions.format_amount(from_amount, from_unit),
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(color="white", size=14),
        )

        # to amount
        fig.add_annotation(
            x=0.95,
            y=0.5,
            text=ArbitrageInstructions.format_amount(to_amount, to_unit),
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(color="white", size=14),
        )

        if instruction:
            fig.add_annotation(
                x=0.5,
                y=0.75,
                text=instruction,
                showarrow=False,
                xref="paper",
                yref="paper",
                font=dict(color="white", size=12),
            )

        if from_usd:
            fig.add_annotation(
                x=0.05,
                y=0.20,
                text="(" + ArbitrageInstructions.format_amount(from_usd, "USD") + ")",
                showarrow=False,
                xref="paper",
                yref="paper",
                font=dict(color="white", size=12),
            )

        if to_usd:
            fig.add_annotation(
                x=0.95,
                y=0.20,
                text="(" + ArbitrageInstructions.format_amount(to_usd, "USD") + ")",
                showarrow=False,
                xref="paper",
                yref="paper",
                font=dict(color="white", size=12),
            )

        # Update layout to hide axes and grid lines
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=ArbitrageInstructions.instruction_height,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template="plotly_dark",
        )

        return fig

    def create_summary_plot(self):
        return dcc.Graph(
            figure=self.create_waterfall_plot(),
            style={
                "height": str(ArbitrageInstructions.instruction_height * 2.2) + "px",
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
            title="Profit and Fees",
            showlegend=False,
            margin=dict(l=10, r=0, b=10, t=50),
            # xaxis_title="Components",
            yaxis_title="Amount (USD)",
            yaxis=dict(tickprefix="$"),
            template="plotly_dark",
        )

        return fig

    @staticmethod
    def format_amount(value, unit):
        if value == 0:
            return f"0 {unit}"
        elif abs(value) >= 10000:
            return f"{abs(value):,.1f} {unit}"
        elif abs(value) >= 1:
            return f"{abs(value):,.2f} {unit}"
        else:
            return f"{abs(value):.3g} {unit}"
