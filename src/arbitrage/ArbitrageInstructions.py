class ArbitrageInstructions:
    def __init__(self):
        self.exchange_1 = None
        self.exchange_2 = None
        self.instruction_diagram = None

    def return_simple_arbitrage_instructions(self):
        return [
            self.create_summary_instruction(),
            self.create_buy_instruction(),
            self.create_transfer_instruction(),
            self.create_sell_instruction(),
        ]

    def create_summary_instruction(self):
        self.exchange_1 = 1
        self.exchange_2 = 2
        return

    def create_buy_instruction(self):
        pass

    def create_transfer_instruction(self):
        pass

    def create_sell_instruction(self):
        pass
