from ArbitrageInstructions import ArbitrageInstructions


class ArbitrageHandler:
    def __init__(self):

    def return_simple_arbitrage(self, exchange_prices, currency_fees, exchange_fees):
        arbitrages = self.identify_simple_arbitrage(exchange_prices, currency_fees)
        instruction_diagrams = []
        for arbitrage in arbitrages:
            arbitrage_instructions = ArbitrageInstructions(arbitrage)
            instructions = arbitrage_instructions.return_simple_arbitrage_instructions()
            intruction_diagrams.append(instructions)
        return instruction_diagrams

    def identify_simple_arbitrage(self, exchange_prices, currency_fees):
        """
        return list of arbitrage opportunities, if none exist return the closest
        """

        lowest_price_fee = ["exchange", float("inf"), "fee"]
        highest_price_fee = ["exchange", float("-inf"), "fee"]
        # print(fees)

        for exchange, prices in exchange_prices.items():
            taker_fee = currency_fees.get(exchange, {}).get("taker", 0)
            close_price = prices.close
            if len(close_price) == 0:
                print(exchange, "has no prices")
                continue
            price = prices.close[-1]
            price_minus_fee = price * (1 - taker_fee)
            price_plus_fee = price * (1 + taker_fee)

            if price_plus_fee < lowest_price_fee[1]:
                lowest_price_fee[0] = exchange
                lowest_price_fee[1] = price_plus_fee
                lowest_price_fee[2] = taker_fee

            if price_minus_fee > highest_price_fee[1]:
                highest_price_fee[0] = exchange
                highest_price_fee[1] = price_minus_fee
                highest_price_fee[2] = taker_fee

        arbitrage_opportunity = highest_price_fee[1] - lowest_price_fee[1]
        print(lowest_price_fee)
        print(highest_price_fee)
        print(f"arbitrage opportunity: {arbitrage_opportunity}")

        if arbitrage_opportunity < 0:
            exchange = lowest_price_fee[0]
            low_price = exchange_prices[exchange].close[-1]
            low_fee = currency_fees[exchange]["taker"]

            exchange = highest_price_fee[0]
            high_price = exchange_prices[exchange].close[-1]
            high_fee = currency_fees[exchange]["taker"]

            total_fees = low_price * low_fee + high_price * high_fee

            print(f"Prices difference needs to be at least {total_fees}")


# get prices across exchanges
# apply all maker/taker fees
# apply all deposit and withdrawal fees
# apply network fees
# get best pair
