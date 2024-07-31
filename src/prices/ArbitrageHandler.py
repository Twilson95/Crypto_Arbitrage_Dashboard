class ArbitrageHandler:

    def return_simple_arbitrage(self, exchange_prices, fees):
        lowest_price_fee = ["exchange", float("inf")]
        highest_price_fee = ["exchange", float("-inf")]

        for exchange, prices in exchange_prices.items():
            taker_fee = fees.get(exchange, {}).get("taker", 0)
            price = prices.close[-1]
            price_minus_fee = price * (1 - taker_fee)
            price_plus_fee = price * (1 + taker_fee)

            if price_plus_fee < lowest_price_fee[1]:
                lowest_price_fee[0] = exchange
                lowest_price_fee[1] = price_plus_fee

            if price_minus_fee > highest_price_fee[1]:
                highest_price_fee[0] = exchange
                highest_price_fee[1] = price_minus_fee

        arbitrage_opportunity = highest_price_fee[1] - lowest_price_fee[1]
        # print(lowest_price_fee)
        # print(highest_price_fee)
        print(f"arbitrage opportunity: {arbitrage_opportunity}")

        if arbitrage_opportunity < 0:
            exchange = lowest_price_fee[0]
            low_price = exchange_prices[exchange].close[-1]
            low_fee = fees[exchange]["taker"]

            exchange = highest_price_fee[0]
            high_price = exchange_prices[exchange].close[-1]
            high_fee = fees[exchange]["taker"]

            total_fees = low_price * low_fee + high_price * high_fee

            print(f"Prices difference needs to be at least {total_fees}")
