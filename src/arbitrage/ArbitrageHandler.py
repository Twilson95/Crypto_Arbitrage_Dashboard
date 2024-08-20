from src.arbitrage.ArbitrageInstructions import ArbitrageInstructions


class ArbitrageHandler:

    @staticmethod
    def return_simple_arbitrage_instructions(
        currency, exchange_prices, currency_fees, exchange_fees, network_fees
    ):
        arbitrages = ArbitrageHandler.identify_simple_arbitrage(
            currency, exchange_prices, currency_fees, exchange_fees, network_fees
        )
        # print("arbitrages", arbitrages)
        instruction_diagrams = []
        for arbitrage in arbitrages:
            arbitrage_instructions = ArbitrageInstructions(arbitrage)
            instructions = arbitrage_instructions.return_simple_arbitrage_instructions()
            instruction_diagrams.append(instructions)
        return instruction_diagrams

    @staticmethod
    def identify_simple_arbitrage(
        currency, exchange_prices, currency_fees, exchange_fees, network_fees
    ):
        """
        Returns a list of arbitrage opportunities, if none exist return the closest opportunity.
        Includes deposit/withdrawal fees, and network fees only when transferring between exchanges.
        Network fee is in the cryptocurrency being traded, with an estimate of its value in USD.
        """
        arbitrage_opportunities = []
        closest_opportunity = None
        closest_difference = float("inf")

        # Iterate over all pairs of exchanges
        for buy_exchange, prices_buy in exchange_prices.items():
            buy_taker_fee = currency_fees.get(buy_exchange, {}).get("taker", 0)
            withdraw_fee = exchange_fees.get(buy_exchange, {}).get("withdraw", 0)

            close_price_buy = prices_buy.close
            if len(close_price_buy) == 0:
                print(buy_exchange, "has no prices")
                continue
            buy_price = close_price_buy[-1]
            price_plus_fee_buy = buy_price * (1 + buy_taker_fee + withdraw_fee)

            for sell_exchange, prices_sell in exchange_prices.items():
                if sell_exchange == buy_exchange:
                    continue

                sell_taker_fee = currency_fees.get(sell_exchange, {}).get("taker", 0)
                deposit_fee = exchange_fees.get(sell_exchange, {}).get("deposit", 0)

                close_price_sell = prices_sell.close
                if len(close_price_sell) == 0:
                    print(sell_exchange, "has no prices")
                    continue
                sell_price = close_price_sell[-1]

                # Calculate the network fee only if a transfer is needed
                network_fee_crypto = (
                    network_fees if buy_exchange != sell_exchange else 0
                )

                # Adjust the amount of cryptocurrency after the network fee
                effective_crypto_amount = (
                    1 - network_fee_crypto
                )  # Assuming starting with 1 unit of crypto
                effective_sell_price = sell_price * effective_crypto_amount

                # Adjust the fees based on the reduced amount of cryptocurrency
                price_minus_fee_sell = effective_sell_price * (
                    1 - sell_taker_fee - deposit_fee
                )

                network_fee_usd = network_fee_crypto * sell_price

                # Calculate potential arbitrage opportunity
                arbitrage_profit = price_minus_fee_sell - price_plus_fee_buy

                # Calculate total fees excluding network_fee_usd since it's already accounted in effective_sell_price
                total_fees = (buy_price * (buy_taker_fee + withdraw_fee)) + (
                    effective_sell_price * (sell_taker_fee + deposit_fee)
                )

                # arbitrage_details = [
                #     {
                #         "instruction": "summary",
                #         "profit": arbitrage_profit,
                #         "total_fees": total_fees,
                #         "network_fees": network_fees,
                #     },
                #     {
                #         "instruction": "buy",
                #         "from_exchange": buy_exchange,
                #         "from_coin": currency.split("/")[1],
                #         "conversion_rate": buy_price,
                #         "fees": {"taker fee": buy_taker_fee},
                #         "to_exchange": buy_exchange,
                #         "to_coin": currency.split("/")[0],
                #         "to_price": sell_price,
                #     },
                #     {
                #         "instruction": "transfer",
                #         "from_exchange": buy_exchange,
                #         "from_coin": currency.split("/")[0],
                #         "conversion_rate": 1,
                #         "fees": {
                #             "Withdrawal Fee": withdraw_fee,
                #             "Deposit Fee": deposit_fee,
                #         },
                #         "to_coin": currency.split("/")[0],
                #         "to_exchange": sell_exchange,
                #     },
                #     {
                #         "instruction": "sell",
                #         "from_exchange": sell_exchange,
                #         "from_coin": currency.split("/")[0],
                #         "conversion_rate": 1 / sell_price,
                #         "fees": {"taker fee": sell_taker_fee},
                #         "to_exchange": sell_exchange,
                #         "to_coin": currency.split("/")[1],
                #     },
                # ]

                arbitrage_details = {
                    "currency": currency.split("/"),
                    "buy_exchange": buy_exchange,
                    "buy_price": buy_price,
                    "buy_taker_fee": buy_taker_fee,
                    "buy_withdraw_fee": withdraw_fee,
                    "sell_exchange": sell_exchange,
                    "sell_price": sell_price,
                    "effective_sell_price": effective_sell_price,
                    "sell_taker_fee": sell_taker_fee,
                    "sell_deposit_fee": deposit_fee,
                    "profit": arbitrage_profit,
                    "network_fees_crypto": network_fee_crypto,
                    "network_fees_usd": network_fee_usd,
                    "total_fees": total_fees,
                }

                if arbitrage_profit > 0:
                    arbitrage_opportunities.append(arbitrage_details)
                else:
                    difference = abs(arbitrage_profit)
                    if difference < closest_difference:
                        closest_difference = difference
                        closest_opportunity = arbitrage_details

        if arbitrage_opportunities:
            return arbitrage_opportunities
        else:
            return [closest_opportunity]

        #
        # def return_triangle_arbitrage_instructions(
        #     prices, currency_fees, exchange_fees
        # ):
        #     # all_prices, all_currency_fees = generate_intercurrency_values(
        #     #     prices, currency_fees
        #     # )
        #     # arbitrages = identify_triangle_arbitrage(all_prices, all_currency_fees, exchange_fees)
        #
        #     instruction_diagrams = []
        #     for arbitrage in arbitrages:
        #         arbitrage_instructions = ArbitrageInstructions(arbitrage)
        #         instructions = (
        #             arbitrage_instructions.return_triangle_arbitrage_instructions()
        #         )
        #         instruction_diagrams.append(instructions)
        #     return instruction_diagrams
        #
        #     pass
        #
        # def identify_triangle_arbitrage(exchange_prices, currency_fees, exchange_fees):
        #     pass
