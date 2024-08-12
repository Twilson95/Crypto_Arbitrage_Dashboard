from src.arbitrage.ArbitrageInstructions import ArbitrageInstructions


class ArbitrageHandler:
    def __init__(self):
        pass

    def return_simple_arbitrage(
        self, currency, exchange_prices, currency_fees, exchange_fees, network_fees
    ):
        arbitrages = self.identify_simple_arbitrage(
            currency, exchange_prices, currency_fees, exchange_fees, network_fees
        )
        # print("arbitrages", arbitrages)
        instruction_diagrams = []
        for arbitrage in arbitrages:
            arbitrage_instructions = ArbitrageInstructions(arbitrage)
            instructions = arbitrage_instructions.return_simple_arbitrage_instructions()
            instruction_diagrams.append(instructions)
        return instruction_diagrams

    # def identify_simple_arbitrage(
    #     self, exchange_prices, currency_fees, exchange_fees, network_fees
    # ):
    #     """
    #     Returns a list of arbitrage opportunities, if none exist return the closest opportunity.
    #     Includes deposit/withdrawal fees, and network fees only when transferring between exchanges.
    #     """
    #     arbitrage_opportunities = []
    #     closest_opportunity = None
    #     closest_difference = float("inf")
    #
    #     # Iterate over all pairs of exchanges
    #     for exchange_buy, prices_buy in exchange_prices.items():
    #         taker_fee_buy = currency_fees.get(exchange_buy, {}).get("taker", 0)
    #         # deposit_fee_buy = exchange_fees.get(exchange_buy, {}).get("deposit", 0)
    #         withdraw_fee_buy = exchange_fees.get(exchange_buy, {}).get("withdraw", 0)
    #
    #         close_price_buy = prices_buy.close
    #         if len(close_price_buy) == 0:
    #             print(exchange_buy, "has no prices")
    #             continue
    #         price_buy = close_price_buy[-1]
    #         price_plus_fee_buy = price_buy * (1 + taker_fee_buy + withdraw_fee_buy)
    #
    #         for exchange_sell, prices_sell in exchange_prices.items():
    #             if exchange_sell == exchange_buy:
    #                 continue
    #
    #             taker_fee_sell = currency_fees.get(exchange_sell, {}).get("taker", 0)
    #             deposit_fee_sell = exchange_fees.get(exchange_sell, {}).get(
    #                 "deposit", 0
    #             )
    #             # withdraw_fee_sell = exchange_fees.get(exchange_sell, {}).get(
    #             #     "withdraw", 0
    #             # )
    #
    #             close_price_sell = prices_sell.close
    #             if len(close_price_sell) == 0:
    #                 print(exchange_sell, "has no prices")
    #                 continue
    #             price_sell = close_price_sell[-1]
    #             price_minus_fee_sell = price_sell * (
    #                 1 - taker_fee_sell - deposit_fee_sell
    #             )
    #
    #             # Calculate the network fee only if a transfer is needed
    #             network_fee = network_fees if exchange_buy != exchange_sell else 0
    #
    #             # Calculate potential arbitrage opportunity
    #             arbitrage_profit = price_minus_fee_sell - (
    #                 price_plus_fee_buy + network_fee
    #             )
    #             total_fees = (
    #                 (price_buy * (taker_fee_buy + withdraw_fee_buy))
    #                 + (price_sell * (taker_fee_sell + deposit_fee_sell))
    #                 + network_fee
    #             )
    #
    #             arbitrage_details = {
    #                 "buy_exchange": exchange_buy,
    #                 "buy_price": price_buy,
    #                 "buy_taker_fee": taker_fee_buy,
    #                 "buy_withdraw_fee": withdraw_fee_buy,
    #                 "sell_exchange": exchange_sell,
    #                 "sell_price": price_sell,
    #                 "sell_taker_fee": taker_fee_sell,
    #                 "sell_deposit_fee": deposit_fee_sell,
    #                 "profit": arbitrage_profit,
    #                 "network_fees": network_fees,
    #                 "total_fees": total_fees,
    #             }
    #
    #             if arbitrage_profit > 0:
    #                 arbitrage_opportunities.append(arbitrage_details)
    #             else:
    #                 difference = abs(arbitrage_profit)
    #                 if difference < closest_difference:
    #                     closest_difference = difference
    #                     closest_opportunity = arbitrage_details
    #
    #     if arbitrage_opportunities:
    #         return arbitrage_opportunities
    #     else:
    #         return [closest_opportunity]

    def identify_simple_arbitrage(
        self, currency, exchange_prices, currency_fees, exchange_fees, network_fees
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
        for exchange_buy, prices_buy in exchange_prices.items():
            taker_fee_buy = currency_fees.get(exchange_buy, {}).get("taker", 0)
            withdraw_fee_buy = exchange_fees.get(exchange_buy, {}).get("withdraw", 0)

            close_price_buy = prices_buy.close
            if len(close_price_buy) == 0:
                print(exchange_buy, "has no prices")
                continue
            price_buy = close_price_buy[-1]
            price_plus_fee_buy = price_buy * (1 + taker_fee_buy + withdraw_fee_buy)

            for exchange_sell, prices_sell in exchange_prices.items():
                if exchange_sell == exchange_buy:
                    continue

                taker_fee_sell = currency_fees.get(exchange_sell, {}).get("taker", 0)
                deposit_fee_sell = exchange_fees.get(exchange_sell, {}).get(
                    "deposit", 0
                )

                close_price_sell = prices_sell.close
                if len(close_price_sell) == 0:
                    print(exchange_sell, "has no prices")
                    continue
                price_sell = close_price_sell[-1]

                # Calculate the network fee only if a transfer is needed
                network_fee_crypto = (
                    network_fees if exchange_buy != exchange_sell else 0
                )

                # Adjust the amount of cryptocurrency after the network fee
                effective_crypto_amount = (
                    1 - network_fee_crypto
                )  # Assuming starting with 1 unit of crypto
                effective_sell_price = price_sell * effective_crypto_amount

                # Adjust the fees based on the reduced amount of cryptocurrency
                price_minus_fee_sell = effective_sell_price * (
                    1 - taker_fee_sell - deposit_fee_sell
                )

                network_fee_usd = network_fee_crypto * price_sell

                # Calculate potential arbitrage opportunity
                arbitrage_profit = price_minus_fee_sell - price_plus_fee_buy

                # Calculate total fees excluding network_fee_usd since it's already accounted in effective_sell_price
                total_fees = (price_buy * (taker_fee_buy + withdraw_fee_buy)) + (
                    effective_sell_price * (taker_fee_sell + deposit_fee_sell)
                )

                arbitrage_details = {
                    "currency": currency.split("/"),
                    "buy_exchange": exchange_buy,
                    "buy_price": price_buy,
                    "buy_taker_fee": taker_fee_buy,
                    "buy_withdraw_fee": withdraw_fee_buy,
                    "sell_exchange": exchange_sell,
                    "sell_price": price_sell,
                    "effective_sell_price": effective_sell_price,
                    "sell_taker_fee": taker_fee_sell,
                    "sell_deposit_fee": deposit_fee_sell,
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
