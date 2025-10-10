import itertools


class TriangularArbitrage:
    @staticmethod
    def calculate_fees(currency_fees, pair, amount, fee_type="taker"):
        """Calculate the fees for a given trading pair."""
        if currency_fees.get(pair, {}) == {}:
            reversed_pair = f"{pair.split('/')[1]}/{pair.split('/')[0]}"
            return amount * currency_fees.get(reversed_pair, {}).get(fee_type, 0)
        else:
            return amount * currency_fees.get(pair, {}).get(fee_type, 0)

    @staticmethod
    def calculate_conversion_rate(prices, pair):
        """Calculate the conversion rate between two coins."""
        if pair in prices:
            return prices[pair]
        else:
            # If the pair is reversed, return the inverse rate
            reversed_pair = f"{pair.split('/')[1]}/{pair.split('/')[0]}"
            if reversed_pair in prices:
                reversed_price = prices[reversed_pair]
                if reversed_price is not None:
                    return 1 / prices[reversed_pair]
                else:
                    print("We do not have prices for pair or its reverse")
                    return None

    @staticmethod
    def identify_triangle_arbitrage(prices, currency_fees, exchange, funds):
        coins = set()
        for pair in prices.keys():
            coin1, coin2 = pair.split("/")
            coins.add(coin1)
            coins.add(coin2)

        arbitrage_opportunities = []
        closest_opportunity = None
        smallest_loss = float("inf")

        # Fix USD as the start and end currency
        coins.discard("USD")

        for coin1, coin2 in itertools.permutations(coins, 2):
            pair1 = f"USD/{coin1}"  # USD to Coin1
            pair2 = f"{coin1}/{coin2}"  # Coin1 to Coin2
            pair3 = f"{coin2}/USD"  # Coin2 back to USD

            # Calculate conversion rates
            rate1 = TriangularArbitrage.calculate_conversion_rate(prices, pair1)
            rate2 = TriangularArbitrage.calculate_conversion_rate(prices, pair2)
            rate3 = TriangularArbitrage.calculate_conversion_rate(prices, pair3)

            if rate1 is None or rate2 is None or rate3 is None:
                continue

            # Start with enough USD to buy 1 unit of Coin1
            usd_start = funds
            amount1 = usd_start * rate1  # Convert USD to Coin1

            # Calculate fees and convert USD to Coin1
            fees1_usd = TriangularArbitrage.calculate_fees(
                currency_fees, pair1, usd_start, fee_type="taker"
            )
            coin1_after_fees = (usd_start - fees1_usd) * rate1

            # Calculate change in USD for the buy step
            usd_after_buy = usd_start - fees1_usd
            change_in_usd_buy = usd_after_buy - usd_start

            # Calculate fees and convert Coin1 to Coin2
            fees2_coin = TriangularArbitrage.calculate_fees(
                currency_fees, pair2, coin1_after_fees, fee_type="taker"
            )
            coin2_after_fees = (coin1_after_fees - fees2_coin) * rate2

            # Convert Coin2 amount to USD for the transfer step
            usd_after_transfer = coin2_after_fees * rate3

            # Calculate change in USD for the transfer step
            change_in_usd_transfer = usd_after_transfer - usd_after_buy

            # Calculate fees and convert Coin2 back to USD
            fees3_coin = TriangularArbitrage.calculate_fees(
                currency_fees, pair3, coin2_after_fees, fee_type="taker"
            )
            usd_end = (coin2_after_fees - fees3_coin) * rate3

            change_in_usd_sell = usd_end - usd_after_transfer

            profit = usd_end - usd_start

            instructions = []

            # 1. Buy step (USD -> Coin1)
            instructions.append(
                {
                    "instruction": "buy",
                    "from_exchange": exchange,
                    "from_currency": "USD",
                    "from_amount": usd_start,
                    "to_exchange": exchange,
                    "to_currency": coin1,
                    "to_amount": coin1_after_fees,
                    "change_in_usd": change_in_usd_buy,
                    "from_usd": None,
                    "to_usd": usd_after_buy,
                }
            )

            # 2. Transfer step (Coin1 -> Coin2)
            instructions.append(
                {
                    "instruction": "transfer",
                    "from_exchange": exchange,
                    "from_currency": coin1,
                    "from_amount": coin1_after_fees,
                    "to_exchange": exchange,
                    "to_currency": coin2,
                    "to_amount": coin2_after_fees,
                    "change_in_usd": change_in_usd_transfer,
                    "from_usd": usd_after_buy,
                    "to_usd": usd_after_transfer,
                }
            )

            # 3. Sell step (Coin2 -> USD)
            instructions.append(
                {
                    "instruction": "sell",
                    "from_exchange": exchange,
                    "from_currency": coin2,
                    "from_amount": coin2_after_fees,
                    "to_exchange": exchange,
                    "to_currency": "USD",
                    "to_amount": usd_end,
                    "change_in_usd": change_in_usd_sell,
                    "from_usd": usd_after_transfer,
                    "to_usd": None,
                }
            )

            potential_profit = usd_start * (rate1 * rate2 * rate3 - 1)
            waterfall_data = {
                "Potential Profit": potential_profit,
                "Buy Fees": -fees1_usd,
                "Transfer Fees": -fees2_coin / rate1,  # Convert Coin1 fees to USD
                "Sell Fees": -fees3_coin * rate3,  # Convert Coin2 fees to USD
            }

            # Create the summary header
            summary_header = {
                "total_profit": profit,
                "coins_used": [coin1, coin2],
                "exchanges_used": exchange,
            }

            arbitrage_opportunity = {
                "summary_header": summary_header,
                "waterfall_data": waterfall_data,
                "instructions": instructions,
                "path": [("USD", coin1), (coin1, coin2), (coin2, "USD")],
            }

            if profit > 0:
                arbitrage_opportunities.append(arbitrage_opportunity)
            else:
                # Track the closest non-profitable opportunity
                if abs(profit) < smallest_loss:
                    smallest_loss = abs(profit)
                    closest_opportunity = arbitrage_opportunity

        if arbitrage_opportunities:
            # Return the most profitable opportunity
            return sorted(
                arbitrage_opportunities,
                key=lambda x: x["summary_header"]["total_profit"],
                reverse=True,  # Set to True for descending order, False for ascending
            )
        elif closest_opportunity is not None:
            return [closest_opportunity]
        else:
            return None
