from src.arbitrage.ArbitrageInstructions import ArbitrageInstructions
import itertools


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
            instructions = arbitrage_instructions.return_single_arbitrage_panels()
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
                # arbitrage_profit = sell_price - buy_price - total_fees

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
                    arbitrage_opportunities.append(
                        ArbitrageHandler.create_arbitrage_simple_instructions_data(
                            arbitrage_details
                        )
                    )

                else:
                    difference = abs(arbitrage_profit)
                    if difference < closest_difference:
                        closest_difference = difference
                        closest_opportunity = (
                            ArbitrageHandler.create_arbitrage_simple_instructions_data(
                                arbitrage_details
                            )
                        )

        if arbitrage_opportunities:
            return arbitrage_opportunities
        else:
            return [closest_opportunity]

    def return_triangle_arbitrage_instructions(self, prices, currency_fees, exchange):
        all_prices, all_fees = self.generate_crypto_to_crypto_pairs(
            prices, currency_fees
        )

        arbitrages = self.identify_triangle_arbitrage(all_prices, all_fees, exchange)

        instruction_diagrams = []
        for arbitrage in arbitrages:
            arbitrage_instructions = ArbitrageInstructions(arbitrage)
            instructions = arbitrage_instructions.return_triangle_arbitrage_panels()
            instruction_diagrams.append(instructions)
        return instruction_diagrams

    @staticmethod
    def generate_crypto_to_crypto_pairs(prices, currency_fees):
        """Generate synthetic crypto-to-crypto pairs and their fees."""
        new_prices = prices.copy()
        new_fees = currency_fees.copy()

        cryptos = set()
        for pair in prices.keys():
            crypto, _ = pair.split("/")
            cryptos.add(crypto)

        # Generate synthetic pairs
        for crypto1, crypto2 in itertools.permutations(cryptos, 2):
            if crypto1 == crypto2:
                continue
            pair = f"{crypto1}/{crypto2}"
            reverse_pair = f"{crypto2}/{crypto1}"

            # Calculate synthetic price
            price_crypto1_usd = prices.get(f"{crypto1}/USD")
            price_crypto2_usd = prices.get(f"{crypto2}/USD")

            if price_crypto1_usd and price_crypto2_usd:
                new_prices[pair] = price_crypto1_usd / price_crypto2_usd
                new_prices[reverse_pair] = price_crypto2_usd / price_crypto1_usd

                # Fees for synthetic pairs are the same as for the crypto/USD pairs
                new_fees[pair] = currency_fees[f"{crypto1}/USD"]
                new_fees[reverse_pair] = currency_fees[f"{crypto2}/USD"]

        return new_prices, new_fees

    @staticmethod
    def calculate_conversion_rate(prices, pair):
        """Calculate the conversion rate between two coins."""
        if pair in prices:
            return prices[pair]
        else:
            # If the pair is reversed, return the inverse rate
            reversed_pair = f"{pair.split('/')[1]}/{pair.split('/')[0]}"
            if reversed_pair in prices:
                return 1 / prices[reversed_pair]
        return None
        # return prices.get(pair)

    @staticmethod
    def calculate_fees(currency_fees, pair, amount, fee_type="taker"):
        """Calculate the fees for a given trading pair."""
        if currency_fees.get(pair, {}) == {}:
            reversed_pair = f"{pair.split('/')[1]}/{pair.split('/')[0]}"
            return amount * currency_fees.get(reversed_pair, {}).get(fee_type, 0)
        else:
            return amount * currency_fees.get(pair, {}).get(fee_type, 0)

    @staticmethod
    def identify_triangle_arbitrage(prices, currency_fees, exchange):
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
            pair1 = f"{coin1}/USD"  # USD to Coin1
            pair2 = f"{coin2}/{coin1}"  # Coin1 to Coin2
            pair3 = f"USD/{coin2}"  # Coin2 back to USD

            # Calculate conversion rates
            rate1 = ArbitrageHandler.calculate_conversion_rate(prices, pair1)
            rate2 = ArbitrageHandler.calculate_conversion_rate(prices, pair2)
            rate3 = ArbitrageHandler.calculate_conversion_rate(prices, pair3)

            if rate1 is None or rate2 is None or rate3 is None:
                continue

            # Start with 1 unit of USD
            amount1 = 1
            amount2 = amount1 * rate1
            amount3 = amount2 * rate2

            # rate2_breakeven = rate1 *
            rate2_breakeven = 1 / (rate3 * rate1)
            print(pair1, rate1, pair2, rate2, pair3, rate3, rate2_breakeven)
            # Potential profit before fees based on difference between actual rate2 and rate2_breakeven
            potential_profit_before_fees = (rate2 - rate2_breakeven) / rate3

            # Apply fees in terms of the relevant currency
            # Fee calculation in the respective coin
            fees1_coin = ArbitrageHandler.calculate_fees(
                currency_fees, pair1, amount1 * rate1, fee_type="taker"
            )
            amount2 -= fees1_coin

            fees2_coin = ArbitrageHandler.calculate_fees(
                currency_fees, pair2, amount2, fee_type="taker"
            )
            amount3 -= fees2_coin

            # Convert to USD for final profit calculation
            final_amount = amount3 * rate3

            # Apply final fees in terms of USD
            fees3_usd = ArbitrageHandler.calculate_fees(
                currency_fees, pair3, final_amount, fee_type="taker"
            )
            final_amount -= fees3_usd

            # Convert all fees to USD for profit calculation
            fees1_usd = fees1_coin / rate1  # Convert Coin1 fees back to USD
            fees2_usd = (fees2_coin / rate3) * rate2  # Convert Coin2 fees back to USD

            total_fees_usd = fees1_usd + fees2_usd + fees3_usd
            profit = final_amount - (amount1 + total_fees_usd)

            arbitrage_opportunity = {
                "path": ["USD", coin1, coin2, "USD"],
                "conversion_rates": [rate1, rate2, rate3],
                "fees_coin": [fees1_coin, fees2_coin, fees3_usd],
                "fees_usd": [fees1_usd, fees2_usd, fees3_usd],
                "final_amount": final_amount,
                "profit": profit,
                "potential_profit": potential_profit_before_fees,
                "exchange": exchange,
            }

            if profit > 0:
                arbitrage_opportunities.append(
                    ArbitrageHandler.create_arbitrage_triangular_instructions_data(
                        arbitrage_opportunity
                    )
                )
            else:
                # Track the closest non-profitable opportunity
                if abs(profit) < smallest_loss:
                    smallest_loss = abs(profit)
                    closest_opportunity = (
                        ArbitrageHandler.create_arbitrage_triangular_instructions_data(
                            arbitrage_opportunity
                        )
                    )

        if arbitrage_opportunities:
            # Return the most profitable opportunity
            return arbitrage_opportunities
        else:
            # Return the closest non-profitable opportunity
            return [closest_opportunity]

    @staticmethod
    def create_arbitrage_triangular_instructions_data(opportunity):
        instructions = []

        # Extract data from opportunity
        coin1, coin2 = opportunity["path"][1], opportunity["path"][2]
        rate1, rate2, rate3 = opportunity["conversion_rates"]
        fees1_usd, fees2_usd, fees3_usd = opportunity["fees_usd"]
        fees1_coin, fees2_coin, fees3_coin = opportunity["fees_coin"]
        total_profit = opportunity["profit"]
        potential_profit = opportunity["potential_profit"]
        exchange = opportunity["exchange"]

        # 1. Buy step (USD -> Coin1)
        from_amount = rate1
        to_amount = (rate1 - fees1_coin) / rate1  # Convert USD to Coin1
        to_usd = rate1 - fees1_usd

        instructions.append(
            {
                "instruction": "buy",
                "from_exchange": exchange,
                "from_currency": "usd",
                "from_amount": from_amount,
                "to_exchange": exchange,
                "to_currency": coin1,
                "to_amount": to_amount,
                "total_fees": fees1_usd,
                "from_usd": None,
                "to_usd": to_usd,
            }
        )

        # Update from_usd for the next step
        from_usd = to_usd

        from_amount = to_amount
        to_amount = (from_amount - fees2_coin) / rate2  # Convert USD to Coin1
        to_usd = to_amount / rate3

        instructions.append(
            {
                "instruction": "transfer",
                "from_exchange": exchange,
                "from_currency": coin1,
                "from_amount": from_amount,
                "to_exchange": exchange,
                "to_currency": coin2,
                "to_amount": to_amount,
                "total_fees": -fees2_usd,
                "from_usd": from_usd,
                "to_usd": to_usd,
            }
        )

        # Update from_usd for the next step
        from_usd = to_usd

        from_amount = to_amount
        to_amount = (from_amount - fees3_coin) / rate3  # Convert USD to Coin1
        instructions.append(
            {
                "instruction": "sell",
                "from_exchange": exchange,
                "from_currency": coin2,
                "from_amount": from_amount,
                "to_exchange": exchange,
                "to_currency": "usd",
                "to_amount": to_amount,
                "total_fees": fees3_usd,
                "from_usd": from_usd,
                "to_usd": None,
            }
        )

        # Calculate the waterfall data
        waterfall_data = {
            "Potential Profit": potential_profit,
            "Buy Fees": -fees1_usd,
            "Transfer Fees": -fees2_usd,
            "Sell Fees": -fees3_usd,
        }

        # Create the summary header
        summary_header = {
            "total_profit": total_profit,
            "coins_used": [coin1, coin2],
            "exchanges_used": exchange,
        }

        return {
            "summary_header": summary_header,
            "waterfall_data": waterfall_data,
            "instructions": instructions,
        }

    @staticmethod
    def create_arbitrage_simple_instructions_data(opportunity):
        # Extract data
        currency_pair = opportunity["currency"]
        buy_exchange = opportunity["buy_exchange"]
        sell_exchange = opportunity["sell_exchange"]
        total_profit = opportunity["profit"]

        # Summary Header
        summary_header = {
            "total_profit": total_profit,
            "currency": currency_pair,
            "exchanges_used": [buy_exchange, sell_exchange],
        }

        # Waterfall Plot Data
        waterfall_data = {
            "Price Delta": opportunity["sell_price"] - opportunity["buy_price"],
            "Buy Fees": -opportunity["buy_price"] * opportunity["buy_taker_fee"],
            "Withdraw Fee": -opportunity["buy_withdraw_fee"],
            "Network Fee": -opportunity["network_fees_usd"],
            "Deposit Fee": -opportunity["effective_sell_price"]
            * opportunity["sell_deposit_fee"],
            "Sell Fees": -opportunity["effective_sell_price"]
            * opportunity["sell_taker_fee"],
        }

        # Instructions
        instructions = []

        # Buy Step
        from_usd = opportunity["buy_price"]
        fees = opportunity["buy_price"] * opportunity["buy_taker_fee"]
        to_crypto = (from_usd - fees) / opportunity[
            "buy_price"
        ]  # Convert USD to Crypto
        to_usd = to_crypto * opportunity["buy_price"]
        funds = to_usd

        instructions.append(
            {
                "instruction": "buy",
                "from_exchange": buy_exchange,
                "from_currency": "USD",
                "from_amount": from_usd,
                "to_exchange": buy_exchange,
                "to_currency": currency_pair[0],
                "to_amount": to_crypto,
                "total_fees": fees,
                "from_usd": None,
                "to_usd": to_usd,
            }
        )

        # Transfer Step (if applicable)
        if opportunity["network_fees_crypto"] > 0:
            from_crypto = to_crypto
            from_usd = funds
            to_crypto = from_crypto * (1 - opportunity["buy_withdraw_fee"])
            to_crypto -= opportunity["network_fees_crypto"]
            to_crypto *= 1 - opportunity["sell_deposit_fee"]

            to_usd = to_crypto * opportunity["sell_price"]
            fees = from_usd - to_usd

            instructions.append(
                {
                    "instruction": "transfer",
                    "from_exchange": buy_exchange,
                    "from_currency": currency_pair[0],
                    "from_amount": from_crypto,
                    "to_exchange": sell_exchange,
                    "to_currency": currency_pair[0],
                    "to_amount": to_crypto,
                    "total_fees": fees,
                    "from_usd": from_usd,
                    "to_usd": to_usd,
                }
            )

        # Sell Step
        from_crypto = to_crypto
        from_usd = to_usd

        fees = opportunity["effective_sell_price"] * opportunity["sell_taker_fee"]
        to_usd = from_crypto * opportunity["sell_price"] - fees

        instructions.append(
            {
                "instruction": "sell",
                "from_exchange": sell_exchange,
                "from_currency": currency_pair[0],
                "from_amount": from_crypto,
                "to_exchange": sell_exchange,
                "to_currency": "USD",
                "to_amount": to_usd,
                "total_fees": fees,
                "from_usd": from_usd,
                "to_usd": None,
            }
        )

        return {
            "summary_header": summary_header,
            "waterfall_data": waterfall_data,
            "instructions": instructions,
        }
