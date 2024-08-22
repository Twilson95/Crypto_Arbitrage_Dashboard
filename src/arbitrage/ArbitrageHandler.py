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

            # Assume you start with 1 unit of USD
            amount1 = 1
            amount2 = amount1 * rate1
            amount3 = amount2 * rate2
            final_amount = amount3 * rate3

            # Calculate fees (assuming taker fees for simplicity)
            fees1 = ArbitrageHandler.calculate_fees(
                currency_fees, pair1, amount1, fee_type="taker"
            )
            fees2 = ArbitrageHandler.calculate_fees(
                currency_fees, pair2, amount2, fee_type="taker"
            )
            fees3 = ArbitrageHandler.calculate_fees(
                currency_fees, pair3, amount3, fee_type="taker"
            )

            total_fees = fees1 + fees2 + fees3
            profit = final_amount - (amount1 + total_fees)

            arbitrage_opportunity = {
                "path": ["usd", coin1, coin2, "usd"],
                "conversion_rates": [rate1, rate2, rate3],
                "fees": [fees1, fees2, fees3],
                "final_amount": final_amount,
                "profit": profit,
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
        fees1, fees2, fees3 = opportunity["fees"]
        total_profit = opportunity["profit"]
        exchange = opportunity["exchange"]

        funds = rate1
        # Initial funds in USD
        from_usd = funds

        # 1. Buy step (USD -> Coin1)
        from_amount = funds
        to_amount = funds / rate1  # Convert USD to Coin1
        to_usd = to_amount * rate1
        funds = to_usd - fees1  # Apply fees

        instructions.append(
            {
                "instruction": "buy",
                "from_exchange": exchange,
                "from_currency": "usd",
                "from_amount": from_amount,
                "to_exchange": exchange,
                "to_currency": coin1,
                "to_amount": to_amount,
                "total_fees": fees1,
                "from_usd": from_usd,
                "to_usd": funds,
            }
        )

        # Update from_usd for the next step
        from_usd = funds

        # 2. Transfer step (Coin1 -> Coin2)
        from_amount = to_amount
        to_amount = from_amount * rate2  # Convert Coin1 to Coin2
        to_usd = to_amount * rate3  # Convert to USD equivalent
        funds = to_usd - fees2  # Apply fees

        instructions.append(
            {
                "instruction": "transfer",
                "from_exchange": exchange,
                "from_currency": coin1,
                "from_amount": from_amount,
                "to_exchange": exchange,
                "to_currency": coin2,
                "to_amount": to_amount,
                "total_fees": fees2,
                "from_usd": from_usd,
                "to_usd": funds,
            }
        )

        # Update from_usd for the next step
        from_usd = funds

        # 3. Sell step (Coin2 -> USD)
        from_amount = to_amount
        to_amount = from_amount * rate3  # Convert Coin2 back to USD
        funds = to_amount - fees3  # Apply fees

        instructions.append(
            {
                "instruction": "sell",
                "from_exchange": exchange,
                "from_currency": coin2,
                "from_amount": from_amount,
                "to_exchange": exchange,
                "to_currency": "usd",
                "to_amount": to_amount,
                "total_fees": fees3,
                "from_usd": from_usd,
                "to_usd": funds,
            }
        )

        # Calculate the waterfall data
        waterfall_data = {
            "Potential Profit": total_profit + fees1 + fees2 + fees3,
            "Buy Fees": -fees1,
            "Transfer Fees": -fees2,
            "Sell Fees": -fees3,
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
        to_crypto = from_usd / opportunity["buy_price"]  # Convert USD to Crypto
        to_usd = to_crypto * opportunity["buy_price"]
        fees = opportunity["buy_price"] * opportunity["buy_taker_fee"]
        funds = to_usd - fees

        instructions.append(
            {
                "instruction": "buy",
                "from_exchange": buy_exchange,
                "from_currency": "USD",
                "from_amount": from_usd,
                "to_exchange": buy_exchange,
                "to_currency": currency_pair[0],
                "to_amount": to_crypto,
                "total_fees": opportunity["buy_taker_fee"],
                "from_usd": from_usd,
                "to_usd": funds,
            }
        )

        # Transfer Step (if applicable)
        if opportunity["network_fees_crypto"] > 0:
            from_crypto = to_crypto
            to_crypto = (
                from_crypto * (1 - opportunity["network_fees_crypto"])
                + from_crypto * opportunity["buy_withdraw_fee"]
            )
            to_crypto *= opportunity["sell_deposit_fee"]

            to_usd = to_crypto * opportunity["sell_price"]
            funds = to_usd
            fees = to_usd - from_usd

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
                    "to_usd": funds,
                }
            )

        # Sell Step
        from_crypto = to_crypto
        to_usd = from_crypto * opportunity["sell_price"]
        fees = opportunity["effective_sell_price"] * opportunity["sell_taker_fee"]
        funds = to_usd - fees

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
                "from_usd": funds,
                "to_usd": funds,
            }
        )

        return {
            "summary_header": summary_header,
            "waterfall_data": waterfall_data,
            "instructions": instructions,
        }
