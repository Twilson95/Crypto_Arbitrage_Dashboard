from src.arbitrage.ArbitrageInstructions import ArbitrageInstructions
import itertools


class ArbitrageHandler:

    @staticmethod
    def return_simple_arbitrage_instructions(
        currency, exchange_prices, currency_fees, exchange_fees, network_fees, funds
    ):
        arbitrages = ArbitrageHandler.identify_simple_arbitrage(
            currency, exchange_prices, currency_fees, exchange_fees, network_fees, funds
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
        currency,
        exchange_prices,
        currency_fees,
        exchange_fees,
        network_fees,
        input_funds,
    ):
        """
        Identifies arbitrage opportunities and returns the detailed instruction data for them.
        If no arbitrage opportunities are found, it returns the closest opportunity.
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

                # Create instruction data
                instructions = []

                # Calculate the Buy Step
                from_usd = input_funds
                buy_fees = (
                    from_usd * buy_taker_fee
                )  # Adjusted buy fees based on initial funds
                to_crypto = (from_usd - buy_fees) / buy_price  # Convert USD to Crypto
                to_usd = to_crypto * buy_price
                funds = to_usd

                instructions.append(
                    {
                        "instruction": "buy",
                        "from_exchange": buy_exchange,
                        "from_currency": "USD",
                        "from_amount": from_usd,
                        "to_exchange": buy_exchange,
                        "to_currency": currency.split("/")[0],
                        "to_amount": to_crypto,
                        "change_in_usd": to_usd - from_usd,
                        "from_usd": None,
                        "to_usd": to_usd,
                    }
                )

                # Calculate the Transfer Step (if applicable)
                if network_fee_crypto > 0:
                    from_crypto = to_crypto
                    from_usd = funds
                    withdraw_fees = (
                        from_crypto * withdraw_fee
                    )  # Adjusted withdraw fees based on initial funds
                    to_crypto = from_crypto * (1 - withdraw_fee)
                    to_crypto -= network_fee_crypto
                    deposit_fees = (
                        to_crypto * deposit_fee
                    )  # Adjusted deposit fees based on remaining crypto
                    to_crypto *= 1 - deposit_fee

                    to_usd = to_crypto * sell_price
                    fees = from_usd - to_usd

                    instructions.append(
                        {
                            "instruction": "transfer",
                            "from_exchange": buy_exchange,
                            "from_currency": currency.split("/")[0],
                            "from_amount": from_crypto,
                            "to_exchange": sell_exchange,
                            "to_currency": currency.split("/")[0],
                            "to_amount": to_crypto,
                            "change_in_usd": to_usd - from_usd,
                            "from_usd": from_usd,
                            "to_usd": to_usd,
                        }
                    )

                # Calculate the Sell Step
                from_crypto = to_crypto
                from_usd = to_usd
                sell_fees_coin = (
                    from_crypto * sell_taker_fee
                )  # Adjusted sell fees based on remaining crypto
                to_usd = (from_crypto - sell_fees_coin) * sell_price
                sell_fees = sell_fees_coin * sell_price
                funds = to_usd

                instructions.append(
                    {
                        "instruction": "sell",
                        "from_exchange": sell_exchange,
                        "from_currency": currency.split("/")[0],
                        "from_amount": from_crypto,
                        "to_exchange": sell_exchange,
                        "to_currency": "USD",
                        "to_amount": to_usd,
                        "change_in_usd": to_usd - from_usd,
                        "from_usd": from_usd,
                        "to_usd": None,
                    }
                )

                # Calculate potential arbitrage opportunity
                arbitrage_profit = funds - input_funds

                amount_bought = input_funds / buy_price
                potential_revenue = amount_bought * sell_price
                price_delta = potential_revenue - input_funds
                # price_delta = input_funds * (sell_price - buy_price) / buy_price

                # Calculate the waterfall data with fees reflecting the funds at each step
                waterfall_data = {
                    "Price Delta": price_delta,
                    "Buy Fees": -buy_fees,
                    "Withdraw Fee": -withdraw_fees if network_fee_crypto > 0 else 0,
                    "Network Fee": -network_fee_usd,
                    "Deposit Fee": -deposit_fees if network_fee_crypto > 0 else 0,
                    "Sell Fees": -sell_fees,
                }

                # Create the summary header
                summary_header = {
                    "total_profit": arbitrage_profit,
                    "currency": currency.split("/"),
                    "exchanges_used": [buy_exchange, sell_exchange],
                }

                arbitrage_data = {
                    "summary_header": summary_header,
                    "waterfall_data": waterfall_data,
                    "instructions": instructions,
                    "path": [
                        ("USD", currency.split("/")[0]),
                        (currency.split("/")[0], currency.split("/")[0]),
                        (currency.split("/")[0], "USD"),
                    ],
                }

                if arbitrage_profit > 0:
                    arbitrage_opportunities.append(arbitrage_data)
                else:
                    difference = abs(arbitrage_profit)
                    if difference < closest_difference:
                        closest_difference = difference
                        closest_opportunity = arbitrage_data

        if arbitrage_opportunities:
            return sorted(
                arbitrage_opportunities,
                key=lambda x: x["summary_header"]["total_profit"],
                reverse=True,  # Set to True for descending order, False for ascending
            )
        else:
            return [closest_opportunity]

    @staticmethod
    def create_arbitrage_simple_instructions_data(opportunity, input_funds):
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
        from_usd = input_funds
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
                "change_in_usd": to_usd - from_usd,
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
                    "change_in_usd": to_usd - from_usd,
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
                "change_in_usd": to_usd - from_usd,
                "from_usd": from_usd,
                "to_usd": None,
            }
        )

        return {
            "summary_header": summary_header,
            "waterfall_data": waterfall_data,
            "instructions": instructions,
        }

    @staticmethod
    def return_triangle_arbitrage_instructions(arbitrages):
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
            rate1 = ArbitrageHandler.calculate_conversion_rate(prices, pair1)
            rate2 = ArbitrageHandler.calculate_conversion_rate(prices, pair2)
            rate3 = ArbitrageHandler.calculate_conversion_rate(prices, pair3)

            if rate1 is None or rate2 is None or rate3 is None:
                continue

            # Start with enough USD to buy 1 unit of Coin1
            usd_start = funds
            amount1 = usd_start * rate1  # Convert USD to Coin1

            # Calculate fees and convert USD to Coin1
            fees1_usd = ArbitrageHandler.calculate_fees(
                currency_fees, pair1, usd_start, fee_type="taker"
            )
            coin1_after_fees = (usd_start - fees1_usd) * rate1

            # Calculate change in USD for the buy step
            usd_after_buy = usd_start - fees1_usd
            change_in_usd_buy = usd_after_buy - usd_start

            # Calculate fees and convert Coin1 to Coin2
            fees2_coin = ArbitrageHandler.calculate_fees(
                currency_fees, pair2, coin1_after_fees, fee_type="taker"
            )
            coin2_after_fees = (coin1_after_fees - fees2_coin) * rate2

            # Convert Coin2 amount to USD for the transfer step
            usd_after_transfer = coin2_after_fees * rate3

            # Calculate change in USD for the transfer step
            change_in_usd_transfer = usd_after_transfer - usd_after_buy

            # Calculate fees and convert Coin2 back to USD
            fees3_coin = ArbitrageHandler.calculate_fees(
                currency_fees, pair3, coin2_after_fees, fee_type="taker"
            )
            usd_end = (coin2_after_fees - fees3_coin) * rate3

            # Calculate change in USD for the sell step
            change_in_usd_sell = usd_end - usd_after_transfer

            # Calculate profit
            profit = usd_end - usd_start

            # Create instructions for each step of the arbitrage
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

            # Calculate the waterfall data
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
        potential_profit = opportunity["potential_profit_before_fees"]
        exchange = opportunity["exchange"]

        # 1. Buy step (USD -> Coin1)
        from_amount = 1 / rate1
        to_amount = (from_amount - fees1_coin) * rate1  # Convert USD to Coin1
        to_usd = from_amount - fees1_usd

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
        to_amount = (from_amount - fees2_coin) * rate2  # Convert USD to Coin1
        # to_usd = to_amount * rate3
        to_usd = from_usd - fees2_usd

        instructions.append(
            {
                "instruction": "transfer",
                "from_exchange": exchange,
                "from_currency": coin1,
                "from_amount": from_amount,
                "to_exchange": exchange,
                "to_currency": coin2,
                "to_amount": to_amount,
                "total_fees": -(to_usd - from_usd),
                "from_usd": from_usd,
                "to_usd": to_usd,
            }
        )

        # Update from_usd for the next step
        from_usd = to_usd

        from_amount = to_amount
        # to_amount = (from_amount - fees3_coin) * rate3  # Convert USD to Coin1
        to_usd = from_usd - fees3_usd
        to_amount = to_usd

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
            "path": [("USD", coin1), (coin1, coin2), (coin2, "USD")],
        }

    @staticmethod
    def identify_all_statistical_arbitrage(
        prices, spreads, currency_fees, exchange, funds, window=30
    ):
        arbitrage_instructions = []
        for pair, spread in spreads.items():
            drawing_instructions = ArbitrageHandler.get_statistical_arbitrage_trades(
                spread, window
            )
            arbitrage_instruction = ArbitrageHandler.identify_statistical_arbitrage(
                drawing_instructions, pair, currency_fees, exchange, funds, prices
            )
            arbitrage_instructions.extend(arbitrage_instruction)
        return arbitrage_instructions

    @staticmethod
    def get_statistical_arbitrage_trades(spread, window=30):
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()

        # Define thresholds for entry and exit signals
        upper_threshold = spread_mean + 2 * spread_std
        lower_threshold = spread_mean - 2 * spread_std

        # Align all data
        spread = spread.align(upper_threshold)[0]
        spread = spread.align(lower_threshold)[0]

        # Mask the initial period where thresholds are not yet available
        spread = spread[spread_mean.notna()]
        spread_mean = spread_mean[spread_mean.notna()]
        upper_threshold = upper_threshold[upper_threshold.notna()]
        lower_threshold = lower_threshold[lower_threshold.notna()]

        # Determine entry and exit points
        entry_points = []
        exit_points = []
        above_upper = False
        below_lower = False

        for i in range(1, len(spread)):
            # Check for entry points
            if spread.iloc[i] > upper_threshold.iloc[i] and not above_upper:
                entry_points.append((spread.index[i], spread.iloc[i], "short"))
                above_upper = True
                below_lower = False
            elif spread.iloc[i] < lower_threshold.iloc[i] and not below_lower:
                entry_points.append((spread.index[i], spread.iloc[i], "long"))
                below_lower = True
                above_upper = False

            # Check for exit points
            if spread.iloc[i] < spread_mean.iloc[i] and above_upper:
                exit_points.append((spread.index[i], spread.iloc[i]))
                above_upper = False
            elif spread.iloc[i] > spread_mean.iloc[i] and below_lower:
                exit_points.append((spread.index[i], spread.iloc[i]))
                below_lower = False

        drawing_instructions = {
            "entry_points": entry_points,
            "exit_points": exit_points,
            "spread_mean": spread_mean,
            "lower_threshold": lower_threshold,
            "upper_threshold": upper_threshold,
        }
        return drawing_instructions

    @staticmethod
    def identify_statistical_arbitrage(
        drawing_instructions, pairs, currency_fees, exchange, funds, price_df
    ):
        entry_points = drawing_instructions["entry_points"]
        exit_points = drawing_instructions["exit_points"]

        # List to store each arbitrage opportunity separately
        arbitrage_opportunities = []

        # Extract coins from the pair
        pair1, pair2 = pairs
        coin1 = pair1.split("/")[0]
        coin2 = pair2.split("/")[0]

        # Iterate over each entry and exit pair
        for entry, exit in zip(entry_points, exit_points):
            entry_time, entry_spread, entry_type = entry
            exit_time, exit_spread = exit

            # Look up the actual prices for the entry and exit times
            entry_price_coin1 = price_df.loc[entry_time, pair1]
            entry_price_coin2 = price_df.loc[entry_time, pair2]
            exit_price_coin1 = price_df.loc[exit_time, pair1]
            exit_price_coin2 = price_df.loc[exit_time, pair2]

            # Define initial amounts (assuming a starting capital in USD)
            usd_start = funds  # Example starting capital in USD

            # Get fees from currency_fees
            buy_fee_coin1 = currency_fees[pair1]["taker"]
            sell_fee_coin2 = currency_fees[pair2]["taker"]

            # Initialize variables for summary and waterfall data
            total_profit = 0
            potential_profit = 0
            fees1_usd, fees2_usd, fees3_usd = 0, 0, 0

            # Entry Instructions
            entry_instructions = []

            if entry_type == "short":
                # Short the spread: Sell short Coin1, Buy Coin2
                amount_sell_coin1 = usd_start / entry_price_coin1
                amount_buy_coin2 = usd_start / entry_price_coin2

                # Calculate fees for short entry
                fees_sell_coin1 = amount_sell_coin1 * sell_fee_coin2
                fees_buy_coin2 = amount_buy_coin2 * buy_fee_coin1

                net_sell_coin1 = amount_sell_coin1 - fees_sell_coin1
                net_buy_coin2 = amount_buy_coin2 - fees_buy_coin2

                entry_instructions = [
                    {
                        "instruction": "sell short",
                        "from_exchange": exchange,
                        "from_currency": coin1,
                        "from_amount": amount_sell_coin1,
                        "to_exchange": exchange,
                        "to_currency": "USD",
                        "to_amount": usd_start - fees_sell_coin1,
                        "change_in_usd": -fees_sell_coin1,
                        "from_usd": amount_sell_coin1 / entry_price_coin1,
                        "to_usd": None,
                    },
                    {
                        "instruction": "buy",
                        "from_exchange": exchange,
                        "from_currency": "USD",
                        "from_amount": usd_start,
                        "to_exchange": exchange,
                        "to_currency": coin2,
                        "to_amount": net_buy_coin2,
                        "change_in_usd": -fees_buy_coin2,
                        "from_usd": None,
                        "to_usd": usd_start - fees_buy_coin2,
                    },
                ]

                # Update fees1_usd for waterfall data
                fees1_usd = fees_sell_coin1 + fees_buy_coin2

            else:  # entry_type == "long"
                # Long the spread: Buy Coin1, Sell short Coin2
                amount_buy_coin1 = usd_start / entry_price_coin1
                amount_sell_coin2 = usd_start / entry_price_coin2

                # Calculate fees for long entry
                fees_buy_coin1 = amount_buy_coin1 * buy_fee_coin1
                fees_sell_coin2 = amount_sell_coin2 * sell_fee_coin2

                net_buy_coin1 = amount_buy_coin1 - fees_buy_coin1
                net_sell_coin2 = amount_sell_coin2 - fees_sell_coin2

                entry_instructions = [
                    {
                        "instruction": "buy",
                        "from_exchange": exchange,
                        "from_currency": "USD",
                        "from_amount": usd_start,
                        "to_exchange": exchange,
                        "to_currency": coin1,
                        "to_amount": net_buy_coin1,
                        "change_in_usd": -fees_buy_coin1,
                        "from_usd": None,
                        "to_usd": usd_start - fees_buy_coin1,
                    },
                    {
                        "instruction": "sell short",
                        "from_exchange": exchange,
                        "from_currency": coin2,
                        "from_amount": amount_sell_coin2,
                        "to_exchange": exchange,
                        "to_currency": "USD",
                        "to_amount": usd_start - fees_sell_coin2,
                        "change_in_usd": -fees_sell_coin2,
                        "from_usd": usd_start,
                        "to_usd": None,
                    },
                ]

                # Update fees1_usd for waterfall data
                fees1_usd = fees_buy_coin1 + fees_sell_coin2

            # Add a "wait" instruction to separate entry and exit
            entry_instructions.append(
                {
                    "instruction": "wait",
                    "details": "Hold positions until exit signal",
                }
            )

            # Exit Instructions
            exit_instructions = []

            if entry_type == "short":
                # Closing short position
                close_buy_coin1 = net_sell_coin1 * exit_price_coin1
                close_sell_coin2 = net_buy_coin2 * exit_price_coin2

                fees_close_buy_coin1 = close_buy_coin1 * buy_fee_coin1
                fees_close_sell_coin2 = close_sell_coin2 * sell_fee_coin2

                usd_after_close = (
                    net_sell_coin1 - fees_close_buy_coin1
                ) * exit_price_coin1 + (
                    net_buy_coin2 - fees_close_sell_coin2
                ) * exit_price_coin2

                exit_instructions = [
                    {
                        "instruction": "buy to cover",
                        "from_exchange": exchange,
                        "from_currency": coin1,
                        "from_amount": close_buy_coin1,
                        "to_exchange": exchange,
                        "to_currency": "USD",
                        "to_amount": usd_after_close,
                        "change_in_usd": -fees_close_buy_coin1,
                        "from_usd": close_buy_coin1 * exit_price_coin1,
                        "to_usd": None,
                    },
                    {
                        "instruction": "sell",
                        "from_exchange": exchange,
                        "from_currency": coin2,
                        "from_amount": close_sell_coin2,
                        "to_exchange": exchange,
                        "to_currency": "USD",
                        "to_amount": usd_after_close,
                        "change_in_usd": -fees_close_sell_coin2,
                        "from_usd": close_sell_coin2 * exit_price_coin2,
                        "to_usd": None,
                    },
                ]

                # Update fees2_usd for waterfall data
                fees2_usd = fees_close_buy_coin1 + fees_close_sell_coin2

            else:  # entry_type == "long"
                # Closing long position
                close_sell_coin1 = net_buy_coin1 * exit_price_coin1
                close_buy_coin2 = net_sell_coin2 * exit_price_coin2

                fees_close_sell_coin1 = close_sell_coin1 * sell_fee_coin2
                fees_close_buy_coin2 = close_buy_coin2 * buy_fee_coin1

                usd_after_close = (
                    net_buy_coin1 - fees_close_sell_coin1
                ) * exit_price_coin1 + (
                    net_sell_coin2 - fees_close_buy_coin2
                ) * exit_price_coin2

                exit_instructions = [
                    {
                        "instruction": "sell",
                        "from_exchange": exchange,
                        "from_currency": coin1,
                        "from_amount": close_sell_coin1,
                        "to_exchange": exchange,
                        "to_currency": "USD",
                        "to_amount": usd_after_close,
                        "change_in_usd": -fees_close_sell_coin1,
                        "from_usd": close_sell_coin1 * exit_price_coin1,
                        "to_usd": None,
                    },
                    {
                        "instruction": "buy to cover",
                        "from_exchange": exchange,
                        "from_currency": coin2,
                        "from_amount": close_buy_coin2,
                        "to_exchange": exchange,
                        "to_currency": "USD",
                        "to_amount": usd_after_close,
                        "change_in_usd": -fees_close_buy_coin2,
                        "from_usd": close_buy_coin2 * exit_price_coin2,
                        "to_usd": None,
                    },
                ]

                # Update fees2_usd for waterfall data
                fees2_usd = fees_close_sell_coin1 + fees_close_buy_coin2

            # Calculate total profit and potential profit
            total_profit = usd_after_close - usd_start
            potential_profit = (
                usd_start * (entry_price_coin1 - exit_price_coin1) / entry_price_coin1
            )

            # Waterfall Plot Data
            waterfall_data = {
                "Potential Profit": potential_profit,
                "Buy Fees": -fees1_usd,
                "Transfer Fees": 0,  # No transfer fee in this context
                "Sell Fees": -fees2_usd,
            }

            # Create the summary header
            summary_header = {
                "total_profit": total_profit,
                "coins_used": [pair1, pair2],
                "exchanges_used": exchange,
            }

            # Combine all data into a single dictionary for this opportunity
            arbitrage_opportunity = {
                "summary_header": summary_header,
                "waterfall_data": waterfall_data,
                "instructions": entry_instructions + exit_instructions,
                "path": [("USD", pair1), (pair1, pair2), (pair2, "USD")],
            }

            # Append the arbitrage opportunity to the list
            arbitrage_opportunities.append(arbitrage_opportunity)

        return arbitrage_opportunities

    @staticmethod
    def return_statistical_arbitrage_instructions(arbitrages):
        instruction_diagrams = []

        for arbitrage in arbitrages:
            arbitrage_instructions = ArbitrageInstructions(arbitrage)
            instructions = arbitrage_instructions.return_statistical_arbitrage_panels()
            instruction_diagrams.append(instructions)
        return instruction_diagrams
