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
        if not arbitrages:
            return {}
        # print("arbitrages", arbitrages)
        instruction_diagrams = []
        for arbitrage in arbitrages:
            arbitrage_instructions = ArbitrageInstructions(arbitrage)
            instructions = arbitrage_instructions.return_single_arbitrage_panels()
            instruction_diagrams.append(instructions)
        return instruction_diagrams

    @staticmethod
    def calculate_buy_step(
        buy_exchange, currency, buy_price, input_funds, buy_taker_fee
    ):
        # Calculate the amount of crypto bought
        buy_fees = input_funds * buy_taker_fee
        to_crypto = (input_funds - buy_fees) / buy_price
        to_usd = to_crypto * buy_price

        # Return the result in a structured format
        return {
            "instruction": "buy",
            "from_exchange": buy_exchange,
            "from_currency": "USD",
            "from_amount": input_funds,
            "to_exchange": buy_exchange,
            "to_currency": currency.split("/")[0],
            "to_amount": to_crypto,
            "change_in_usd": to_usd - input_funds,
            "from_usd": None,
            "to_usd": to_usd,
            "buy_fees": buy_fees,
        }

    @staticmethod
    def calculate_transfer_step(
        from_exchange,
        to_exchange,
        from_crypto,
        from_usd,
        sell_price,
        withdraw_fee,
        deposit_fee,
        network_fee_crypto,
        currency,
    ):
        withdraw_fees = from_crypto * withdraw_fee
        to_crypto = from_crypto
        to_crypto -= withdraw_fees
        to_crypto -= network_fee_crypto
        deposit_fees = to_crypto * deposit_fee
        to_crypto -= deposit_fees

        to_usd = to_crypto * sell_price

        to_crypto_without_network_fee = from_crypto
        to_crypto_without_network_fee -= withdraw_fees
        deposit_fees = to_crypto_without_network_fee * deposit_fee
        to_crypto_without_network_fee -= deposit_fees
        network_fee_usd = abs(to_crypto_without_network_fee - to_crypto) * sell_price

        return {
            "instruction": "transfer",
            "from_exchange": from_exchange,
            "from_currency": currency.split("/")[0],
            "from_amount": from_crypto,
            "to_exchange": to_exchange,
            "to_currency": currency.split("/")[0],
            "to_amount": to_crypto,
            "change_in_usd": to_usd - from_usd,
            "from_usd": from_usd,
            "to_usd": to_usd,
            "fees": {
                "withdraw_fees": withdraw_fees * sell_price,
                "network_fees": network_fee_usd,
                "deposit_fees": deposit_fees * sell_price,
            },
        }

    @staticmethod
    def calculate_sell_step(
        sell_exchange, currency, from_crypto, from_usd, sell_price, sell_taker_fee
    ):

        # Calculate the amount of USD received after selling the crypto
        sell_fees = from_crypto * sell_taker_fee
        to_usd = (from_crypto - sell_fees) * sell_price

        # Return the result in a structured format
        return {
            "instruction": "sell",
            "from_exchange": sell_exchange,
            "from_currency": currency.split("/")[0],
            "from_amount": from_crypto,
            "to_exchange": sell_exchange,
            "to_currency": "USD",
            "to_amount": to_usd,
            "change_in_usd": to_usd - from_usd,
            "from_usd": None,
            "to_usd": None,
            "sell_fees": sell_fees * sell_price,
        }

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
            # if len(close_price_buy) == 0:
            #     print(buy_exchange, "has no prices")
            #     continue
            buy_price = close_price_buy[-1]

            for sell_exchange, prices_sell in exchange_prices.items():
                if sell_exchange == buy_exchange:
                    continue

                sell_taker_fee = currency_fees.get(sell_exchange, {}).get("taker", 0)
                deposit_fee = exchange_fees.get(sell_exchange, {}).get("deposit", 0)

                close_price_sell = prices_sell.close
                # if len(close_price_sell) == 0:
                #     print(sell_exchange, "has no prices")
                #     continue

                sell_price = close_price_sell[-1]
                network_fee_usd = network_fees * sell_price

                # Calculate the Buy Step
                buy_instruction = ArbitrageHandler.calculate_buy_step(
                    buy_exchange, currency, buy_price, input_funds, buy_taker_fee
                )
                current_usd = buy_instruction["to_usd"]
                current_amount = buy_instruction["to_amount"]

                # transfer instructions
                transfer_instruction = ArbitrageHandler.calculate_transfer_step(
                    buy_exchange,
                    sell_exchange,
                    current_amount,
                    current_usd,
                    sell_price,
                    withdraw_fee,
                    deposit_fee,
                    network_fees,
                    currency,
                )
                current_usd = transfer_instruction["to_usd"]
                current_amount = transfer_instruction["to_amount"]

                # Calculate the Sell Step
                sell_instruction = ArbitrageHandler.calculate_sell_step(
                    sell_exchange,
                    currency,
                    current_amount,
                    current_usd,
                    sell_price,
                    sell_taker_fee,
                )
                current_usd = sell_instruction["to_amount"]

                # Calculate potential arbitrage opportunity
                arbitrage_profit = current_usd - input_funds

                amount_bought = input_funds / buy_price
                potential_revenue = amount_bought * sell_price
                price_delta = potential_revenue - input_funds

                # Calculate the waterfall data with fees reflecting the funds at each step
                waterfall_data = {
                    "Price Delta": price_delta,
                    "Buy Fees": -buy_instruction["buy_fees"],
                    "Withdraw Fee": -transfer_instruction["fees"].get(
                        "withdraw_fees", 0
                    ),
                    "Network Fee": -transfer_instruction["fees"].get("network_fees", 0),
                    "Deposit Fee": -transfer_instruction["fees"].get("deposit_fees", 0),
                    "Sell Fees": -sell_instruction["sell_fees"],
                }

                # Create the summary header
                summary_header = {
                    "total_profit": arbitrage_profit,
                    "currency": currency.split("/"),
                    "exchanges_used": [buy_exchange, sell_exchange],
                }

                instructions = [buy_instruction, transfer_instruction, sell_instruction]

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
        elif closest_opportunity is not None:
            return [closest_opportunity]
        else:
            return None

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
        prices, pair, spread_details, currency_fees, exchange, funds, window=30
    ):
        spread = spread_details["spread"]
        hedge_ratio = spread_details["hedge_ratio"]
        drawing_instructions = ArbitrageHandler.get_statistical_arbitrage_trades(
            spread, window
        )
        arbitrage_instructions = ArbitrageHandler.identify_statistical_arbitrage(
            drawing_instructions,
            pair,
            currency_fees,
            exchange,
            funds,
            prices,
            hedge_ratio,
        )
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

        while len(exit_points) < len(entry_points):
            exit_points.append((None, None))

        drawing_instructions = {
            "entry_points": entry_points,
            "exit_points": exit_points,
            "spread_mean": spread_mean,
            "lower_threshold": lower_threshold,
            "upper_threshold": upper_threshold,
        }
        return drawing_instructions

    # @staticmethod
    # def identify_statistical_arbitrage(
    #     drawing_instructions,
    #     pairs,
    #     currency_fees,
    #     exchange,
    #     funds,
    #     price_df,
    #     hedge_ratio,
    # ):
    #     entry_points = drawing_instructions["entry_points"]
    #     exit_points = drawing_instructions["exit_points"]
    #
    #     # List to store each arbitrage opportunity separately
    #     arbitrage_opportunities = []
    #
    #     # Extract coins from the pair
    #     pair1, pair2 = pairs
    #     coin1 = pair1.split("/")[0]
    #     coin2 = pair2.split("/")[0]
    #
    #     # Iterate over each entry and exit pair
    #     for entry, exit in zip(entry_points, exit_points):
    #         entry_time, entry_spread, entry_type = entry
    #         exit_time, exit_spread = exit
    #
    #         # Look up the actual prices for the entry and exit times
    #         entry_price_coin1 = price_df.loc[entry_time, pair1]
    #         entry_price_coin2 = price_df.loc[entry_time, pair2]
    #         exit_price_coin1 = price_df.loc[exit_time, pair1]
    #         exit_price_coin2 = price_df.loc[exit_time, pair2]
    #
    #         # Define initial amounts (assuming a starting capital in USD)
    #         usd_start = funds  # Example starting capital in USD
    #
    #         # Get fees from currency_fees
    #         buy_fee_coin1 = currency_fees[pair1]["taker"]
    #         sell_fee_coin2 = currency_fees[pair2]["taker"]
    #
    #         if entry_type == "short":
    #             # Short the spread: Sell short Coin1, Buy Coin2
    #             amount_buy_coin2 = usd_start / entry_price_coin2
    #             amount_sell_coin1 = amount_buy_coin2 / hedge_ratio
    #
    #             # Calculate fees for short entry
    #             fees_sell_coin1 = amount_sell_coin1 * sell_fee_coin2
    #             fees_buy_coin2 = amount_buy_coin2 * buy_fee_coin1
    #
    #             net_sell_coin1 = amount_sell_coin1 - fees_sell_coin1
    #             net_buy_coin2 = amount_buy_coin2 - fees_buy_coin2
    #
    #             # Calculate the USD changes for each trade
    #             change_in_usd_buy = -fees_buy_coin2
    #             change_in_usd_sell_short = -fees_sell_coin1
    #
    #             entry_instructions = [
    #                 {
    #                     "instruction": "buy",
    #                     "from_exchange": exchange,
    #                     "from_currency": "USD",
    #                     "from_amount": usd_start,
    #                     "to_exchange": exchange,
    #                     "to_currency": coin2,
    #                     "to_amount": net_buy_coin2,
    #                     "change_in_usd": change_in_usd_buy,
    #                     "from_usd": None,
    #                     "to_usd": usd_start + change_in_usd_buy,
    #                 },
    #                 {
    #                     "instruction": "sell short",
    #                     "from_exchange": exchange,
    #                     "from_currency": coin1,
    #                     "from_amount": amount_sell_coin1,
    #                     "to_exchange": exchange,
    #                     "to_currency": "USD",
    #                     "to_amount": amount_sell_coin1 * entry_price_coin1
    #                     + change_in_usd_sell_short,
    #                     "change_in_usd": change_in_usd_sell_short,
    #                     "from_usd": amount_sell_coin1 * entry_price_coin1,
    #                     "to_usd": None,
    #                 },
    #             ]
    #
    #             # Update fees1_usd for waterfall data
    #             fees1_usd = fees_sell_coin1 + fees_buy_coin2
    #
    #         else:  # entry_type == "long"
    #             # Long the spread: Buy Coin1, Sell short Coin2
    #             amount_buy_coin1 = usd_start / entry_price_coin1
    #             # Use hedge ratio to calculate the amount of Coin2 to sell short
    #             amount_sell_coin2 = amount_buy_coin1 * hedge_ratio
    #
    #             # Calculate fees for long entry
    #             fees_buy_coin1 = amount_buy_coin1 * buy_fee_coin1
    #             fees_sell_coin2 = amount_sell_coin2 * sell_fee_coin2
    #
    #             net_buy_coin1 = amount_buy_coin1 - fees_buy_coin1
    #             net_sell_coin2 = amount_sell_coin2 - fees_sell_coin2
    #
    #             # Calculate the USD changes for each trade
    #             change_in_usd_buy = -fees_buy_coin1
    #             change_in_usd_sell_short = -fees_sell_coin2
    #
    #             entry_instructions = [
    #                 {
    #                     "instruction": "buy",
    #                     "from_exchange": exchange,
    #                     "from_currency": "USD",
    #                     "from_amount": usd_start,
    #                     "to_exchange": exchange,
    #                     "to_currency": coin1,
    #                     "to_amount": net_buy_coin1,
    #                     "change_in_usd": change_in_usd_buy,
    #                     "from_usd": None,
    #                     "to_usd": usd_start + change_in_usd_buy,
    #                 },
    #                 {
    #                     "instruction": "sell short",
    #                     "from_exchange": exchange,
    #                     "from_currency": coin2,
    #                     "from_amount": amount_sell_coin2,
    #                     "to_exchange": exchange,
    #                     "to_currency": "USD",
    #                     "to_amount": usd_start + change_in_usd_sell_short,
    #                     "change_in_usd": change_in_usd_sell_short,
    #                     "from_usd": amount_sell_coin2 * entry_price_coin2,
    #                     "to_usd": None,
    #                 },
    #             ]
    #
    #             # Update fees1_usd for waterfall data
    #             fees1_usd = fees_buy_coin1 + fees_sell_coin2
    #
    #         # Add a "wait" instruction to separate entry and exit
    #         entry_instructions.append(
    #             {
    #                 "details": "Hold positions until exit signal",
    #             }
    #         )
    #
    #         if entry_type == "short":
    #             # Closing short position: Buy back Coin1, Sell Coin2
    #             amount_buy_to_cover_coin1 = net_sell_coin1
    #             amount_sell_coin2 = net_buy_coin2
    #
    #             # Calculate fees for closing the short position
    #             fees_buy_to_cover_coin1 = amount_buy_to_cover_coin1 * buy_fee_coin1
    #             fees_sell_coin2 = amount_sell_coin2 * sell_fee_coin2
    #
    #             # Calculate the USD changes for each trade
    #             usd_from_sell_coin2 = (
    #                 amount_sell_coin2 * exit_price_coin2
    #             ) - fees_sell_coin2
    #             usd_for_buy_to_cover_coin1 = (
    #                 amount_buy_to_cover_coin1 * exit_price_coin1
    #             ) + fees_buy_to_cover_coin1
    #
    #             change_in_usd_sell = -fees_sell_coin2
    #             change_in_usd_close_short = -fees_buy_to_cover_coin1
    #
    #             exit_instructions = [
    #                 {
    #                     "instruction": "sell",
    #                     "from_exchange": exchange,
    #                     "from_currency": coin2,
    #                     "from_amount": amount_sell_coin2,
    #                     "to_exchange": exchange,
    #                     "to_currency": "USD",
    #                     "to_amount": usd_from_sell_coin2,
    #                     "change_in_usd": change_in_usd_sell,
    #                     "from_usd": amount_sell_coin2 * exit_price_coin2,
    #                     "to_usd": None,
    #                 },
    #                 {
    #                     "instruction": "buy to cover",
    #                     "from_exchange": exchange,
    #                     "from_currency": "USD",
    #                     "from_amount": usd_for_buy_to_cover_coin1,
    #                     "to_exchange": exchange,
    #                     "to_currency": coin1,
    #                     "to_amount": amount_buy_to_cover_coin1,
    #                     "change_in_usd": change_in_usd_close_short,
    #                     "from_usd": None,
    #                     "to_usd": amount_buy_to_cover_coin1 * exit_price_coin1,
    #                 },
    #             ]
    #             usd_after_close = (
    #                 usd_from_sell_coin2 + amount_buy_to_cover_coin1 * exit_price_coin1
    #             )
    #
    #             # Update fees2_usd for waterfall data
    #             fees2_usd = fees_buy_to_cover_coin1 + fees_sell_coin2
    #
    #         else:  # entry_type == "long"
    #             # Closing long position: Sell Coin1, Buy back Coin2
    #             amount_sell_coin1 = net_buy_coin1
    #             amount_buy_to_cover_coin2 = net_sell_coin2
    #
    #             # Calculate fees for closing the long position
    #             fees_sell_coin1 = amount_sell_coin1 * sell_fee_coin2
    #             fees_buy_to_cover_coin2 = amount_buy_to_cover_coin2 * buy_fee_coin1
    #
    #             # Calculate the USD changes for each trade
    #             usd_from_sell_coin1 = (
    #                 amount_sell_coin1 * exit_price_coin1
    #             ) - fees_sell_coin1
    #             usd_for_buy_to_cover_coin2 = (
    #                 amount_buy_to_cover_coin2 * exit_price_coin2
    #             ) + fees_buy_to_cover_coin2
    #
    #             change_in_usd_sell = -fees_sell_coin1
    #             change_in_usd_close_short = -fees_buy_to_cover_coin2
    #
    #             exit_instructions = [
    #                 {
    #                     "instruction": "sell",
    #                     "from_exchange": exchange,
    #                     "from_currency": coin1,
    #                     "from_amount": amount_sell_coin1,
    #                     "to_exchange": exchange,
    #                     "to_currency": "USD",
    #                     "to_amount": usd_from_sell_coin1,
    #                     "change_in_usd": change_in_usd_sell,
    #                     "from_usd": amount_sell_coin1 * exit_price_coin1,
    #                     "to_usd": None,
    #                 },
    #                 {
    #                     "instruction": "buy to cover",
    #                     "from_exchange": exchange,
    #                     "from_currency": "USD",
    #                     "from_amount": usd_for_buy_to_cover_coin2,
    #                     "to_exchange": exchange,
    #                     "to_currency": coin2,
    #                     "to_amount": amount_buy_to_cover_coin2,
    #                     "change_in_usd": change_in_usd_close_short,
    #                     "from_usd": None,
    #                     "to_usd": amount_buy_to_cover_coin2 * exit_price_coin2,
    #                 },
    #             ]
    #             usd_after_close = (
    #                 amount_buy_to_cover_coin2 * exit_price_coin2 + usd_from_sell_coin1
    #             )
    #
    #             # Update fees2_usd for waterfall data
    #             fees2_usd = fees_sell_coin1 + fees_buy_to_cover_coin2
    #
    #         # Calculate total profit and potential profit
    #         total_profit = usd_after_close - usd_start
    #
    #         if entry_type == "short":
    #             potential_profit = (usd_start / entry_price_coin1) * (
    #                 entry_price_coin1 - exit_price_coin1
    #             ) + (usd_start / entry_price_coin2) * (
    #                 exit_price_coin2 - entry_price_coin2
    #             )
    #         else:
    #             potential_profit = (usd_start / entry_price_coin1) * (
    #                 exit_price_coin1 - entry_price_coin1
    #             ) + (usd_start / entry_price_coin2) * (
    #                 entry_price_coin2 - exit_price_coin2
    #             )
    #
    #         # Waterfall Plot Data
    #         waterfall_data = {
    #             "Potential Profit": abs(potential_profit),
    #             "Buy Fees": -fees1_usd,
    #             "Sell Fees": -fees2_usd,
    #         }
    #
    #         # Create the summary header
    #         summary_header = {
    #             "total_profit": total_profit,
    #             "coins_used": [pair1, pair2],
    #             "exchanges_used": exchange,
    #         }
    #
    #         # Combine all data into a single dictionary for this opportunity
    #         arbitrage_opportunity = {
    #             "summary_header": summary_header,
    #             "waterfall_data": waterfall_data,
    #             "instructions": entry_instructions + exit_instructions,
    #             "path": [("USD", pair1), (pair1, pair2), (pair2, "USD")],
    #         }
    #
    #         # Append the arbitrage opportunity to the list
    #         arbitrage_opportunities.append(arbitrage_opportunity)
    #
    #     return arbitrage_opportunities

    @staticmethod
    def return_statistical_arbitrage_instructions(arbitrages):
        instruction_diagrams = []

        for arbitrage in arbitrages:
            arbitrage_instructions = ArbitrageInstructions(arbitrage)
            instructions = arbitrage_instructions.return_statistical_arbitrage_panels()
            instruction_diagrams.append(instructions)
        return instruction_diagrams

    @staticmethod
    def handle_buy_and_sell_coin(
        usd_start, entry_price, exit_price, coin, currency_fee, exchange
    ):
        # Entry: Buy Coin1
        amount_buy_coin = usd_start / entry_price
        fees_buy_coin = amount_buy_coin * currency_fee
        net_buy_coin = amount_buy_coin - fees_buy_coin

        change_in_usd_buy = -fees_buy_coin  # Change in USD is just the fees paid
        usd_after_buy = usd_start + change_in_usd_buy  # Adjusted USD after buy

        buy_instruction = {
            "instruction": "buy",
            "from_exchange": exchange,
            "from_currency": "USD",
            "from_amount": usd_start,
            "to_exchange": exchange,
            "to_currency": coin,
            "to_amount": net_buy_coin,
            "change_in_usd": change_in_usd_buy,
            "from_usd": None,
            "to_usd": usd_after_buy,
        }

        buy_fee_usd = fees_buy_coin * entry_price  # Convert buy fees to USD

        if exit_price is None:
            return (
                [buy_instruction],
                usd_after_buy,
                None,
                change_in_usd_buy,
                amount_buy_coin,
                None,
            )

        # Exit: Sell Coin1
        amount_sell_coin = net_buy_coin
        fees_sell_coin = amount_sell_coin * currency_fee
        usd_after_sell = (amount_sell_coin - fees_sell_coin) * exit_price

        change_in_usd_sell = usd_after_sell - usd_after_buy  # Adjust USD after sell

        sell_instruction = {
            "instruction": "sell",
            "from_exchange": exchange,
            "from_currency": coin,
            "from_amount": amount_sell_coin,
            "to_exchange": exchange,
            "to_currency": "USD",
            "to_amount": usd_after_sell,
            "change_in_usd": change_in_usd_sell,
            "from_usd": usd_after_buy,
            "to_usd": None,
        }

        # Calculate the total fees (fees in USD)
        sell_fee_usd = fees_sell_coin * exit_price  # Convert sell fees to USD
        total_fees_usd = buy_fee_usd + sell_fee_usd  # Sum of all fees
        profit = usd_after_sell - usd_start

        # Return instructions and final USD amount
        return (
            [buy_instruction, sell_instruction],
            usd_after_buy,
            usd_after_sell,
            total_fees_usd,  # Corrected total fees in USD
            amount_buy_coin,
            profit,
        )

    @staticmethod
    def handle_short_and_cover_coin(
        entry_price,
        exit_price,
        coin,
        hedge_ratio,
        currency_fee,
        exchange,
        amount_coin1,
    ):
        # Entry: Short Coin (sell short) adjusted by hedge ratio
        # Use hedge ratio to determine the amount of Coin to sell short
        amount_sell_coin = amount_coin1 * hedge_ratio
        fees_sell_coin = amount_sell_coin * currency_fee  # Fees for shorting the coin
        net_sell_coin = amount_sell_coin - fees_sell_coin  # Net amount after fees

        # Calculate the USD equivalent after the short
        usd_after_sell_short = (
            amount_sell_coin * entry_price
        )  # Amount you receive from selling short
        change_in_usd_sell_short = -fees_sell_coin * entry_price  # Deduct fees from USD

        sell_short_instruction = {
            "instruction": "sell short",
            "from_exchange": exchange,
            "from_currency": coin,
            "from_amount": amount_sell_coin,
            "to_exchange": exchange,
            "to_currency": "USD",
            "to_amount": usd_after_sell_short,
            "change_in_usd": change_in_usd_sell_short,
            "from_usd": amount_sell_coin * entry_price,
            "to_usd": None,  # Adjust for fee impact
        }

        if exit_price is None:
            return (
                [sell_short_instruction],
                usd_after_sell_short,
                None,
                change_in_usd_sell_short,
                None,
            )

        # Exit: Buy to cover Coin (buy back the coin to cover the short)
        # Use the hedge ratio to ensure we cover the correct amount based on the original short position
        amount_buy_to_cover_coin = (
            net_sell_coin  # Amount to buy back is the net amount after fees
        )
        fees_buy_to_cover_coin = (
            amount_buy_to_cover_coin * currency_fee
        )  # Fees for buying to cover
        usd_after_cover = (
            amount_buy_to_cover_coin - fees_buy_to_cover_coin
        ) * exit_price  # USD spent to cover the short

        change_in_usd_cover = (
            usd_after_sell_short - usd_after_cover
        )  # Difference in USD after covering
        to_usd = usd_after_sell_short + change_in_usd_cover

        buy_to_cover_instruction = {
            "instruction": "buy to cover",
            "from_exchange": exchange,
            "from_currency": "USD",
            "from_amount": usd_after_sell_short,  # The USD you use to buy back the coin
            "to_exchange": exchange,
            "to_currency": coin,
            "to_amount": amount_buy_to_cover_coin,
            "change_in_usd": change_in_usd_cover,
            "from_usd": None,
            "to_usd": to_usd,
        }

        # Total fees are the sum of shorting and covering the coin
        total_fees_usd = (
            fees_sell_coin * entry_price + fees_buy_to_cover_coin * exit_price
        )

        profit = to_usd - amount_sell_coin * entry_price

        # Return instructions and final USD amount
        return (
            [sell_short_instruction, buy_to_cover_instruction],
            usd_after_sell_short,
            to_usd,
            total_fees_usd,
            profit,
        )

    @staticmethod
    def identify_statistical_arbitrage(
        drawing_instructions,
        pairs,
        currency_fees,
        exchange,
        funds,
        price_df,
        hedge_ratio,
    ):
        entry_points = drawing_instructions["entry_points"]
        exit_points = drawing_instructions["exit_points"]

        # List to store each arbitrage opportunity separately
        arbitrage_opportunities = []

        # Iterate over each entry and exit pair
        for entry, exit in zip(entry_points, exit_points):
            arbitrage_opportunity = ArbitrageHandler.statistical_arbitrage_iteration(
                entry,
                exit,
                pairs,
                currency_fees,
                price_df,
                funds,
                hedge_ratio,
                exchange,
            )
            # Append the arbitrage opportunity to the list
            arbitrage_opportunities.append(arbitrage_opportunity)

        return arbitrage_opportunities

    @staticmethod
    def create_summary(
        total_profit,
        potential_profit,
        coin1_fees_usd,
        coin2_fees_usd,
        pair1,
        pair2,
        exchange,
    ):
        waterfall_data = {
            "Potential Profit": potential_profit,
            "Buy Fees": -coin1_fees_usd,
            "Sell Fees": -coin2_fees_usd,
        }

        summary_header = {
            "total_profit": total_profit,
            "coins_used": [pair1.split("/")[0], pair2.split("/")[0]],
            "exchanges_used": exchange,
        }

        return waterfall_data, summary_header

    @staticmethod
    def calculate_potential_profit(
        usd_start,
        entry_price_coin1,
        entry_price_coin2,
        exit_price_coin1,
        exit_price_coin2,
        hedge_ratio,
    ):
        if exit_price_coin1 is None:
            return 0

        potential_profit = (usd_start / entry_price_coin1) * (
            exit_price_coin1 - entry_price_coin1
        ) + ((usd_start / entry_price_coin1) * hedge_ratio) * (
            entry_price_coin2 - exit_price_coin2
        )

        return potential_profit  # You can add further logic to calculate the actual total profit

    @staticmethod
    def statistical_arbitrage_iteration(
        entry,
        exit,
        pairs,
        currency_fees,
        price_df,
        usd_start,
        hedge_ratio,
        exchange,
    ):
        # print(pairs, hedge_ratio)
        pair1, pair2 = pairs
        coin1 = pair1.split("/")[0]
        coin2 = pair2.split("/")[0]
        coin1_fee = currency_fees[pair1]["taker"]
        coin2_fee = currency_fees[pair1]["taker"]

        entry_time, entry_spread, entry_type = entry
        exit_time, exit_spread = exit

        # Look up the actual prices for the entry and exit times
        entry_price_coin1 = price_df.loc[entry_time, pair1]
        entry_price_coin2 = price_df.loc[entry_time, pair2]

        if exit_time is not None:
            exit_price_coin1 = price_df.loc[exit_time, pair1]
            exit_price_coin2 = price_df.loc[exit_time, pair2]
        else:
            exit_price_coin1 = None
            exit_price_coin2 = None

        if hedge_ratio < 0:
            # hedge_ratio = 1 / hedge_ratio
            print(hedge_ratio)
            pass

        if entry_type == "short":
            hedge_ratio = 1 / hedge_ratio
            coin1, coin2 = coin2, coin1
            pair1, pair2 = pair2, pair1
            coin1_fee, coin2_fee = coin2_fee, coin1_fee
            entry_price_coin1, entry_price_coin2 = (
                entry_price_coin2,
                entry_price_coin1,
            )
            exit_price_coin1, exit_price_coin2 = exit_price_coin2, exit_price_coin1

        hedge_ratio = abs(hedge_ratio)

        # Handle buying and selling of the first coin
        (
            instructions_coin1,
            usd_after_buy_coin1,
            usd_after_sell_coin1,
            coin1_fees_usd,
            amount_buy_coin1,
            coin1_profit,
        ) = ArbitrageHandler.handle_buy_and_sell_coin(
            usd_start,
            entry_price_coin1,
            exit_price_coin1,
            coin1,
            coin1_fee,
            exchange,
        )

        wait_instruction = {
            "details": "Hold positions until exit signal",
        }

        # Handle shorting and covering of the second coin
        (
            instructions_coin2,
            usd_after_sell_short,
            usd_after_buy_cover,
            coin2_fees_usd,
            coin2_profit,
        ) = ArbitrageHandler.handle_short_and_cover_coin(
            entry_price_coin2,
            exit_price_coin2,
            coin2,
            hedge_ratio,
            coin2_fee,
            exchange,
            amount_buy_coin1,
        )

        if coin1_profit is not None:
            total_profit = coin1_profit + coin2_profit
        else:
            total_profit = None

        # Combine instructions from both trades
        combined_instructions = [
            instructions_coin1[0],
            instructions_coin2[0],
            wait_instruction,
        ]
        if len(instructions_coin1) > 1:
            combined_instructions.append(instructions_coin1[1])
            combined_instructions.append(instructions_coin2[1])

        # Calculate total and potential profit
        potential_profit = ArbitrageHandler.calculate_potential_profit(
            usd_start,
            entry_price_coin1,
            entry_price_coin2,
            exit_price_coin1,
            exit_price_coin2,
            hedge_ratio,
        )

        # Create waterfall and summary data
        waterfall_data, summary_header = ArbitrageHandler.create_summary(
            total_profit,
            potential_profit,
            coin1_fees_usd,
            coin2_fees_usd,
            pair1,
            pair2,
            exchange,
        )

        # Combine all data into a single dictionary for this opportunity
        return {
            "summary_header": summary_header,
            "waterfall_data": waterfall_data,
            "instructions": combined_instructions,
            "path": [("USD", pair1), (pair1, pair2), (pair2, "USD")],
        }
