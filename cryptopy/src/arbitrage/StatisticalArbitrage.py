class StatisticalArbitrage:

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
            print("hedge ratio is negative", hedge_ratio)
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
        ) = StatisticalArbitrage.handle_buy_and_sell_coin(
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
        ) = StatisticalArbitrage.handle_short_and_cover_coin(
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
            total_profit = 0

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
        potential_profit = StatisticalArbitrage.calculate_potential_profit(
            usd_start,
            entry_price_coin1,
            entry_price_coin2,
            exit_price_coin1,
            exit_price_coin2,
            hedge_ratio,
        )

        # Create waterfall and summary data
        waterfall_data, summary_header = StatisticalArbitrage.create_summary(
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

    @staticmethod
    def handle_buy_and_sell_coin(
        usd_start, entry_price, exit_price, coin, currency_fee, exchange
    ):
        # Entry: Buy Coin1
        amount_buy_coin = usd_start / entry_price
        fees_buy_coin = amount_buy_coin * currency_fee
        net_buy_coin = amount_buy_coin - fees_buy_coin
        usd_after_buy = net_buy_coin * entry_price
        change_in_usd_buy = usd_after_buy - usd_start

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
                buy_fee_usd,
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

        return (
            [buy_instruction, sell_instruction],
            usd_after_buy,
            usd_after_sell,
            total_fees_usd,
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
                fees_sell_coin * entry_price,
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
            arbitrage_opportunity = (
                StatisticalArbitrage.statistical_arbitrage_iteration(
                    entry,
                    exit,
                    pairs,
                    currency_fees,
                    price_df,
                    funds,
                    hedge_ratio,
                    exchange,
                )
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
        total_fees = coin1_fees_usd + coin2_fees_usd
        potential_profit = total_profit + total_fees

        waterfall_data = {
            "Potential Profit": potential_profit,
            "Coin-1 Fees": -coin1_fees_usd,
            "Coin-2 Fees": -coin2_fees_usd,
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

        return potential_profit

    @staticmethod
    def identify_all_statistical_arbitrage(
        prices, pair, spread_details, currency_fees, exchange, funds, window=30
    ):
        spread = spread_details["spread"]
        hedge_ratio = spread_details["hedge_ratio"]
        drawing_instructions = StatisticalArbitrage.get_statistical_arbitrage_trades(
            spread, window
        )
        arbitrage_instructions = StatisticalArbitrage.identify_statistical_arbitrage(
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
