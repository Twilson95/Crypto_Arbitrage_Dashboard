class SimpleArbitrage:

    @staticmethod
    def calculate_buy_step(
        buy_exchange, currency, buy_price, input_funds, buy_taker_fee
    ):
        buy_fees = input_funds * buy_taker_fee
        to_crypto = (input_funds - buy_fees) / buy_price
        to_usd = to_crypto * buy_price

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
        sell_fees = from_crypto * sell_taker_fee
        to_usd = (from_crypto - sell_fees) * sell_price

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
    def identify_arbitrage(
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
            buy_price = prices_buy.close[-1]

            for sell_exchange, prices_sell in exchange_prices.items():
                if sell_exchange == buy_exchange:
                    continue

                sell_taker_fee = currency_fees.get(sell_exchange, {}).get("taker", 0)
                deposit_fee = exchange_fees.get(sell_exchange, {}).get("deposit", 0)
                sell_price = prices_sell.close[-1]

                # Calculate the Buy Step
                buy_instruction = SimpleArbitrage.calculate_buy_step(
                    buy_exchange, currency, buy_price, input_funds, buy_taker_fee
                )
                current_usd = buy_instruction["to_usd"]
                current_amount = buy_instruction["to_amount"]

                # transfer instructions
                transfer_instruction = SimpleArbitrage.calculate_transfer_step(
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
                sell_instruction = SimpleArbitrage.calculate_sell_step(
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
