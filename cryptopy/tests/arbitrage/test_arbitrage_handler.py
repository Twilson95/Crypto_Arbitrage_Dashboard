import unittest
from cryptopy.src.arbitrage.ArbitrageHandler import ArbitrageHandler


class TestArbitrageHandler(unittest.TestCase):

    def test_calculate_buy_step(self):
        # Test inputs
        buy_exchange = "ExchangeA"
        currency = "BTC/USD"
        buy_price = 100  # USD/BTC
        input_funds = 1000  # USD
        buy_taker_fee = 0.01  # 1% taker fee

        # Call the method
        result = ArbitrageHandler.calculate_buy_step(
            buy_exchange, currency, buy_price, input_funds, buy_taker_fee
        )

        # Expected values
        expected_crypto = 9.9  # (1000 - 10) / 100 = 9.9 BTC
        expected_buy_fees = 10  # 1% of 1000 USD
        expected_usd = 990  # 9.9 BTC * 100 USD/BTC

        # Assertions
        self.assertEqual(result["instruction"], "buy")
        self.assertEqual(result["from_exchange"], buy_exchange)
        self.assertEqual(result["to_currency"], "BTC")
        self.assertAlmostEqual(result["to_amount"], expected_crypto, places=3)
        self.assertAlmostEqual(result["buy_fees"], expected_buy_fees, places=3)
        self.assertAlmostEqual(result["to_usd"], expected_usd, places=3)

    def test_calculate_transfer_step(self):
        # Test inputs
        from_exchange = "ExchangeA"
        to_exchange = "ExchangeB"
        from_crypto = 1.0  # 1 BTC
        from_usd = 100  # USD equivalent
        sell_price = 100  # USD/BTC
        withdraw_fee = 0.01  # 1% withdraw fee
        deposit_fee = 0.02  # 2% deposit fee
        network_fee_crypto = 0.005  # 0.005 BTC network fee
        currency = "BTC/USD"

        # Call the method
        result = ArbitrageHandler.calculate_transfer_step(
            from_exchange,
            to_exchange,
            from_crypto,
            from_usd,
            sell_price,
            withdraw_fee,
            deposit_fee,
            network_fee_crypto,
            currency,
        )

        expected_crypto_after_fees = 0.9653  # (1 BTC - 0.01 BTC withdraw - 0.005 BTC network) * (1 - 2% deposit fee)
        expected_usd = expected_crypto_after_fees * sell_price
        expected_network_fees_usd = 0.49

        self.assertEqual(result["instruction"], "transfer")
        self.assertEqual(result["from_exchange"], from_exchange)
        self.assertEqual(result["to_exchange"], to_exchange)
        self.assertAlmostEqual(
            result["to_amount"], expected_crypto_after_fees, places=3
        )
        self.assertAlmostEqual(result["to_usd"], expected_usd, places=3)
        self.assertAlmostEqual(
            result["fees"]["network_fees"], expected_network_fees_usd, places=3
        )

    def test_calculate_sell_step(self):
        sell_exchange = "ExchangeB"
        currency = "BTC/USD"
        from_crypto = 1.0  # 1 BTC
        from_usd = 100  # USD equivalent
        sell_price = 110  # USD/BTC
        sell_taker_fee = 0.02  # 2% taker fee

        # Call the method
        result = ArbitrageHandler.calculate_sell_step(
            sell_exchange,
            currency,
            from_crypto,
            from_usd,
            sell_price,
            sell_taker_fee,
        )

        # Expected values
        expected_sell_fees = 0.02 * from_crypto  # 2% of 1 BTC = 0.02 BTC
        expected_usd = (
            from_crypto - expected_sell_fees
        ) * sell_price  # USD after sell fees

        # Assertions
        self.assertEqual(result["instruction"], "sell")
        self.assertEqual(result["from_exchange"], sell_exchange)
        self.assertEqual(result["to_currency"], "USD")
        self.assertAlmostEqual(result["to_amount"], expected_usd, places=3)
        self.assertAlmostEqual(
            result["sell_fees"], expected_sell_fees * sell_price, places=3
        )


if __name__ == "__main__":
    unittest.main()
