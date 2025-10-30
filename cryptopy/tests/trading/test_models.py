import pytest

from cryptopy.src.trading.triangular_arbitrage.models import OrderBookSnapshot


def test_order_book_snapshot_from_ccxt_normalises_levels():
    order_book = {
        "bids": [
            [100.0, 1.5, 1680000000000],
            [99.0, "0.5", "ignored"],
            [98.0],
        ],
        "asks": [
            [101.0, 2.0, 1680000005000],
            None,
            [102.0, "not-a-number"],
        ],
        "timestamp": 1680000010000,
    }

    snapshot = OrderBookSnapshot.from_ccxt("BTC/USD", order_book)

    assert snapshot.bids == [(100.0, 1.5), (99.0, 0.5)]
    assert snapshot.asks == [(101.0, 2.0)]
    assert snapshot.timestamp == pytest.approx(1680000010.0)
