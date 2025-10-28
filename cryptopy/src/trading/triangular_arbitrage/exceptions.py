"""Custom exceptions for triangular arbitrage utilities."""


class InsufficientLiquidityError(RuntimeError):
    """Raised when the order book does not provide enough depth for a trade."""


class ExchangeRequestTimeout(RuntimeError):
    """Raised when the exchange does not respond to an order submission in time."""
