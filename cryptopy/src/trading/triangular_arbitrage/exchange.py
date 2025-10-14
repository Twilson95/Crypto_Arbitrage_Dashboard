"""Exchange connectivity helpers for triangular arbitrage."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, Optional, Sequence

from .models import OrderBookSnapshot

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import ccxt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    ccxt = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from ccxt.base.errors import NotSupported as CcxtNotSupported  # type: ignore
except Exception:  # pragma: no cover - fallback if ccxt is missing or outdated
    CcxtNotSupported = ()  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import ccxt.pro as ccxtpro  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade if ccxt.pro is not available
    ccxtpro = None  # type: ignore


class ExchangeConnection:
    """Encapsulates exchange specific functionality for data and trading."""

    def __init__(
        self,
        exchange_name: str,
        *,
        credentials: Optional[Dict[str, str]] = None,
        use_testnet: bool = True,
        enable_websocket: bool = True,
        make_trades: bool = False,
        rest_client: Optional[Any] = None,
        websocket_client: Optional[Any] = None,
    ) -> None:
        if ccxt is None:  # pragma: no cover - dependency not installed
            raise ModuleNotFoundError(
                "ccxt is required to use ExchangeConnection. Install ccxt or provide a custom client."
            )
        self.exchange_name = exchange_name.lower()
        self.credentials = credentials or {}
        self.make_trades = make_trades
        self._use_testnet = use_testnet
        self._rest_sandbox_enabled = False
        self._ws_sandbox_enabled = False

        exchange_class = getattr(ccxt, self.exchange_name)
        rest_config = {
            "enableRateLimit": True,
            **self.credentials,
        }
        self.rest_client = rest_client or exchange_class(rest_config)

        if use_testnet:
            self._rest_sandbox_enabled = self._enable_sandbox_mode(self.rest_client, "REST")

        self.websocket_client = websocket_client
        if websocket_client is None and enable_websocket and ccxtpro is not None:
            ws_class = getattr(ccxtpro, self.exchange_name)
            ws_config = {
                "enableRateLimit": True,
                **self.credentials,
            }
            self.websocket_client = ws_class(ws_config)
            if use_testnet:
                self._ws_sandbox_enabled = self._enable_sandbox_mode(self.websocket_client, "websocket")

        self._market_cache = self.rest_client.load_markets()
        self._default_fee = (
            self.rest_client.fees.get("trading", {}).get("taker") if hasattr(self.rest_client, "fees") else None
        )

    def get_markets(self) -> Dict[str, Any]:
        """Return the cached market metadata loaded during initialisation."""

        return self._market_cache

    def list_symbols(self) -> Sequence[str]:
        """Return the list of symbols supported by the exchange."""

        return list(self._market_cache.keys())

    def get_taker_fee(self, symbol: str) -> float:
        market = self._market_cache.get(symbol, {})
        fee = market.get("taker")
        if fee is None:
            return float(self._default_fee or 0.0)
        return float(fee)

    def get_order_book(self, symbol: str, *, limit: int = 10) -> OrderBookSnapshot:
        order_book = self.rest_client.fetch_order_book(symbol, limit)
        return OrderBookSnapshot.from_ccxt(symbol, order_book)

    async def watch_order_book(
        self,
        symbol: str,
        *,
        limit: int = 10,
        poll_interval: float = 2.0,
    ) -> AsyncIterator[OrderBookSnapshot]:
        async def _poll_rest() -> AsyncIterator[OrderBookSnapshot]:
            while True:
                order_book = await asyncio.to_thread(self.rest_client.fetch_order_book, symbol, limit)
                yield OrderBookSnapshot.from_ccxt(symbol, order_book)
                await asyncio.sleep(poll_interval)

        use_websocket = self.websocket_client is not None
        websocket_failed = False
        websocket_permanently_unavailable = False

        while True:
            if use_websocket and self.websocket_client is not None and not websocket_permanently_unavailable:
                try:
                    order_book = await self.websocket_client.watch_order_book(symbol, limit)
                except (CcxtNotSupported, AttributeError):  # type: ignore[misc]
                    if not websocket_failed:
                        logger.info(
                            "%s websocket order book not supported for %s; falling back to REST polling.",
                            self.exchange_name,
                            symbol,
                        )
                    use_websocket = False
                    websocket_failed = True
                    websocket_permanently_unavailable = True
                    continue
                except Exception as exc:  # pragma: no cover - network failure path
                    if not websocket_failed:
                        logger.warning(
                            "%s websocket order book failed for %s; falling back to REST polling (%s).",
                            self.exchange_name,
                            symbol,
                            exc,
                        )
                    else:
                        logger.debug(
                            "%s websocket order book still unavailable for %s (%s); polling via REST.",
                            self.exchange_name,
                            symbol,
                            exc,
                        )
                    use_websocket = False
                    websocket_failed = True
                    continue
                else:
                    yield OrderBookSnapshot.from_ccxt(symbol, order_book)
                    continue

            async for snapshot in _poll_rest():
                yield snapshot
                if (
                    self.websocket_client is not None
                    and websocket_failed
                    and not websocket_permanently_unavailable
                ):
                    # Periodically attempt to return to websocket streaming if available again.
                    use_websocket = True
                    break

    def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        *,
        params: Optional[Dict[str, Any]] = None,
        test_order: Optional[bool] = None,
    ) -> Dict[str, Any]:
        params = params or {}
        if test_order is None:
            test_order = not self.make_trades
        if test_order:
            return {
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "params": params,
                "test_order": True,
            }
        return self.rest_client.create_order(symbol, "market", side, amount, params=params)

    @property
    def sandbox_supported(self) -> bool:
        """Return ``True`` if ccxt successfully enabled sandbox mode."""

        return self._rest_sandbox_enabled or self._ws_sandbox_enabled

    def _enable_sandbox_mode(self, client: Any, transport: str) -> bool:
        """Attempt to enable ccxt sandbox mode and log when unsupported."""

        if not hasattr(client, "set_sandbox_mode"):
            logger.info(
                "%s sandbox mode is not available for %s connections via ccxt.",
                self.exchange_name,
                transport,
            )
            return False

        try:
            client.set_sandbox_mode(True)
            return True
        except CcxtNotSupported as exc:  # type: ignore[misc]
            logger.info(
                "%s does not provide a ccxt sandbox endpoint; running against production API. (%s)",
                self.exchange_name,
                exc,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Failed to enable sandbox mode for %s (%s connection): %s",
                self.exchange_name,
                transport,
                exc,
            )
        return False

    def close(self) -> None:
        if hasattr(self.rest_client, "close"):
            try:
                self.rest_client.close()
            except Exception:
                pass
        if self.websocket_client and hasattr(self.websocket_client, "close"):
            try:
                loop = None
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    pass
                close_coro = self.websocket_client.close()
                if loop and loop.is_running():
                    loop.create_task(close_coro)
                else:
                    asyncio.run(close_coro)
            except Exception:
                pass

    async def aclose(self) -> None:
        if hasattr(self.rest_client, "close"):
            try:
                self.rest_client.close()
            except Exception:
                pass
        if self.websocket_client and hasattr(self.websocket_client, "close"):
            try:
                await self.websocket_client.close()
            except Exception:
                pass
