"""Exchange connectivity helpers for triangular arbitrage."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, Optional, Sequence

from cryptopy.src.trading.triangular_arbitrage.models import OrderBookSnapshot

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
        websocket_timeout: Optional[float] = 10.0,
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
                    if websocket_timeout and websocket_timeout > 0:
                        order_book = await asyncio.wait_for(
                            self.websocket_client.watch_order_book(symbol, limit),
                            timeout=websocket_timeout,
                        )
                    else:
                        order_book = await self.websocket_client.watch_order_book(symbol, limit)
                except (CcxtNotSupported, AttributeError):  # type: ignore[misc]
                    if not websocket_failed:
                        logger.info(
                            f"{self.exchange_name} websocket order book not supported for {symbol}; falling back to REST polling."
                        )
                    use_websocket = False
                    websocket_failed = True
                    websocket_permanently_unavailable = True
                    continue
                except asyncio.TimeoutError:
                    if not websocket_failed:
                        logger.warning(
                            f"{self.exchange_name} websocket order book timed out for {symbol}; falling back to REST polling."
                        )
                    else:
                        logger.debug(
                            f"{self.exchange_name} websocket order book still timing out for {symbol}; polling via REST."
                        )
                    use_websocket = False
                    websocket_failed = True
                    continue
                except Exception as exc:  # pragma: no cover - network failure path
                    if not websocket_failed:
                        logger.warning(
                            f"{self.exchange_name} websocket order book failed for {symbol}; falling back to REST polling ({exc})."
                        )
                    else:
                        logger.debug(
                            f"{self.exchange_name} websocket order book still unavailable for {symbol} ({exc}); polling via REST."
                        )
                    use_websocket = False
                    websocket_failed = True
                    continue
                else:
                    websocket_failed = False
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

    async def watch_tickers(
        self,
        symbols: Sequence[str],
        *,
        poll_interval: float = 2.0,
        websocket_timeout: Optional[float] = 10.0,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield the latest ticker data for ``symbols``.

        The implementation favours websocket streaming when supported by the
        exchange, falling back to REST polling via :meth:`fetch_tickers` and
        :meth:`fetch_ticker` when websockets are unavailable. The yielded
        dictionaries always contain the requested symbols (when data could be
        fetched) mapped to the native ccxt ticker payloads.
        """

        if not symbols:
            return

        symbol_list = list(dict.fromkeys(symbols))

        async def _rest_poll() -> AsyncIterator[Dict[str, Any]]:
            while True:
                tickers: Dict[str, Any] = {}
                try:
                    fetched = await asyncio.to_thread(self.rest_client.fetch_tickers, symbol_list)
                except TypeError:
                    fetched = None
                except Exception as exc:  # pragma: no cover - network failure path
                    logger.debug(
                        f"{self.exchange_name} REST fetch_tickers failed for {symbol_list}: {exc}"
                    )
                    fetched = None

                if fetched:
                    tickers = {symbol: fetched.get(symbol) for symbol in symbol_list if fetched.get(symbol) is not None}
                if not tickers:
                    for symbol in symbol_list:
                        try:
                            ticker = await asyncio.to_thread(self.rest_client.fetch_ticker, symbol)
                        except Exception:
                            continue
                        else:
                            tickers[symbol] = ticker

                if tickers:
                    yield tickers

                await asyncio.sleep(max(poll_interval, 0.1))

        use_websocket = self.websocket_client is not None
        websocket_failed = False

        while True:
            if use_websocket and self.websocket_client is not None:
                try:
                    if hasattr(self.websocket_client, "watch_tickers"):
                        if websocket_timeout and websocket_timeout > 0:
                            payload = await asyncio.wait_for(
                                self.websocket_client.watch_tickers(symbol_list),
                                timeout=websocket_timeout,
                            )
                        else:
                            payload = await self.websocket_client.watch_tickers(symbol_list)
                        tickers = {
                            symbol: payload.get(symbol)
                            for symbol in symbol_list
                            if payload and payload.get(symbol) is not None
                        }
                        if tickers:
                            websocket_failed = False
                            yield tickers
                            continue
                    elif len(symbol_list) == 1 and hasattr(self.websocket_client, "watch_ticker"):
                        symbol = symbol_list[0]
                        if websocket_timeout and websocket_timeout > 0:
                            ticker = await asyncio.wait_for(
                                self.websocket_client.watch_ticker(symbol),
                                timeout=websocket_timeout,
                            )
                        else:
                            ticker = await self.websocket_client.watch_ticker(symbol)
                        if ticker:
                            websocket_failed = False
                            yield {symbol: ticker}
                            continue
                    else:
                        raise AttributeError("watch_tickers not supported")
                except (CcxtNotSupported, AttributeError):  # type: ignore[misc]
                    if not websocket_failed:
                        logger.info(
                            f"{self.exchange_name} websocket ticker stream not supported; falling back to REST polling."
                        )
                    use_websocket = False
                    websocket_failed = True
                    continue
                except asyncio.TimeoutError:
                    if not websocket_failed:
                        logger.warning(
                            f"{self.exchange_name} websocket ticker stream timed out; falling back to REST polling."
                        )
                    use_websocket = False
                    websocket_failed = True
                    continue
                except Exception as exc:  # pragma: no cover - network failure path
                    if not websocket_failed:
                        logger.warning(
                            f"{self.exchange_name} websocket ticker stream failed; falling back to REST polling ({exc})."
                        )
                    use_websocket = False
                    websocket_failed = True
                    continue

            async for tickers in _rest_poll():
                yield tickers
                if self.websocket_client is not None and websocket_failed:
                    # Periodically attempt to resume websocket streaming when available again.
                    use_websocket = True
                    break

    @property
    def sandbox_supported(self) -> bool:
        """Return ``True`` if ccxt successfully enabled sandbox mode."""

        return self._rest_sandbox_enabled or self._ws_sandbox_enabled

    def _enable_sandbox_mode(self, client: Any, transport: str) -> bool:
        """Attempt to enable ccxt sandbox mode and log when unsupported."""

        if not hasattr(client, "set_sandbox_mode"):
            logger.info(
                f"{self.exchange_name} sandbox mode is not available for {transport} connections via ccxt."
            )
            return False

        try:
            client.set_sandbox_mode(True)
            return True
        except CcxtNotSupported as exc:  # type: ignore[misc]
            logger.info(
                f"{self.exchange_name} does not provide a ccxt sandbox endpoint; running against production API. ({exc})"
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                f"Failed to enable sandbox mode for {self.exchange_name} ({transport} connection): {exc}"
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
