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


SANDBOX_MARKET_DATA_UNAVAILABLE = {
    "kraken",
}


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
        market_data_rest_client: Optional[Any] = None,
        market_data_websocket_client: Optional[Any] = None,
    ) -> None:
        if ccxt is None:  # pragma: no cover - dependency not installed
            raise ModuleNotFoundError(
                "ccxt is required to use ExchangeConnection. Install ccxt or provide a custom client."
            )
        self.exchange_name = exchange_name.lower()
        self.credentials = self._normalise_credentials(credentials)
        self.make_trades = make_trades
        self._use_testnet = use_testnet
        self._rest_sandbox_enabled = False
        self._market_data_rest_sandbox_enabled = False
        self._ws_sandbox_enabled = False

        exchange_class = getattr(ccxt, self.exchange_name)
        rest_config = {
            "enableRateLimit": True,
            **self.credentials,
        }
        self.rest_client = rest_client or exchange_class(rest_config)
        self._apply_credentials(self.rest_client)

        if use_testnet:
            self._rest_sandbox_enabled = self._enable_sandbox_mode(self.rest_client, "REST trading")
            # Some exchanges reset credential fields when toggling sandbox mode.
            # Reapply credentials so subsequent authenticated calls inherit them.
            self._apply_credentials(self.rest_client)

        market_data_config = {
            "enableRateLimit": True,
            **self.credentials,
        }

        if market_data_rest_client is not None:
            self.market_data_client = market_data_rest_client
        elif use_testnet and self.exchange_name in SANDBOX_MARKET_DATA_UNAVAILABLE:
            self.market_data_client = exchange_class(market_data_config)
            logger.info(
                f"{self.exchange_name} does not expose sandbox market data; using production REST endpoints for price feeds."
            )
        else:
            self.market_data_client = self.rest_client
        if self.market_data_client is not self.rest_client:
            self._apply_credentials(self.market_data_client)

        if (
            use_testnet
            and self.market_data_client is not self.rest_client
            and self.exchange_name not in SANDBOX_MARKET_DATA_UNAVAILABLE
        ):
            self._market_data_rest_sandbox_enabled = self._enable_sandbox_mode(
                self.market_data_client,
                "REST market data",
            )
            self._apply_credentials(self.market_data_client)
        elif self.market_data_client is self.rest_client:
            self._market_data_rest_sandbox_enabled = self._rest_sandbox_enabled

        self.websocket_client = websocket_client
        if websocket_client is None and enable_websocket and ccxtpro is not None:
            ws_class = getattr(ccxtpro, self.exchange_name)
            ws_config = {
                "enableRateLimit": True,
                **self.credentials,
            }
            self.websocket_client = ws_class(ws_config)
            self._apply_credentials(self.websocket_client)

        self.market_data_websocket_client = market_data_websocket_client or self.websocket_client
        if (
            self.market_data_websocket_client is not None
            and self.market_data_websocket_client is not self.websocket_client
        ):
            self._apply_credentials(self.market_data_websocket_client)

        if self.market_data_websocket_client is not None and use_testnet:
            if self.exchange_name in SANDBOX_MARKET_DATA_UNAVAILABLE:
                self._ws_sandbox_enabled = False
                logger.info(
                    f"{self.exchange_name} websocket sandbox feeds are unavailable; using production market data stream."
                )
            else:
                self._ws_sandbox_enabled = self._enable_sandbox_mode(
                    self.market_data_websocket_client,
                    "websocket",
                )
                self._apply_credentials(self.market_data_websocket_client)
        else:
            self._ws_sandbox_enabled = False

        self._market_cache = self.market_data_client.load_markets()
        if self.market_data_client is not self.rest_client:
            try:
                self.rest_client.load_markets()
            except Exception:
                logger.debug(
                    f"Failed to load markets on trading client for {self.exchange_name}",
                    exc_info=True,
                )
        self._default_fee = (
            self.rest_client.fees.get("trading", {}).get("taker") if hasattr(self.rest_client, "fees") else None
        )
        try:
            self._default_fee = float(self._default_fee) if self._default_fee is not None else None
        except (TypeError, ValueError):  # pragma: no cover - defensive cast
            self._default_fee = None

        self._trading_fee_cache: Dict[str, float] = {}
        self._fee_source: Dict[str, str] = {}
        self._fee_fetch_failures: set[str] = set()

        for symbol, market in self._market_cache.items():
            taker = market.get("taker") if isinstance(market, dict) else None
            if taker is None:
                continue
            try:
                rate = float(taker)
            except (TypeError, ValueError):  # pragma: no cover - defensive cast
                continue
            self._trading_fee_cache[symbol] = rate
            self._fee_source[symbol] = "market"

        self._prime_trading_fee_cache()

    def get_markets(self) -> Dict[str, Any]:
        """Return the cached market metadata loaded during initialisation."""

        return self._market_cache

    @staticmethod
    def _normalise_credentials(credentials: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Return a ccxt-compatible credential mapping."""

        if not credentials:
            return {}

        normalised: Dict[str, str] = {}
        for raw_key, raw_value in credentials.items():
            if raw_value in (None, ""):
                continue
            key = str(raw_key)
            value = str(raw_value)
            lower = key.lower()
            if lower in {"apikey", "api_key", "key"}:
                normalised["apiKey"] = value
            elif lower in {"secret", "api_secret", "secretkey"}:
                normalised["secret"] = value
            elif lower in {"password", "passphrase"}:
                normalised["password"] = value
            else:
                normalised[key] = value
        return normalised

    def _apply_credentials(self, client: Any) -> None:
        """Set credential attributes on ``client`` for ccxt compatibility."""

        if not self.credentials or client is None:
            return
        api_key = self.credentials.get("apiKey")
        secret = self.credentials.get("secret")
        password = self.credentials.get("password")

        if api_key:
            for attr in ("apiKey", "api_key", "key"):
                try:
                    setattr(client, attr, api_key)
                except AttributeError:
                    # Some ccxt clients use slots; ignore attributes that cannot be set.
                    pass
        if secret:
            for attr in ("secret", "secretKey", "secret_key"):
                try:
                    setattr(client, attr, secret)
                except AttributeError:
                    pass
        if password:
            for attr in ("password", "passphrase"):
                try:
                    setattr(client, attr, password)
                except AttributeError:
                    pass

        if hasattr(client, "options") and isinstance(client.options, dict):
            if api_key:
                for key in ("apiKey", "api_key", "key"):
                    client.options[key] = api_key
            if secret:
                for key in ("secret", "secretKey", "secret_key"):
                    client.options[key] = secret
            if password:
                for key in ("password", "passphrase"):
                    client.options[key] = password

        required = getattr(client, "requiredCredentials", None)
        if isinstance(required, dict):
            for field, needed in required.items():
                if not needed:
                    continue
                if getattr(client, field, None):
                    continue
                value = None
                if field == "apiKey":
                    value = api_key
                elif field == "secret":
                    value = secret
                elif field in {"password", "passphrase"}:
                    value = password
                else:
                    value = self.credentials.get(field)
                if value:
                    try:
                        setattr(client, field, value)
                    except AttributeError:
                        pass

    def list_symbols(self) -> Sequence[str]:
        """Return the list of symbols supported by the exchange."""

        return list(self._market_cache.keys())

    def get_taker_fee(self, symbol: str) -> float:
        cached = self._trading_fee_cache.get(symbol)
        if cached is not None:
            return cached

        market = self._market_cache.get(symbol, {})
        fee = market.get("taker") if isinstance(market, dict) else None
        if fee is not None:
            try:
                rate = float(fee)
            except (TypeError, ValueError):  # pragma: no cover - defensive cast
                rate = 0.0
            self._trading_fee_cache[symbol] = rate
            self._fee_source.setdefault(symbol, "market")
            return rate

        if symbol not in self._fee_fetch_failures:
            client_has = getattr(self.rest_client, "has", {})
            has_single = bool(client_has.get("fetchTradingFee")) if isinstance(client_has, dict) else False
            if not has_single and hasattr(client_has, "get"):
                has_single = bool(client_has.get("fetchTradingFee"))  # type: ignore[arg-type]
            if has_single:
                try:
                    payload = self.rest_client.fetch_trading_fee(symbol)
                except Exception:  # pragma: no cover - network dependent
                    self._fee_fetch_failures.add(symbol)
                    logger.debug(
                        f"Failed to fetch taker fee for {symbol} via fetch_trading_fee", exc_info=True
                    )
                else:
                    taker_fee = payload.get("taker") if isinstance(payload, dict) else None
                    if taker_fee is not None:
                        try:
                            rate = float(taker_fee)
                        except (TypeError, ValueError):  # pragma: no cover
                            rate = 0.0
                        self._trading_fee_cache[symbol] = rate
                        self._fee_source[symbol] = "fetchTradingFee"
                        return rate

        rate = float(self._default_fee or 0.0)
        self._trading_fee_cache[symbol] = rate
        self._fee_source.setdefault(symbol, "default")
        return rate

    def get_fee_sources(self) -> Dict[str, str]:
        """Return a mapping of symbols to the source of their taker fee."""

        return dict(self._fee_source)

    def _prime_trading_fee_cache(self) -> None:
        """Attempt to populate the taker fee cache using exchange endpoints."""

        client_has = getattr(self.rest_client, "has", {})
        has_bulk = bool(client_has.get("fetchTradingFees")) if isinstance(client_has, dict) else False
        if not has_bulk and hasattr(client_has, "get"):
            has_bulk = bool(client_has.get("fetchTradingFees"))  # type: ignore[arg-type]

        if not has_bulk:
            return

        try:
            payload = self.rest_client.fetch_trading_fees()
        except Exception:  # pragma: no cover - network dependent
            logger.debug("fetch_trading_fees call failed", exc_info=True)
            return

        if not isinstance(payload, dict):
            return

        for symbol, info in payload.items():
            taker_fee = info.get("taker") if isinstance(info, dict) else None
            if taker_fee is None:
                continue
            try:
                rate = float(taker_fee)
            except (TypeError, ValueError):  # pragma: no cover - defensive cast
                continue
            self._trading_fee_cache[symbol] = rate
            self._fee_source[symbol] = "fetchTradingFees"

    def get_order_book(self, symbol: str, *, limit: int = 10) -> OrderBookSnapshot:
        order_book = self.market_data_client.fetch_order_book(symbol, limit)
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
                order_book = await asyncio.to_thread(
                    self.market_data_client.fetch_order_book,
                    symbol,
                    limit,
                )
                yield OrderBookSnapshot.from_ccxt(symbol, order_book)
                await asyncio.sleep(poll_interval)

        use_websocket = self.market_data_websocket_client is not None
        websocket_failed = False
        websocket_permanently_unavailable = False

        while True:
            if (
                use_websocket
                and self.market_data_websocket_client is not None
                and not websocket_permanently_unavailable
            ):
                try:
                    if websocket_timeout and websocket_timeout > 0:
                        order_book = await asyncio.wait_for(
                            self.market_data_websocket_client.watch_order_book(symbol, limit),
                            timeout=websocket_timeout,
                        )
                    else:
                        order_book = await self.market_data_websocket_client.watch_order_book(symbol, limit)
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
                    self.market_data_websocket_client is not None
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
        self._apply_credentials(self.rest_client)
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
                    fetched = await asyncio.to_thread(
                        self.market_data_client.fetch_tickers,
                        symbol_list,
                    )
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
                            ticker = await asyncio.to_thread(
                                self.market_data_client.fetch_ticker,
                                symbol,
                            )
                        except Exception:
                            continue
                        else:
                            tickers[symbol] = ticker

                if tickers:
                    yield tickers

                await asyncio.sleep(max(poll_interval, 0.1))

        use_websocket = self.market_data_websocket_client is not None
        websocket_failed = False

        while True:
            if use_websocket and self.market_data_websocket_client is not None:
                try:
                    if hasattr(self.market_data_websocket_client, "watch_tickers"):
                        if websocket_timeout and websocket_timeout > 0:
                            payload = await asyncio.wait_for(
                                self.market_data_websocket_client.watch_tickers(symbol_list),
                                timeout=websocket_timeout,
                            )
                        else:
                            payload = await self.market_data_websocket_client.watch_tickers(symbol_list)
                        tickers = {
                            symbol: payload.get(symbol)
                            for symbol in symbol_list
                            if payload and payload.get(symbol) is not None
                        }
                        if tickers:
                            websocket_failed = False
                            yield tickers
                            continue
                    elif len(symbol_list) == 1 and hasattr(self.market_data_websocket_client, "watch_ticker"):
                        symbol = symbol_list[0]
                        if websocket_timeout and websocket_timeout > 0:
                            ticker = await asyncio.wait_for(
                                self.market_data_websocket_client.watch_ticker(symbol),
                                timeout=websocket_timeout,
                            )
                        else:
                            ticker = await self.market_data_websocket_client.watch_ticker(symbol)
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
                if self.market_data_websocket_client is not None and websocket_failed:
                    # Periodically attempt to resume websocket streaming when available again.
                    use_websocket = True
                    break

    @property
    def sandbox_supported(self) -> bool:
        """Return ``True`` if ccxt successfully enabled sandbox mode."""

        return (
            self._rest_sandbox_enabled
            or self._market_data_rest_sandbox_enabled
            or self._ws_sandbox_enabled
        )

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
        if self.market_data_client is not self.rest_client and hasattr(self.market_data_client, "close"):
            try:
                self.market_data_client.close()
            except Exception:
                pass
        websocket_to_close = self.market_data_websocket_client
        if websocket_to_close and hasattr(websocket_to_close, "close"):
            try:
                loop = None
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    pass
                close_coro = websocket_to_close.close()
                if loop and loop.is_running():
                    loop.create_task(close_coro)
                else:
                    asyncio.run(close_coro)
            except Exception:
                pass
        if (
            self.websocket_client
            and self.websocket_client is not self.market_data_websocket_client
            and hasattr(self.websocket_client, "close")
        ):
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
        if self.market_data_client is not self.rest_client and hasattr(self.market_data_client, "close"):
            try:
                self.market_data_client.close()
            except Exception:
                pass
        websocket_to_close = self.market_data_websocket_client
        if websocket_to_close and hasattr(websocket_to_close, "close"):
            try:
                await websocket_to_close.close()
            except Exception:
                pass
        if (
            self.websocket_client
            and self.websocket_client is not self.market_data_websocket_client
            and hasattr(self.websocket_client, "close")
        ):
            try:
                await self.websocket_client.close()
            except Exception:
                pass
