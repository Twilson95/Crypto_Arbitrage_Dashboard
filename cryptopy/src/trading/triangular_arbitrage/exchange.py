"""Exchange connectivity helpers for triangular arbitrage."""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence

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
    from ccxt.base.errors import AuthenticationError as CcxtAuthenticationError  # type: ignore
except Exception:  # pragma: no cover - fallback if ccxt is missing or outdated
    class _FallbackAuthenticationError(Exception):
        """Fallback authentication error used when ccxt is unavailable."""

    CcxtAuthenticationError = _FallbackAuthenticationError  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import ccxt.pro as ccxtpro  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade if ccxt.pro is not available
    ccxtpro = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import requests
except ModuleNotFoundError:  # pragma: no cover - optional dependency missing
    requests = None  # type: ignore


SANDBOX_MARKET_DATA_UNAVAILABLE = {
    "kraken",
}


class ExchangeConnection:
    """Encapsulates exchange specific functionality for data and trading."""

    _PREWARM_KEEPALIVE_INTERVAL = 480.0

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
        self._shared_http_session = requests.Session() if requests is not None else None
        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_loop_ready = threading.Event()
        self._order_endpoint_prewarm_attempted = False
        self._order_endpoint_prewarm_succeeded = False
        self._prewarm_keepalive_stop = threading.Event()
        self._prewarm_keepalive_thread: Optional[threading.Thread] = None

        exchange_class = getattr(ccxt, self.exchange_name)
        rest_config = {
            "enableRateLimit": True,
            **self.credentials,
        }
        self.rest_client = rest_client or exchange_class(rest_config)
        self._attach_shared_session(self.rest_client)
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
            self._attach_shared_session(self.market_data_client)
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

        if self._supports_private_order_streams():
            self._start_order_watch_loop()

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
        self._verify_authenticated_access()
        self._start_prewarm_keepalive_loop()

    def get_markets(self) -> Dict[str, Any]:
        """Return the cached market metadata loaded during initialisation."""

        return self._market_cache

    def refresh_trading_fees(self, symbols: Optional[Sequence[str]] = None) -> Dict[str, float]:
        """Refresh cached taker fee rates for ``symbols`` when supported by the exchange.

        Returns a mapping of symbols whose fees were updated to their refreshed taker
        rates. When ``symbols`` is omitted, the method attempts to refresh all
        available pairs via the bulk endpoint without issuing per-symbol requests.
        """

        updated: Dict[str, float] = {}
        if symbols:
            unique_symbols = [symbol for symbol in dict.fromkeys(symbols) if symbol in self._market_cache]
        else:
            unique_symbols = []

        requested = set(unique_symbols)

        client_has = getattr(self.rest_client, "has", {})
        has_bulk = bool(client_has.get("fetchTradingFees")) if isinstance(client_has, dict) else False
        if not has_bulk and hasattr(client_has, "get"):
            has_bulk = bool(client_has.get("fetchTradingFees"))  # type: ignore[arg-type]

        if has_bulk:
            try:
                payload = self.rest_client.fetch_trading_fees()
            except Exception:  # pragma: no cover - network dependent
                logger.debug("fetch_trading_fees refresh call failed", exc_info=True)
            else:
                if isinstance(payload, dict):
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
                        self._fee_fetch_failures.discard(symbol)
                        if not requested or symbol in requested:
                            updated[symbol] = rate

                if not symbols:
                    return updated

        targets = [symbol for symbol in unique_symbols if symbol not in updated]
        if not targets and symbols:
            targets = [symbol for symbol in requested if symbol not in updated]

        has_single = bool(client_has.get("fetchTradingFee")) if isinstance(client_has, dict) else False
        if not has_single and hasattr(client_has, "get"):
            has_single = bool(client_has.get("fetchTradingFee"))  # type: ignore[arg-type]

        if has_single and targets:
            for symbol in targets:
                try:
                    payload = self.rest_client.fetch_trading_fee(symbol)
                except Exception:  # pragma: no cover - network dependent
                    self._fee_fetch_failures.add(symbol)
                    logger.debug(
                        "fetch_trading_fee refresh call failed for %s", symbol, exc_info=True
                    )
                    continue

                taker_fee = payload.get("taker") if isinstance(payload, dict) else None
                if taker_fee is None:
                    continue
                try:
                    rate = float(taker_fee)
                except (TypeError, ValueError):  # pragma: no cover - defensive cast
                    continue

                self._trading_fee_cache[symbol] = rate
                self._fee_source[symbol] = "fetchTradingFee"
                self._fee_fetch_failures.discard(symbol)
                updated[symbol] = rate

        return updated

    def _select_prewarm_symbol(self, preferred: Optional[str] = None) -> Optional[str]:
        if preferred and preferred in self._market_cache:
            return preferred

        for candidate, market in self._market_cache.items():
            if not isinstance(market, dict):
                continue
            if market.get("active") is False:
                continue
            market_type = market.get("type")
            if market.get("spot") is False and market_type not in (None, "spot"):
                continue
            return candidate

        if self._market_cache:
            return next(iter(self._market_cache))
        return None

    def _prewarm_generic_private_call(self) -> bool:
        if not self.make_trades:
            return False

        fetch_balance = getattr(self.rest_client, "fetch_balance", None)
        if not callable(fetch_balance):
            return False

        try:
            self._apply_credentials(self.rest_client)
            self._ensure_required_credentials(self.rest_client)
            fetch_balance()
        except Exception:  # pragma: no cover - network dependent warmup
            logger.debug(
                "Generic trading prewarm call failed for %s", self.exchange_name, exc_info=True
            )
            return False

        return True

    def _prewarm_kraken_order_endpoint(self, symbol: Optional[str]) -> bool:
        candidate = self._select_prewarm_symbol(symbol)
        if not candidate:
            return False

        market = self._market_cache.get(candidate, {})
        min_values = self.get_min_trade_values(candidate)
        amount = min_values.get("amount")
        if amount is None:
            amount = self._coerce_float(market.get("lot"))
        if amount is None or amount <= 0:
            amount = 1.0

        try:
            precise_amount = self.amount_to_precision(candidate, float(amount))
        except Exception:
            precise_amount = float(amount)

        if precise_amount <= 0:
            precise_amount = float(amount) if float(amount) > 0 else 1.0

        params = {"validate": True}

        try:
            self._apply_credentials(self.rest_client)
            self._ensure_required_credentials(self.rest_client)
            self.rest_client.create_order(candidate, "market", "buy", precise_amount, params=params)
        except Exception:  # pragma: no cover - depends on exchange/network
            logger.debug(
                "Kraken order endpoint prewarm failed for %s", candidate, exc_info=True
            )
            return False

        logger.debug(
            "Kraken trading session prewarmed via validate order on %s", candidate
        )
        return True

    def prewarm_trading_connection(
        self, symbol: Optional[str] = None, *, force: bool = False
    ) -> None:
        if not self.make_trades:
            return

        if self._order_endpoint_prewarm_attempted and not force:
            if self._order_endpoint_prewarm_succeeded or symbol is None or self.exchange_name != "kraken":
                return

        order_success = False
        generic_success = False
        attempted = False

        if self.exchange_name == "kraken":
            attempted = True
            order_success = self._prewarm_kraken_order_endpoint(symbol)
            if not order_success:
                generic_success = self._prewarm_generic_private_call()
        else:
            attempted = True
            generic_success = self._prewarm_generic_private_call()

        if attempted:
            self._order_endpoint_prewarm_attempted = True
        if order_success or (self.exchange_name != "kraken" and generic_success):
            self._order_endpoint_prewarm_succeeded = True

    def _start_prewarm_keepalive_loop(self) -> None:
        if not self.make_trades:
            return

        prewarm = getattr(self, "prewarm_trading_connection", None)
        if not callable(prewarm):
            return

        if self._prewarm_keepalive_thread and self._prewarm_keepalive_thread.is_alive():
            return

        interval = max(float(self._PREWARM_KEEPALIVE_INTERVAL), 60.0)

        def _run() -> None:
            while not self._prewarm_keepalive_stop.wait(interval):
                try:
                    prewarm(None, force=True)
                except Exception:
                    logger.debug(
                        "Periodic trading connection prewarm failed for %s",
                        self.exchange_name,
                        exc_info=True,
                    )

        self._prewarm_keepalive_stop.clear()
        thread = threading.Thread(
            target=_run,
            name=f"{self.exchange_name}-prewarm-keepalive",
            daemon=True,
        )
        thread.start()
        self._prewarm_keepalive_thread = thread

    def _stop_prewarm_keepalive_loop(self) -> None:
        thread = self._prewarm_keepalive_thread
        if not thread:
            return

        self._prewarm_keepalive_stop.set()
        thread.join(timeout=5.0)
        self._prewarm_keepalive_thread = None

    def _attach_shared_session(self, client: Any) -> None:
        if not client or self._shared_http_session is None:
            return
        if hasattr(client, "session"):
            try:
                client.session = self._shared_http_session
            except Exception:
                logger.debug(
                    "Unable to attach shared HTTP session to %s client", client,
                    exc_info=True,
                )

    def _start_order_watch_loop(self) -> None:
        if self._ws_loop is not None:
            return

        loop = asyncio.new_event_loop()
        self._ws_loop = loop

        def _run_loop() -> None:
            asyncio.set_event_loop(loop)
            self._ws_loop_ready.set()
            try:
                loop.run_forever()
            finally:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                try:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        thread = threading.Thread(target=_run_loop, name="exchange-order-watch", daemon=True)
        self._ws_thread = thread
        thread.start()
        self._ws_loop_ready.wait(timeout=5.0)

    def _stop_order_watch_loop(self) -> None:
        loop = self._ws_loop
        thread = self._ws_thread
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._ws_loop = None
        self._ws_thread = None
        self._ws_loop_ready = threading.Event()

    def _close_websocket_client(self, client: Any) -> None:
        if not client or not hasattr(client, "close"):
            return
        try:
            close_result = client.close()
        except Exception:
            logger.debug("Failed to close websocket client", exc_info=True)
            return
        if asyncio.iscoroutine(close_result):
            loop = self._ws_loop
            try:
                if loop is not None and loop.is_running():
                    asyncio.run_coroutine_threadsafe(close_result, loop).result(timeout=5.0)
                else:
                    asyncio.run(close_result)
            except Exception:
                logger.debug("Error awaiting websocket client close", exc_info=True)

    def _supports_private_order_streams(self) -> bool:
        client = self.websocket_client
        return bool(client and hasattr(client, "watch_order"))

    @staticmethod
    def _order_payload_closed(payload: Dict[str, Any]) -> bool:
        status = str(payload.get("status") or "").lower()
        if status in {"closed", "canceled", "cancelled", "rejected"}:
            return True
        remaining = payload.get("remaining")
        if remaining is not None:
            try:
                return float(remaining) <= 0
            except (TypeError, ValueError):
                pass
        return False

    async def _watch_order_until_complete(
        self,
        order_id: str,
        symbol: str,
        timeout: float,
    ) -> Optional[Dict[str, Any]]:
        client = self.websocket_client
        if client is None or not hasattr(client, "watch_order"):
            return None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(timeout, 0.0)
        last_payload: Optional[Dict[str, Any]] = None

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return last_payload
            try:
                payload = await asyncio.wait_for(
                    client.watch_order(order_id, symbol),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                return last_payload
            except Exception:
                logger.debug(
                    "watch_order update failed for %s %s", order_id, symbol, exc_info=True
                )
                return last_payload

            if not isinstance(payload, dict):
                continue

            last_payload = payload
            if self._order_payload_closed(payload):
                return payload

    def watch_order_via_websocket(
        self, order_id: str, symbol: str, *, timeout: float
    ) -> Optional[Dict[str, Any]]:
        """Attempt to stream order updates via websocket for faster fills."""

        if not self._supports_private_order_streams():
            return None
        if self._ws_loop is None:
            self._start_order_watch_loop()
        coroutine = self._watch_order_until_complete(order_id, symbol, timeout)
        try:
            result = asyncio.run_coroutine_threadsafe(coroutine, self._ws_loop).result(
                timeout + 1.0
            )
        except (concurrent.futures.TimeoutError, RuntimeError):
            return None
        except Exception:
            logger.debug(
                "watch_order coroutine errored for %s %s", order_id, symbol, exc_info=True
            )
            return None
        return result

    def _to_precision(self, method_name: str, symbol: str, value: float) -> float:
        method = getattr(self.rest_client, method_name, None)
        if method is None:
            return float(value)
        try:
            return float(method(symbol, float(value)))
        except Exception:  # pragma: no cover - precision helpers should never raise
            return float(value)

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        """Round ``amount`` down to the exchange's supported precision."""

        return self._to_precision("amount_to_precision", symbol, amount)

    def price_to_precision(self, symbol: str, price: float) -> float:
        """Round ``price`` down to the exchange's supported precision."""

        return self._to_precision("price_to_precision", symbol, price)

    def cost_to_precision(self, symbol: str, cost: float) -> float:
        """Round ``cost`` down to the exchange's supported precision."""

        method = getattr(self.rest_client, "cost_to_precision", None)
        if method is None:
            return float(cost)
        try:
            return float(method(symbol, float(cost)))
        except Exception:  # pragma: no cover - precision helpers should never raise
            return float(cost)

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
            if hasattr(client, "config") and isinstance(client.config, dict):
                for key in ("apiKey", "api_key", "key"):
                    client.config[key] = api_key
            headers = getattr(client, "headers", None)
            if isinstance(headers, dict):
                headers.setdefault("API-Key", api_key)
                headers.setdefault("apiKey", api_key)
            elif headers is None and hasattr(client, "headers"):
                try:
                    client.headers = {"API-Key": api_key}
                except AttributeError:
                    pass
        if secret:
            for attr in ("secret", "secretKey", "secret_key"):
                try:
                    setattr(client, attr, secret)
                except AttributeError:
                    pass
            if hasattr(client, "config") and isinstance(client.config, dict):
                for key in ("secret", "secretKey", "secret_key"):
                    client.config[key] = secret
        if password:
            for attr in ("password", "passphrase"):
                try:
                    setattr(client, attr, password)
                except AttributeError:
                    pass
            if hasattr(client, "config") and isinstance(client.config, dict):
                for key in ("password", "passphrase"):
                    client.config[key] = password

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

    def _ensure_required_credentials(self, client: Any) -> None:
        """Ensure ccxt required credentials are populated on ``client``."""

        if not self.credentials or client is None:
            return

        check_required = getattr(client, "check_required_credentials", None)
        if not callable(check_required):
            return

        # Reapply credentials immediately beforehand in case ccxt reset them.
        self._apply_credentials(client)

        try:
            check_required()
        except CcxtAuthenticationError as exc:  # type: ignore[misc]
            raise CcxtAuthenticationError(  # type: ignore[call-arg]
                f"Missing required credentials for {self.exchange_name}: {exc}"
            ) from exc
        except Exception:
            # If ccxt raises for other reasons (e.g. partially implemented exchanges),
            # surface the original behaviour without interrupting execution.
            logger.debug(
                f"check_required_credentials failed for {self.exchange_name}",
                exc_info=True,
            )

    def _verify_authenticated_access(self) -> None:
        """Call a private endpoint to surface credential issues early."""

        if not self.credentials:
            return

        private_checks: list[tuple[str, Callable[[], Any]]] = []
        client = self.rest_client

        self._ensure_required_credentials(client)

        fetch_balance = getattr(client, "fetch_balance", None)
        if callable(fetch_balance):
            private_checks.append(("fetch_balance", lambda: fetch_balance()))

        fetch_open_orders = getattr(client, "fetch_open_orders", None)
        if callable(fetch_open_orders):
            private_checks.append(("fetch_open_orders", lambda: fetch_open_orders()))

        if not private_checks:
            if self.make_trades:
                logger.debug(
                    f"No private credential checks available for {self.exchange_name}; "
                    "trading requests may surface authentication errors later."
                )
            return

        for name, method in private_checks:
            try:
                method()
            except CcxtAuthenticationError as exc:  # type: ignore[misc]
                raise CcxtAuthenticationError(  # type: ignore[call-arg]
                    f"Authentication failed when calling {name} on {self.exchange_name}: {exc}"
                ) from exc
            except Exception as exc:  # pragma: no cover - network dependent
                logger.debug(
                    f"Credential verification via {name} on {self.exchange_name} failed with "
                    f"{exc.__class__.__name__}: {exc}",
                    exc_info=True,
                )
                continue
            else:
                logger.debug(
                    f"Verified authenticated access for {self.exchange_name} via {name}."
                )
                return

        if self.make_trades:
            logger.warning(
                f"Unable to confirm trading credentials for {self.exchange_name}; "
                "orders may fail with authentication errors."
            )

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

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def get_min_trade_values(self, symbol: str) -> Dict[str, float]:
        """Return the minimum trade limits for ``symbol`` when available."""

        market = self._market_cache.get(symbol, {})
        if not isinstance(market, dict):
            return {}

        result: Dict[str, float] = {}
        limits = market.get("limits")
        if isinstance(limits, dict):
            for key in ("amount", "cost"):
                raw = limits.get(key)
                if isinstance(raw, dict):
                    coerced = self._coerce_float(raw.get("min"))
                    if coerced is not None and coerced > 0:
                        result[key] = coerced

        if "amount" not in result:
            info = market.get("info")
            if isinstance(info, dict):
                coerced = self._coerce_float(info.get("ordermin"))
                if coerced is not None and coerced > 0:
                    result["amount"] = coerced

        return result

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
        require_websocket: bool = False,
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
                if require_websocket and not hasattr(
                    self.market_data_websocket_client, "watch_order_book"
                ):
                    raise AttributeError(
                        f"{self.exchange_name} websocket order book not supported for {symbol}"
                    )
                try:
                    if websocket_timeout and websocket_timeout > 0:
                        order_book = await asyncio.wait_for(
                            self.market_data_websocket_client.watch_order_book(symbol, limit),
                            timeout=websocket_timeout,
                        )
                    else:
                        order_book = await self.market_data_websocket_client.watch_order_book(symbol, limit)
                except (CcxtNotSupported, AttributeError):  # type: ignore[misc]
                    if require_websocket:
                        raise
                    if not websocket_failed:
                        logger.info(
                            f"{self.exchange_name} websocket order book not supported for {symbol}; falling back to REST polling."
                        )
                    use_websocket = False
                    websocket_failed = True
                    websocket_permanently_unavailable = True
                    continue
                except asyncio.TimeoutError:
                    if require_websocket:
                        raise
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
                    if require_websocket:
                        raise
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

            if require_websocket:
                raise RuntimeError(
                    f"{self.exchange_name} websocket order book unavailable for {symbol}"
                )

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
        self._ensure_required_credentials(self.rest_client)
        return self.rest_client.create_order(symbol, "market", side, amount, params=params)

    def fetch_balance(self) -> Dict[str, Any]:
        """Fetch the authenticated account balances from the exchange."""

        self._apply_credentials(self.rest_client)
        self._ensure_required_credentials(self.rest_client)
        return self.rest_client.fetch_balance()

    def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Fetch a specific order from the exchange."""

        self._apply_credentials(self.rest_client)
        self._ensure_required_credentials(self.rest_client)
        return self.rest_client.fetch_order(order_id, symbol)

    def fetch_my_trades(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch recent trades for the authenticated account."""

        self._apply_credentials(self.rest_client)
        self._ensure_required_credentials(self.rest_client)
        return self.rest_client.fetch_my_trades(symbol, since=since, limit=limit)

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
        self._stop_prewarm_keepalive_loop()
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
        if websocket_to_close:
            self._close_websocket_client(websocket_to_close)
        if (
            self.websocket_client
            and self.websocket_client is not self.market_data_websocket_client
            and hasattr(self.websocket_client, "close")
        ):
            self._close_websocket_client(self.websocket_client)
        self._stop_order_watch_loop()
        if self._shared_http_session is not None:
            try:
                self._shared_http_session.close()
            except Exception:
                logger.debug("Failed to close shared HTTP session", exc_info=True)
            self._shared_http_session = None

    async def aclose(self) -> None:
        self._stop_prewarm_keepalive_loop()
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
        self._stop_order_watch_loop()
        if self._shared_http_session is not None:
            try:
                self._shared_http_session.close()
            except Exception:
                logger.debug("Failed to close shared HTTP session", exc_info=True)
            self._shared_http_session = None
