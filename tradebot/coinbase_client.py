from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


def _d(value: Any) -> Decimal:
    return Decimal(str(value))


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float, str)):
        try:
            return _d(value)
        except Exception:  # pragma: no cover - defensive parse
            return None
    if isinstance(value, dict):
        for key in ("value", "amount", "price", "commission_total", "total"):
            if key in value:
                parsed = _to_decimal(value.get(key))
                if parsed is not None:
                    return parsed
    return None


def _find_numeric(data: Any, candidate_keys: tuple[str, ...]) -> Decimal | None:
    if data is None:
        return None
    stack = [data]
    lowered = {k.lower() for k in candidate_keys}
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() in lowered:
                    parsed = _to_decimal(value)
                    if parsed is not None:
                        return parsed
                stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return None


def _path_get(data: Any, path: tuple[Any, ...]) -> Any:
    current = data
    for part in path:
        if isinstance(part, int):
            if not isinstance(current, list) or part >= len(current):
                return None
            current = current[part]
            continue
        if not isinstance(current, dict):
            return None
        if part not in current:
            return None
        current = current[part]
    return current


def _normalize_rate(value: Decimal | None) -> Decimal | None:
    if value is None:
        return None
    if value < 0:
        return None
    # Coinbase fee rates are normally decimal fractions (e.g. 0.006 = 0.6%).
    # If an API surface returns percent-like values (e.g. 1.2), normalize safely.
    if value > Decimal("1"):
        if value <= Decimal("100"):
            return value / Decimal("100")
        if value <= Decimal("10000"):
            return value / Decimal("10000")
        return None
    return value


@dataclass(slots=True)
class FeeRates:
    maker_rate: Decimal
    taker_rate: Decimal

    @property
    def roundtrip_bps(self) -> Decimal:
        return (self.taker_rate * Decimal("2")) * Decimal("10000")


def _parse_message_timestamp(value: Any) -> datetime:
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if text:
            normalized = text.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(normalized)
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)


def _is_loopback_discard_proxy(value: str) -> bool:
    text = str(value).strip()
    if not text:
        return False
    parsed = urllib.parse.urlparse(text if "://" in text else f"http://{text}")
    host = str(parsed.hostname or "").strip().lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return False
    try:
        return int(parsed.port or 0) == 9
    except Exception:
        return False


def _sanitize_broken_proxy_env() -> list[str]:
    cleared: list[str] = []
    # Some launcher environments force loopback discard proxies like 127.0.0.1:9,
    # which makes Coinbase REST and websocket calls fail immediately.
    for key in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "GIT_HTTP_PROXY",
        "GIT_HTTPS_PROXY",
    ):
        value = os.environ.get(key)
        if value and _is_loopback_discard_proxy(value):
            os.environ.pop(key, None)
            cleared.append(key)
    return cleared


@dataclass(slots=True)
class TickerSnapshot:
    product_id: str
    price: Decimal
    best_bid: Decimal | None
    best_ask: Decimal | None
    received_at: datetime
    update_id: int


class CoinbaseTickerWebSocketFeed:
    def __init__(
        self,
        *,
        market_data_url: str,
        product_ids: list[str],
        channel: str = "ticker",
        stale_seconds: float = 5.0,
        ping_interval_seconds: int = 20,
        ping_timeout_seconds: int = 10,
        max_reconnect_seconds: int = 30,
        subscribe_heartbeats: bool = True,
    ):
        self.market_data_url = str(market_data_url)
        self.product_ids = sorted({str(item).upper() for item in product_ids if str(item).strip()})
        self.channel = str(channel).strip().lower()
        self.stale_seconds = float(stale_seconds)
        self.ping_interval_seconds = int(ping_interval_seconds)
        self.ping_timeout_seconds = int(ping_timeout_seconds)
        self.max_reconnect_seconds = int(max_reconnect_seconds)
        self.subscribe_heartbeats = bool(subscribe_heartbeats)

        self._lock = threading.Lock()
        self._latest_by_product: dict[str, TickerSnapshot] = {}
        self._last_message_at: datetime | None = None
        self._last_error: str | None = None
        self._connected = False
        self._update_counter = 0

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ws_module: Any | None = None
        self._ws_app: Any | None = None
        self._session_connected = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not self.product_ids:
            raise RuntimeError("Coinbase websocket feed requires at least one product_id")
        try:
            import websocket as websocket_module
        except Exception as exc:
            raise RuntimeError(
                "websocket-client package is required for websocket market data. "
                "Install with: pip install websocket-client"
            ) from exc
        self._ws_module = websocket_module
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_forever,
            daemon=True,
            name="CoinbaseTickerWebSocketFeed",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        ws = self._ws_app
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def is_connected(self) -> bool:
        return bool(self._connected)

    def last_error(self) -> str | None:
        return self._last_error

    def last_message_age_seconds(self) -> float | None:
        with self._lock:
            last = self._last_message_at
        if last is None:
            return None
        return (datetime.now(timezone.utc) - last).total_seconds()

    def latest_mid_price(
        self,
        product_id: str,
        max_age_seconds: float | None = None,
    ) -> Decimal | None:
        target = str(product_id).upper()
        with self._lock:
            snapshot = self._latest_by_product.get(target)
        if snapshot is None:
            return None
        ttl = self.stale_seconds if max_age_seconds is None else float(max_age_seconds)
        age = (datetime.now(timezone.utc) - snapshot.received_at).total_seconds()
        if age > ttl:
            return None
        if snapshot.best_bid is not None and snapshot.best_ask is not None:
            if snapshot.best_bid > 0 and snapshot.best_ask > 0:
                return (snapshot.best_bid + snapshot.best_ask) / Decimal("2")
        if snapshot.price > 0:
            return snapshot.price
        return None

    def latest_snapshot(
        self,
        product_id: str,
    ) -> dict[str, Any] | None:
        target = str(product_id).upper()
        with self._lock:
            snapshot = self._latest_by_product.get(target)
        if snapshot is None:
            return None
        age = (datetime.now(timezone.utc) - snapshot.received_at).total_seconds()
        return {
            "product_id": snapshot.product_id,
            "price": str(snapshot.price),
            "best_bid": str(snapshot.best_bid) if snapshot.best_bid is not None else None,
            "best_ask": str(snapshot.best_ask) if snapshot.best_ask is not None else None,
            "received_at": snapshot.received_at.isoformat(),
            "update_id": snapshot.update_id,
            "age_seconds": round(age, 3),
        }

    def ingest_message(self, payload: str | dict[str, Any]) -> None:
        if isinstance(payload, str):
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                return
            if not isinstance(data, dict):
                return
            self._ingest_message_dict(data)
            return
        if isinstance(payload, dict):
            self._ingest_message_dict(payload)

    def _run_forever(self) -> None:
        if self._ws_module is None:
            return
        backoff_seconds = 1.0
        while not self._stop_event.is_set():
            self._session_connected = False
            self._connected = False
            self._ws_app = self._ws_module.WebSocketApp(
                self.market_data_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            try:
                self._ws_app.run_forever(
                    ping_interval=self.ping_interval_seconds,
                    ping_timeout=self.ping_timeout_seconds,
                )
            except Exception as exc:
                self._last_error = f"websocket run error: {exc}"
            finally:
                self._connected = False
                self._ws_app = None
            if self._stop_event.is_set():
                break
            if self._session_connected:
                backoff_seconds = 1.0
            else:
                backoff_seconds = min(
                    float(self.max_reconnect_seconds),
                    max(1.0, backoff_seconds * 2.0),
                )
            time.sleep(backoff_seconds)

    def _on_open(self, ws: Any) -> None:
        self._connected = True
        self._session_connected = True
        self._last_error = None
        subscribe = {
            "type": "subscribe",
            "channel": self.channel,
            "product_ids": self.product_ids,
        }
        ws.send(json.dumps(subscribe))
        if self.subscribe_heartbeats:
            heartbeat_subscribe = {
                "type": "subscribe",
                "channel": "heartbeats",
                "product_ids": self.product_ids,
            }
            ws.send(json.dumps(heartbeat_subscribe))

    def _on_message(self, _ws: Any, message: str) -> None:
        self.ingest_message(message)

    def _on_error(self, _ws: Any, error: Any) -> None:
        self._last_error = f"websocket error: {error}"

    def _on_close(self, _ws: Any, _status_code: Any, _message: Any) -> None:
        self._connected = False

    def _ingest_message_dict(self, data: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._last_message_at = now

        events = data.get("events")
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                tickers = event.get("tickers")
                if isinstance(tickers, list):
                    for ticker in tickers:
                        if isinstance(ticker, dict):
                            self._ingest_ticker_row(ticker, data)
            return
        if "product_id" in data and (
            "price" in data or "best_bid" in data or "best_ask" in data
        ):
            self._ingest_ticker_row(data, data)

    def _ingest_ticker_row(self, ticker: dict[str, Any], envelope: dict[str, Any]) -> None:
        product_id = str(ticker.get("product_id", "")).upper().strip()
        if not product_id:
            return
        price = _to_decimal(ticker.get("price"))
        best_bid = _to_decimal(ticker.get("best_bid"))
        best_ask = _to_decimal(ticker.get("best_ask"))
        if price is None and best_bid is not None and best_ask is not None:
            if best_bid > 0 and best_ask > 0:
                price = (best_bid + best_ask) / Decimal("2")
        if price is None:
            return
        row_ts = (
            ticker.get("time")
            or ticker.get("timestamp")
            or envelope.get("timestamp")
            or envelope.get("time")
        )
        received_at = _parse_message_timestamp(row_ts)
        with self._lock:
            self._update_counter += 1
            self._latest_by_product[product_id] = TickerSnapshot(
                product_id=product_id,
                price=price,
                best_bid=best_bid,
                best_ask=best_ask,
                received_at=received_at,
                update_id=self._update_counter,
            )


class CoinbaseAdvancedClient:
    _BASE_URL = "https://api.coinbase.com"
    _MAX_REQUEST_ATTEMPTS = 4

    def __init__(self, key_file: str, timeout_seconds: int = 30):
        self._cleared_proxy_env = _sanitize_broken_proxy_env()
        self.key_file = str(Path(key_file))
        self.timeout_seconds = timeout_seconds
        self.jwt_helper = str(Path(__file__).with_name("jwt_helper.mjs"))
        if not Path(self.key_file).exists():
            raise RuntimeError(f"Coinbase key file not found: {self.key_file}")
        if not Path(self.jwt_helper).exists():
            raise RuntimeError(f"JWT helper not found: {self.jwt_helper}")

    def _build_jwt(self, method: str, path: str) -> str:
        cmd = ["node", self.jwt_helper, self.key_file, method.upper(), path]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
            raise RuntimeError(f"JWT helper failed: {detail}")
        token = proc.stdout.strip()
        if not token:
            raise RuntimeError("JWT helper returned an empty token")
        return token

    def _request(
        self,
        method: str,
        path: str,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if query:
            encoded = urllib.parse.urlencode(
                {k: v for k, v in query.items() if v is not None},
                doseq=True,
            )
            full_path = f"{path}?{encoded}"
        else:
            full_path = path
        payload = None if body is None else json.dumps(body).encode("utf-8")
        try:
            # Preferred signing format for Coinbase App keys.
            return self._request_once(
                method=method,
                signed_path=path,
                full_path=full_path,
                payload=payload,
            )
        except RuntimeError as exc:
            # Some environments require the query string to be part of the JWT uri.
            is_401 = " 401 " in str(exc) or "401 Unauthorized" in str(exc)
            if query and full_path != path and is_401:
                return self._request_once(
                    method=method,
                    signed_path=full_path,
                    full_path=full_path,
                    payload=payload,
                )
            raise

    def _request_once(
        self,
        method: str,
        signed_path: str,
        full_path: str,
        payload: bytes | None,
    ) -> dict[str, Any]:
        for attempt in range(1, self._MAX_REQUEST_ATTEMPTS + 1):
            token = self._build_jwt(method, signed_path)
            req = urllib.request.Request(
                url=f"{self._BASE_URL}{full_path}",
                data=payload,
                method=method.upper(),
            )
            req.add_header("Authorization", f"Bearer {token}")
            req.add_header("Content-Type", "application/json")
            req.add_header("User-Agent", "TradeBot/1.0")

            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                error_text = exc.read().decode("utf-8", errors="replace")
                if (
                    attempt < self._MAX_REQUEST_ATTEMPTS
                    and self._is_retryable_http_error(exc.code, error_text)
                ):
                    delay = self._retry_delay_seconds(attempt)
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Coinbase API error {exc.code} {exc.reason} on {method.upper()} {full_path}: {error_text}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < self._MAX_REQUEST_ATTEMPTS:
                    delay = self._retry_delay_seconds(attempt)
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Coinbase API network error on {method.upper()} {full_path}: {exc.reason}"
                ) from exc

        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            return {"data": parsed}
        except json.JSONDecodeError:
            return {"raw": raw}

    @staticmethod
    def _is_retryable_http_error(code: int, error_text: str) -> bool:
        if code in {408, 425, 429, 500, 502, 503, 504}:
            return True
        if code == 403:
            lowered = error_text.lower()
            if "too many errors" in lowered or "rate" in lowered or "throttle" in lowered:
                return True
        return False

    @staticmethod
    def _retry_delay_seconds(attempt_number: int) -> float:
        # Exponential backoff capped to keep loop responsiveness reasonable.
        return min(6.0, 0.5 * (2 ** max(0, attempt_number - 1)))

    def _all_accounts(self) -> list[dict[str, Any]]:
        accounts: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(20):
            payload = self._request(
                "GET",
                "/api/v3/brokerage/accounts",
                query={"limit": 250, "cursor": cursor},
            )
            page = payload.get("accounts", [])
            if isinstance(page, list):
                accounts.extend(item for item in page if isinstance(item, dict))
            has_next = bool(payload.get("has_next"))
            cursor = payload.get("cursor")
            if not has_next or not cursor:
                break
        return accounts

    def recent_candles(
        self, product_id: str, granularity: str, lookback: int
    ) -> list[dict[str, Any]]:
        seconds_per = {
            "ONE_MINUTE": 60,
            "FIVE_MINUTE": 300,
            "FIFTEEN_MINUTE": 900,
            "THIRTY_MINUTE": 1800,
            "ONE_HOUR": 3600,
            "TWO_HOUR": 7200,
            "SIX_HOUR": 21600,
            "ONE_DAY": 86400,
        }.get(granularity, 3600)
        # Coinbase rejects candle ranges that are too large ("should be less than 350").
        # Fetch in smaller windows and merge.
        max_candles_per_request = 300
        effective_lookback = max(int(lookback), 50)
        total_seconds = seconds_per * effective_lookback
        end_ts = int(datetime.now(timezone.utc).timestamp())
        start_ts = end_ts - total_seconds

        merged_by_start: dict[int, dict[str, Any]] = {}
        cursor = start_ts
        while cursor < end_ts:
            window_end = min(end_ts, cursor + (seconds_per * max_candles_per_request))
            data = self._request(
                "GET",
                f"/api/v3/brokerage/products/{urllib.parse.quote(product_id)}/candles",
                query={
                    "start": int(cursor),
                    "end": int(window_end),
                    "granularity": granularity,
                },
            )
            candles = data.get("candles", []) if isinstance(data, dict) else []
            for candle in candles:
                if not isinstance(candle, dict):
                    continue
                if candle.get("close") is None:
                    continue
                try:
                    start_value = int(candle.get("start", 0))
                except Exception:
                    continue
                merged_by_start[start_value] = candle
            if window_end <= cursor:
                break
            cursor = window_end

        merged = sorted(merged_by_start.values(), key=lambda c: int(c.get("start", 0)))
        if len(merged) > effective_lookback:
            merged = merged[-effective_lookback:]
        return merged

    def mid_price(self, product_id: str) -> Decimal:
        data = self._request(
            "GET",
            "/api/v3/brokerage/best_bid_ask",
            query={"product_ids": product_id},
        )
        bid = _to_decimal(_path_get(data, ("pricebooks", 0, "bids", 0, "price")))
        ask = _to_decimal(_path_get(data, ("pricebooks", 0, "asks", 0, "price")))
        if bid is not None and ask is not None:
            return (bid + ask) / Decimal("2")
        if bid is not None:
            return bid
        if ask is not None:
            return ask
        fallback = _find_numeric(data, ("mid_market_price", "price"))
        if fallback is None:
            raise RuntimeError(f"Could not parse price from Coinbase response for {product_id}")
        return fallback

    def best_bid_ask(self, product_id: str) -> tuple[Decimal | None, Decimal | None]:
        data = self._request(
            "GET",
            "/api/v3/brokerage/best_bid_ask",
            query={"product_ids": product_id},
        )
        bid = _to_decimal(_path_get(data, ("pricebooks", 0, "bids", 0, "price")))
        ask = _to_decimal(_path_get(data, ("pricebooks", 0, "asks", 0, "price")))
        return bid, ask

    def _parse_fee_rates_from_tier(self, tier: dict[str, Any]) -> FeeRates | None:
        maker = _normalize_rate(_to_decimal(tier.get("maker_fee_rate")))
        taker = _normalize_rate(_to_decimal(tier.get("taker_fee_rate")))
        if maker is None or taker is None:
            return None
        return FeeRates(maker_rate=maker, taker_rate=taker)

    def _parse_advanced_spot_fee_rates(self, data: dict[str, Any]) -> FeeRates | None:
        # Preferred shape for Advanced Trade transaction summary.
        tier = data.get("fee_tier")
        if isinstance(tier, dict):
            parsed = self._parse_fee_rates_from_tier(tier)
            if parsed is not None:
                return parsed

        # Alternative shape used by some versions: fees array with ranges.
        fees = data.get("fees")
        if isinstance(fees, list):
            total_volume = _to_decimal(data.get("total_volume"))
            candidate_rows = [row for row in fees if isinstance(row, dict)]
            if total_volume is not None:
                for row in candidate_rows:
                    usd_from = _to_decimal(row.get("usd_from"))
                    usd_to = _to_decimal(row.get("usd_to"))
                    if usd_from is None:
                        continue
                    upper_ok = usd_to is None or usd_to == 0 or total_volume < usd_to
                    if total_volume >= usd_from and upper_ok:
                        parsed = self._parse_fee_rates_from_tier(row)
                        if parsed is not None:
                            return parsed
            for row in candidate_rows:
                parsed = self._parse_fee_rates_from_tier(row)
                if parsed is not None:
                    return parsed
        return None

    def fee_rates(self, product_type: str = "SPOT") -> FeeRates:
        data = self._request(
            "GET",
            "/api/v3/brokerage/transaction_summary",
            query={"product_type": product_type},
        )
        parsed = self._parse_advanced_spot_fee_rates(data)
        if parsed is not None:
            return parsed
        maker = _normalize_rate(_find_numeric(data, ("maker_fee_rate",)))
        taker = _normalize_rate(_find_numeric(data, ("taker_fee_rate",)))
        if maker is None or taker is None:
            raise RuntimeError("Could not parse maker/taker fee rates from transaction summary")
        return FeeRates(maker_rate=maker, taker_rate=taker)

    def currency_balances(self) -> dict[str, Decimal]:
        out: dict[str, Decimal] = {}
        accounts = self._all_accounts()
        for account in accounts:
            currency = str(account.get("currency", "")).upper().strip()
            available = _path_get(account, ("available_balance", "value"))
            amount = _to_decimal(available)
            if not currency or amount is None:
                continue
            out[currency] = out.get(currency, Decimal("0")) + amount
        return out

    def currency_account_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for account in self._all_accounts():
            currency = str(account.get("currency", "")).upper().strip()
            account_id = str(account.get("uuid", "")).strip()
            if currency and account_id and currency not in out:
                out[currency] = account_id
        return out

    def preview_market_buy(self, product_id: str, quote_size: Decimal) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v3/brokerage/orders/preview",
            body={
                "product_id": product_id,
                "side": "BUY",
                "order_configuration": {
                    "market_market_ioc": {"quote_size": str(quote_size)},
                },
            },
        )

    def preview_market_sell(self, product_id: str, base_size: Decimal) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v3/brokerage/orders/preview",
            body={
                "product_id": product_id,
                "side": "SELL",
                "order_configuration": {
                    "market_market_ioc": {"base_size": str(base_size)},
                },
            },
        )

    def preview_commission(self, preview_response: dict[str, Any]) -> Decimal:
        value = _find_numeric(
            preview_response,
            ("commission_total", "total_commission", "commission", "fee", "fees"),
        )
        return value if value is not None else Decimal("0")

    def market_buy(self, product_id: str, quote_size: Decimal) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v3/brokerage/orders",
            body={
                "client_order_id": str(uuid.uuid4()),
                "product_id": product_id,
                "side": "BUY",
                "order_configuration": {
                    "market_market_ioc": {"quote_size": str(quote_size)},
                },
            },
        )

    def market_sell(self, product_id: str, base_size: Decimal) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v3/brokerage/orders",
            body={
                "client_order_id": str(uuid.uuid4()),
                "product_id": product_id,
                "side": "SELL",
                "order_configuration": {
                    "market_market_ioc": {"base_size": str(base_size)},
                },
            },
        )

    def limit_buy_post_only(
        self, product_id: str, base_size: Decimal, limit_price: Decimal
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v3/brokerage/orders",
            body={
                "client_order_id": str(uuid.uuid4()),
                "product_id": product_id,
                "side": "BUY",
                "order_configuration": {
                    "limit_limit_gtc": {
                        "base_size": str(base_size),
                        "limit_price": str(limit_price),
                        "post_only": True,
                    }
                },
            },
        )

    def limit_sell_post_only(
        self, product_id: str, base_size: Decimal, limit_price: Decimal
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v3/brokerage/orders",
            body={
                "client_order_id": str(uuid.uuid4()),
                "product_id": product_id,
                "side": "SELL",
                "order_configuration": {
                    "limit_limit_gtc": {
                        "base_size": str(base_size),
                        "limit_price": str(limit_price),
                        "post_only": True,
                    }
                },
            },
        )

    def get_order(self, order_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/api/v3/brokerage/orders/historical/{urllib.parse.quote(order_id)}",
        )

    def ticker_websocket_feed(
        self,
        *,
        product_ids: list[str],
        market_data_url: str,
        channel: str,
        stale_seconds: float,
        ping_interval_seconds: int,
        ping_timeout_seconds: int,
        max_reconnect_seconds: int,
        subscribe_heartbeats: bool,
    ) -> CoinbaseTickerWebSocketFeed:
        return CoinbaseTickerWebSocketFeed(
            market_data_url=market_data_url,
            product_ids=product_ids,
            channel=channel,
            stale_seconds=stale_seconds,
            ping_interval_seconds=ping_interval_seconds,
            ping_timeout_seconds=ping_timeout_seconds,
            max_reconnect_seconds=max_reconnect_seconds,
            subscribe_heartbeats=subscribe_heartbeats,
        )

    def open_short(
        self,
        product_id: str,
        base_size: Decimal,
        leverage: Decimal,
        margin_type: str = "CROSS",
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v3/brokerage/orders",
            body={
                "client_order_id": str(uuid.uuid4()),
                "product_id": product_id,
                "side": "SELL",
                "leverage": str(leverage),
                "margin_type": margin_type,
                "order_configuration": {
                    "market_market_ioc": {"base_size": str(base_size)},
                },
            },
        )

    def close_position(self, product_id: str, size: Decimal | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "client_order_id": str(uuid.uuid4()),
            "product_id": product_id,
        }
        if size is not None:
            body["size"] = str(size)
        return self._request("POST", "/api/v3/brokerage/orders/close_position", body=body)

    def create_convert_quote(
        self, from_account: str, to_account: str, amount: Decimal
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v3/brokerage/convert/quote",
            body={
                "from_account": from_account,
                "to_account": to_account,
                "amount": str(amount),
            },
        )

    def commit_convert(
        self, trade_id: str, from_account: str, to_account: str
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/v3/brokerage/convert/{urllib.parse.quote(trade_id)}",
            body={"from_account": from_account, "to_account": to_account},
        )
