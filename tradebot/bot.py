from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from .coinbase_client import CoinbaseAdvancedClient, CoinbaseTickerWebSocketFeed, FeeRates
from .backtest import run_scenario_suite, walk_forward_backtest
from .config import BotConfig
from .metrics import append_equity_point, append_trade, load_metrics, save_metrics
from .risk import RiskContext, buy_checks, sell_checks, short_open_checks
from .state import (
    PositionState,
    ShortPositionState,
    add_daily_buy_notional,
    add_daily_realized_pnl,
    add_daily_short_open_notional,
    add_total_fees,
    add_total_realized_pnl,
    add_trade_outcome,
    append_trade_log,
    clear_pending_order,
    clear_short_position,
    get_daily_buy_notional,
    get_daily_realized_pnl,
    get_daily_short_open_notional,
    get_last_trade_ts,
    get_pending_order,
    get_short_position,
    get_total_fees,
    get_position,
    get_total_realized_pnl,
    increment_daily_trades,
    load_state,
    save_state,
    set_last_trade_ts,
    set_pending_order,
    set_position,
    set_short_position,
)
from .strategy import (
    Signal,
    generate_short_signal,
    generate_signal,
    generate_subminute_short_signal,
    generate_subminute_signal,
)


def _d(value: object) -> Decimal:
    return Decimal(str(value))


def _q4(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.0001")))


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _split_product(product_id: str) -> tuple[str, str]:
    if "-" not in product_id:
        raise ValueError(f"Unexpected product format: {product_id}")
    base, quote = product_id.split("-", 1)
    return base.upper(), quote.upper()


def _is_successful_order(response: dict[str, Any]) -> bool:
    if not isinstance(response, dict):
        return True
    if response.get("success") is False:
        return False
    if response.get("error_response"):
        return False
    if response.get("success_response"):
        return True
    return True


def _find_text_value(data: Any, candidates: tuple[str, ...]) -> str | None:
    lowered = {key.lower() for key in candidates}
    stack = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() in lowered and isinstance(value, str):
                    return value
                stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return None


def _find_decimal_value(data: Any, candidates: tuple[str, ...]) -> Decimal | None:
    lowered = {key.lower() for key in candidates}
    stack = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() in lowered:
                    try:
                        return _d(value)
                    except Exception:
                        pass
                stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return None


class _Ansi:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


@dataclass(slots=True)
class CycleResult:
    action: str
    reason: str
    mode: str
    product_id: str
    price: Decimal
    details: dict[str, Any]


class TradeBot:
    def __init__(self, config: BotConfig):
        self.config = config
        self.client = CoinbaseAdvancedClient(config.key_file)
        self.state = load_state(config.state_file, config.paper.starting_usd)
        self.metrics = load_metrics(config.metrics_file)
        self.product_id = config.product_id
        self.base_currency, self.quote_currency = _split_product(self.product_id)
        self._cached_closes: dict[tuple[str, str, int], tuple[datetime, list[float]]] = {}
        self._cached_fees: dict[str, tuple[datetime, FeeRates]] = {}
        self._cached_prices: dict[str, tuple[datetime, Decimal]] = {}
        self._cached_spreads: dict[str, tuple[datetime, Decimal, Decimal]] = {}
        self._last_auto_check_ts: dict[str, datetime] = {}
        self._tick_history: dict[str, list[tuple[datetime, float]]] = {}
        self._invalid_products: set[str] = set()
        self._last_printed_line: str | None = None
        self._price_source: dict[str, str] = {}
        self._ticker_feed: CoinbaseTickerWebSocketFeed | None = None
        self._ticker_feed_disabled_reason: str | None = None
        self._ticker_warning_printed = False

    def close(self) -> None:
        feed = self._ticker_feed
        if feed is not None:
            feed.stop()
            self._ticker_feed = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def run_loop(self, execute_live: bool) -> None:
        self._ensure_ticker_feed()
        if self._ticker_feed_disabled_reason and not self._ticker_warning_printed:
            print(f"[warn] websocket feed disabled: {self._ticker_feed_disabled_reason}")
            self._ticker_warning_printed = True
        while True:
            started = time.monotonic()
            try:
                result = self.run_cycle(execute_live=execute_live)
                line = self.format_cycle_result(result)
                if line != self._last_printed_line:
                    print(line)
                    self._last_printed_line = line
            except RuntimeError as exc:
                if self._is_transient_cycle_error(exc):
                    print(f"[warn] transient API error: {exc}")
                    time.sleep(max(float(self.config.loop_seconds), 5.0))
                    continue
                else:
                    raise
            elapsed = time.monotonic() - started
            sleep_for = max(0.0, float(self.config.loop_seconds) - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def status(self) -> dict[str, Any]:
        price = self._get_mid_price(self.product_id)
        price_source = self._price_source.get(self.product_id, "unknown")
        fees = self._get_fee_rates(product_type=self.config.product_type)
        position = get_position(self.state, self.product_id)
        short_position = get_short_position(self.state, self.config.auto_actions.short_product_id)
        pending_order = get_pending_order(self.state, self.product_id)
        balances = self._get_balances()
        summary = self._pnl_summary(price, position)
        daily_buy = self._daily_buy_snapshot()
        daily_short = self._daily_short_snapshot()
        return {
            "mode": self.config.mode,
            "product_id": self.product_id,
            "price": str(price),
            "price_source": price_source,
            "maker_fee_rate": str(fees.maker_rate),
            "taker_fee_rate": str(fees.taker_rate),
            "balances": {k: str(v) for k, v in balances.items()},
            "position": {
                "base_size": str(position.base_size),
                "avg_entry_price": str(position.avg_entry_price),
            },
            "short_position": {
                "product_id": self.config.auto_actions.short_product_id,
                "base_size": str(short_position.base_size),
                "avg_entry_price": str(short_position.avg_entry_price),
            },
            "pending_order": pending_order or {},
            "websocket": self._websocket_status(self.product_id),
            "performance": summary,
            "daily_buy": daily_buy,
            "daily_short_open": daily_short,
            "metrics_file": self.config.metrics_file,
        }

    def backtest_walk_forward(
        self,
        lookback_candles: int,
        train_candles: int,
        test_candles: int,
        step_candles: int,
        include_scenarios: bool = False,
        scenario_length: int = 1800,
        scenarios_only: bool = False,
    ) -> dict[str, Any]:
        if not scenarios_only and lookback_candles <= 0:
            raise ValueError("lookback_candles must be > 0")
        if include_scenarios and scenario_length <= 0:
            raise ValueError("scenario_length must be > 0 when include_scenarios is enabled")
        if scenarios_only and not include_scenarios:
            include_scenarios = True

        try:
            fees = self._get_fee_rates(product_type=self.config.product_type)
            maker_rate = float(fees.maker_rate)
            taker_rate = float(fees.taker_rate)
        except Exception:
            # Safe fallback to avoid blocking offline-style backtests.
            maker_rate = 0.006
            taker_rate = 0.012

        if scenarios_only:
            result: dict[str, Any] = {
                "summary": {},
                "windows": [],
                "meta": {
                    "product_id": self.product_id,
                    "granularity": self.config.granularity,
                    "candles_used": 0,
                    "train_candles": train_candles,
                    "test_candles": test_candles,
                    "step_candles": step_candles,
                    "scenarios_only": True,
                },
            }
        else:
            candles = self.client.recent_candles(
                self.product_id,
                self.config.granularity,
                lookback_candles,
            )
            closes = [float(candle["close"]) for candle in candles if "close" in candle]
            if len(closes) < (train_candles + test_candles + 2):
                raise RuntimeError(
                    f"Need at least {train_candles + test_candles + 2} candles; received {len(closes)}"
                )
            result = walk_forward_backtest(
                closes=closes,
                strategy=self.config.strategy,
                guardrails=self.config.guardrails,
                initial_usd=float(self.config.paper.starting_usd),
                maker_fee_rate=maker_rate,
                taker_fee_rate=taker_rate,
                prefer_maker_orders=bool(self.config.execution.prefer_maker_orders),
                train_candles=train_candles,
                test_candles=test_candles,
                step_candles=step_candles,
            )
            result["meta"] = {
                "product_id": self.product_id,
                "granularity": self.config.granularity,
                "candles_used": len(closes),
                "train_candles": train_candles,
                "test_candles": test_candles,
                "step_candles": step_candles,
                "scenarios_only": False,
            }

        if include_scenarios:
            result["scenario_suite"] = run_scenario_suite(
                strategy=self.config.strategy,
                guardrails=self.config.guardrails,
                initial_usd=float(self.config.paper.starting_usd),
                maker_fee_rate=maker_rate,
                taker_fee_rate=taker_rate,
                prefer_maker_orders=bool(self.config.execution.prefer_maker_orders),
                train_candles=train_candles,
                test_candles=test_candles,
                step_candles=step_candles,
                scenario_length=scenario_length,
            )
        return result

    def run_cycle(self, execute_live: bool = False) -> CycleResult:
        if self.product_id not in self.config.guardrails.allowed_products:
            raise RuntimeError(
                f"Product {self.product_id} is not in guardrails.allowed_products"
            )

        closes = self._get_closes()
        price = self._get_mid_price(self.product_id)
        self._record_tick(self.product_id, price)
        fees = self._get_fee_rates()
        roundtrip_fee_bps = float(fees.roundtrip_bps) + self.config.strategy.slippage_buffer_bps
        spread_bps = self._current_spread_bps(self.product_id)
        pending_info = self._reconcile_pending_spot_order()

        position = get_position(self.state, self.product_id)
        candle_signal = generate_signal(
            closes=closes,
            roundtrip_fee_bps=roundtrip_fee_bps,
            settings=self.config.strategy,
            has_open_long=position.base_size > 0,
        )
        signal = candle_signal
        signal_source = "candle"
        subminute_signal: Signal | None = None
        subminute_prices: list[float] = []
        if self.config.strategy.enable_subminute_signals:
            subminute_prices = self._get_subminute_prices(self.product_id)
            subminute_signal = generate_subminute_signal(
                prices=subminute_prices,
                roundtrip_fee_bps=roundtrip_fee_bps,
                settings=self.config.strategy,
                has_open_long=position.base_size > 0,
            )
            if signal.action == "HOLD" and subminute_signal.action in {"BUY", "SELL"}:
                signal = subminute_signal
                signal_source = "subminute"
        signal = self._apply_stop_rules(signal, position, price)

        details = {
            "signal": signal.action,
            "signal_reason": signal.reason,
            "signal_source": signal_source,
            "price_source": self._price_source.get(self.product_id, "unknown"),
            "trend_bps": round(signal.trend_bps, 2),
            "rsi": round(signal.rsi, 2),
            "expected_edge_bps": round(signal.expected_edge_bps, 2),
            "signal_threshold_bps": round(signal.threshold_bps, 2),
            "signal_volatility_bps": round(signal.volatility_bps, 2),
            "signal_regime": signal.regime,
            "maker_fee_bps": round(float(fees.maker_rate * Decimal("10000")), 2),
            "taker_fee_bps": round(float(fees.taker_rate * Decimal("10000")), 2),
            "roundtrip_fee_bps": round(roundtrip_fee_bps, 2),
            "spread_bps": round(spread_bps, 2) if spread_bps is not None else None,
            "max_spread_bps": round(float(self.config.execution.max_spread_bps), 2),
            "candle_signal": candle_signal.action,
            "candle_signal_reason": candle_signal.reason,
            "candle_trend_bps": round(candle_signal.trend_bps, 2),
            "candle_rsi": round(candle_signal.rsi, 2),
            "candle_expected_edge_bps": round(candle_signal.expected_edge_bps, 2),
            "candle_threshold_bps": round(candle_signal.threshold_bps, 2),
            "candle_volatility_bps": round(candle_signal.volatility_bps, 2),
            "candle_regime": candle_signal.regime,
        }
        if subminute_signal is not None:
            details.update(
                {
                    "subminute_signal": subminute_signal.action,
                    "subminute_signal_reason": subminute_signal.reason,
                    "subminute_trend_bps": round(subminute_signal.trend_bps, 2),
                    "subminute_rsi": round(subminute_signal.rsi, 2),
                    "subminute_expected_edge_bps": round(subminute_signal.expected_edge_bps, 2),
                    "subminute_threshold_bps": round(subminute_signal.threshold_bps, 2),
                    "subminute_volatility_bps": round(subminute_signal.volatility_bps, 2),
                    "subminute_regime": subminute_signal.regime,
                    "subminute_ticks": len(subminute_prices),
                }
            )
        details.update(pending_info)

        action = "HOLD"
        reason = signal.reason
        if signal.action == "BUY":
            action, reason, extra = self._attempt_buy(
                signal=signal,
                position=position,
                price=price,
                balances=self._get_balances(),
                maker_fee_rate=fees.maker_rate,
                taker_fee_rate=fees.taker_rate,
                spread_bps=spread_bps,
                execute_live=execute_live,
            )
            details.update(extra)
        elif signal.action == "SELL":
            action, reason, extra = self._attempt_sell(
                signal=signal,
                position=position,
                price=price,
                balances=self._get_balances(),
                maker_fee_rate=fees.maker_rate,
                taker_fee_rate=fees.taker_rate,
                spread_bps=spread_bps,
                execute_live=execute_live,
            )
            details.update(extra)
        else:
            action, auto_reason, extra = self._attempt_auto_actions(execute_live=execute_live)
            details.update(extra)
            if action == "HOLD":
                if auto_reason and auto_reason != "no auto action executed":
                    reason = f"{signal.reason} | {auto_reason}"
                else:
                    reason = signal.reason
            else:
                reason = auto_reason

        position_after = get_position(self.state, self.product_id)
        short_after = get_short_position(self.state, self.config.auto_actions.short_product_id)
        summary = self._pnl_summary(price, position_after)
        details.update(
            {
                "realized_pnl_usd": summary["realized_pnl_usd"],
                "unrealized_pnl_usd": summary["unrealized_pnl_usd"],
                "net_profit_usd": summary["net_profit_usd"],
                "total_fees_usd": summary["total_fees_usd"],
                "short_position_base": _q4(short_after.base_size),
            }
        )
        details.update(self._daily_buy_snapshot())
        details.update(self._daily_short_snapshot())
        result = CycleResult(
            action=action,
            reason=reason,
            mode=self.config.mode,
            product_id=self.product_id,
            price=price,
            details=details,
        )
        self._record_cycle_point(price, result.action, result.reason, position_after)
        self._persist()
        return result

    def convert(
        self,
        from_currency: str,
        to_currency: str,
        amount: Decimal,
        execute_live: bool,
    ) -> dict[str, Any]:
        pair_key = f"{from_currency.upper()}:{to_currency.upper()}"
        if not self.config.guardrails.allow_convert:
            raise RuntimeError("Convert is disabled by guardrails")
        if pair_key not in self.config.guardrails.allowed_convert_pairs:
            raise RuntimeError(f"Convert pair {pair_key} is not allowlisted")
        account_map = self.client.currency_account_map()
        from_account = account_map.get(from_currency.upper())
        to_account = account_map.get(to_currency.upper())
        if not from_account or not to_account:
            raise RuntimeError("Could not find account IDs for requested convert currencies")
        quote = self.client.create_convert_quote(
            from_account=from_account,
            to_account=to_account,
            amount=amount,
        )
        trade_id = _find_text_value(quote, ("trade_id", "quote_id", "id"))
        if not execute_live or not self._live_orders_allowed(execute_live):
            return {"action": "CONVERT_PREVIEW", "quote": quote, "executed": False}
        if not trade_id:
            raise RuntimeError("Convert quote did not return a trade_id")
        commit = self.client.commit_convert(
            trade_id=trade_id,
            from_account=from_account,
            to_account=to_account,
        )
        event = {
            "ts": _now_utc().isoformat(),
            "action": "CONVERT",
            "mode": self.config.mode,
            "pair": pair_key,
            "amount": str(amount),
            "realized_pnl_usd": "0",
            "fee_usd": "0",
        }
        append_trade_log(
            self.state,
            {
                **event,
                "response": commit,
            },
        )
        append_trade(self.metrics, event)
        self._persist()
        return {"action": "CONVERT", "executed": True, "commit": commit}

    def open_short(
        self,
        product_id: str,
        base_size: Decimal,
        leverage: Decimal,
        execute_live: bool,
        margin_type: str = "CROSS",
    ) -> dict[str, Any]:
        if not self.config.guardrails.allow_short:
            raise RuntimeError("Shorting is disabled by guardrails")
        if "-PERP" not in product_id and "-FUT" not in product_id:
            raise RuntimeError("Shorting is restricted to derivative products (PERP/FUT)")
        if leverage > _d(self.config.guardrails.max_short_leverage):
            raise RuntimeError("Requested leverage exceeds max_short_leverage")
        mark_price = self._get_mid_price(product_id)
        notional = base_size * mark_price
        reasons = short_open_checks(
            state=self.state,
            guardrails=self.config.guardrails,
            notional_usd=notional,
            leverage=leverage,
        )
        if reasons:
            raise RuntimeError(f"Short blocked by guardrails: {', '.join(reasons)}")
        if not execute_live or not self._live_orders_allowed(execute_live):
            return {
                "action": "SHORT_PREVIEW",
                "product_id": product_id,
                "base_size": str(base_size),
                "notional_usd": str(notional),
                "executed": False,
            }
        response = self.client.open_short(
            product_id=product_id,
            base_size=base_size,
            leverage=leverage,
            margin_type=margin_type,
        )
        event = {
            "ts": _now_utc().isoformat(),
            "action": "SHORT_OPEN",
            "mode": self.config.mode,
            "product_id": product_id,
            "side": "SHORT_OPEN",
            "base_size": str(base_size),
            "quote_notional_usd": str(notional),
            "price_usd": str(mark_price),
            "fee_usd": "0",
            "realized_pnl_usd": "0",
        }
        position = get_short_position(self.state, product_id)
        old_notional = position.base_size * position.avg_entry_price
        new_base = position.base_size + base_size
        position.base_size = new_base
        position.avg_entry_price = (
            (old_notional + (base_size * mark_price)) / new_base if new_base > 0 else Decimal("0")
        )
        set_short_position(self.state, product_id, position)
        add_daily_short_open_notional(self.state, notional)
        append_trade_log(
            self.state,
            {
                **event,
                "leverage": str(leverage),
                "response": response,
            },
        )
        append_trade(self.metrics, event)
        increment_daily_trades(self.state)
        set_last_trade_ts(self.state, _now_utc())
        self._persist()
        return {"action": "SHORT_OPEN", "executed": True, "response": response}

    def close_position(
        self,
        product_id: str,
        size: Decimal | None,
        execute_live: bool,
    ) -> dict[str, Any]:
        if not execute_live or not self._live_orders_allowed(execute_live):
            return {
                "action": "CLOSE_POSITION_PREVIEW",
                "product_id": product_id,
                "size": str(size) if size is not None else "all",
                "executed": False,
            }
        short_position = get_short_position(self.state, product_id)
        requested_size = size if size is not None else short_position.base_size
        close_size = min(requested_size, short_position.base_size)
        if close_size <= 0:
            return {
                "action": "CLOSE_POSITION_PREVIEW",
                "product_id": product_id,
                "size": str(size) if size is not None else "all",
                "executed": False,
                "reason": "no short position size available to close",
            }
        response_size = close_size if size is not None else None
        response = self.client.close_position(product_id=product_id, size=response_size)
        mark_price = self._get_mid_price(product_id)
        fees = self._get_fee_rates(product_type=self.config.auto_actions.short_product_type)
        estimated_fee = (close_size * mark_price) * fees.taker_rate if close_size > 0 else Decimal("0")
        realized = (
            (short_position.avg_entry_price - mark_price) * close_size - estimated_fee
            if close_size > 0
            else Decimal("0")
        )
        if close_size > 0:
            remaining = max(Decimal("0"), short_position.base_size - close_size)
            if remaining <= Decimal("0.00000001"):
                clear_short_position(self.state, product_id)
            else:
                short_position.base_size = remaining
                set_short_position(self.state, product_id, short_position)
            add_daily_realized_pnl(self.state, realized)
            add_total_realized_pnl(self.state, realized)
            add_total_fees(self.state, estimated_fee)
            add_trade_outcome(self.state, realized)
        event = {
            "ts": _now_utc().isoformat(),
            "action": "CLOSE_POSITION",
            "mode": self.config.mode,
            "product_id": product_id,
            "side": "CLOSE_SHORT",
            "size": str(close_size) if size is not None else "all",
            "fee_usd": str(estimated_fee),
            "realized_pnl_usd": str(realized),
            "price_usd": str(mark_price),
        }
        append_trade_log(
            self.state,
            {
                **event,
                "response": response,
            },
        )
        append_trade(self.metrics, event)
        increment_daily_trades(self.state)
        set_last_trade_ts(self.state, _now_utc())
        self._persist()
        return {"action": "CLOSE_POSITION", "executed": True, "response": response}

    def _apply_stop_rules(
        self, signal: Signal, position: PositionState, price: Decimal
    ) -> Signal:
        if position.base_size <= 0 or position.avg_entry_price <= 0:
            return signal
        pnl_pct = (price - position.avg_entry_price) / position.avg_entry_price
        if pnl_pct <= -_d(self.config.strategy.stop_loss_pct):
            return Signal(
                action="SELL",
                reason=f"stop loss triggered ({float(pnl_pct):.2%})",
                expected_edge_bps=signal.expected_edge_bps,
                trend_bps=signal.trend_bps,
                rsi=signal.rsi,
                threshold_bps=signal.threshold_bps,
                volatility_bps=signal.volatility_bps,
                regime=signal.regime,
            )
        if pnl_pct >= _d(self.config.strategy.take_profit_pct):
            return Signal(
                action="SELL",
                reason=f"take profit triggered ({float(pnl_pct):.2%})",
                expected_edge_bps=signal.expected_edge_bps,
                trend_bps=signal.trend_bps,
                rsi=signal.rsi,
                threshold_bps=signal.threshold_bps,
                volatility_bps=signal.volatility_bps,
                regime=signal.regime,
            )
        return signal

    def _apply_short_stop_rules(
        self, signal: Signal, position: ShortPositionState, price: Decimal
    ) -> Signal:
        if position.base_size <= 0 or position.avg_entry_price <= 0:
            return signal
        pnl_pct = (position.avg_entry_price - price) / position.avg_entry_price
        if pnl_pct <= -_d(self.config.strategy.stop_loss_pct):
            return Signal(
                action="CLOSE_SHORT",
                reason=f"short stop loss triggered ({float(pnl_pct):.2%})",
                expected_edge_bps=signal.expected_edge_bps,
                trend_bps=signal.trend_bps,
                rsi=signal.rsi,
                threshold_bps=signal.threshold_bps,
                volatility_bps=signal.volatility_bps,
                regime=signal.regime,
            )
        if pnl_pct >= _d(self.config.strategy.take_profit_pct):
            return Signal(
                action="CLOSE_SHORT",
                reason=f"short take profit triggered ({float(pnl_pct):.2%})",
                expected_edge_bps=signal.expected_edge_bps,
                trend_bps=signal.trend_bps,
                rsi=signal.rsi,
                threshold_bps=signal.threshold_bps,
                volatility_bps=signal.volatility_bps,
                regime=signal.regime,
            )
        return signal

    def _attempt_auto_actions(self, execute_live: bool) -> tuple[str, str, dict[str, Any]]:
        pending = get_pending_order(self.state, self.product_id)
        if pending is not None:
            return (
                "HOLD",
                "spot maker order pending fill",
                {"pending_order_id": str(pending.get("order_id", ""))},
            )
        auto = self.config.auto_actions
        attempts: list[tuple[str, Any]] = []
        if auto.enable_auto_close_position:
            attempts.append(("auto_close", self._attempt_auto_close_short))
        if auto.enable_auto_short:
            attempts.append(("auto_short", self._attempt_auto_short))
        if auto.enable_auto_convert:
            attempts.append(("auto_convert", self._attempt_auto_convert))
        if not attempts:
            return "HOLD", "auto actions disabled", {}

        hold_reasons: dict[str, str] = {}
        for name, fn in attempts:
            try:
                action, reason, details = fn(execute_live=execute_live)
            except Exception as exc:
                action, reason, details = "HOLD", f"{name} unavailable: {exc}", {}
            if action != "HOLD":
                for prior_name, prior_reason in hold_reasons.items():
                    details.setdefault(f"{prior_name}_reason", prior_reason)
                details.setdefault("auto_action", name)
                return action, reason, details
            hold_reasons[name] = reason
        return (
            "HOLD",
            "auto actions considered; none executed",
            {
                "auto_close_reason": hold_reasons.get("auto_close", "not enabled"),
                "auto_short_reason": hold_reasons.get("auto_short", "not enabled"),
                "auto_convert_reason": hold_reasons.get("auto_convert", "not enabled"),
            },
        )

    def _attempt_auto_short(self, execute_live: bool) -> tuple[str, str, dict[str, Any]]:
        auto = self.config.auto_actions
        if not self.config.guardrails.allow_short:
            return "HOLD", "auto short disabled by guardrails.allow_short=false", {}
        if self._trade_cooldown_active():
            return "HOLD", "auto short blocked by cooldown", {}
        product_id = auto.short_product_id
        if product_id in self._invalid_products:
            return "HOLD", f"auto short product {product_id} disabled after invalid-product API error", {}
        if "-PERP" not in product_id and "-FUT" not in product_id:
            return "HOLD", "auto short requires derivative product id (PERP/FUT)", {}
        position = get_short_position(self.state, product_id)
        if position.base_size > 0:
            return "HOLD", "short position already open", {}
        if self._auto_check_throttled("auto_short", auto.short_check_interval_seconds):
            return "HOLD", "auto short check throttled", {}

        try:
            price = self._get_mid_price(product_id)
        except RuntimeError as exc:
            if self._is_invalid_product_error(exc):
                self._invalid_products.add(product_id)
                return (
                    "HOLD",
                    f"auto short product {product_id} is invalid or unsupported for this account",
                    {},
                )
            return "HOLD", f"auto short market data unavailable: {exc}", {}
        self._record_tick(product_id, price)
        order_notional = min(
            _d(auto.short_order_usd),
            _d(self.config.guardrails.max_short_notional_usd),
        )
        leverage = _d(auto.short_leverage)
        reasons = short_open_checks(
            state=self.state,
            guardrails=self.config.guardrails,
            notional_usd=order_notional,
            leverage=leverage,
        )
        if reasons:
            return "HOLD", f"auto short blocked by guardrails: {', '.join(reasons)}", {}

        try:
            closes = self._get_closes(product_id=product_id)
        except RuntimeError as exc:
            if self._is_invalid_product_error(exc):
                self._invalid_products.add(product_id)
                return (
                    "HOLD",
                    f"auto short product {product_id} is invalid or unsupported for this account",
                    {},
                )
            return "HOLD", f"auto short candles unavailable: {exc}", {}
        fees = self._get_fee_rates(product_type=auto.short_product_type)
        roundtrip_fee_bps = float(fees.roundtrip_bps) + self.config.strategy.slippage_buffer_bps
        signal = generate_short_signal(
            closes=closes,
            roundtrip_fee_bps=roundtrip_fee_bps,
            settings=self.config.strategy,
            has_open_short=False,
        )
        short_signal_source = "candle"
        short_subminute_signal: Signal | None = None
        short_subminute_prices: list[float] = []
        if self.config.strategy.enable_subminute_signals:
            short_subminute_prices = self._get_subminute_prices(product_id)
            short_subminute_signal = generate_subminute_short_signal(
                prices=short_subminute_prices,
                roundtrip_fee_bps=roundtrip_fee_bps,
                settings=self.config.strategy,
                has_open_short=False,
            )
            if signal.action == "HOLD" and short_subminute_signal.action == "SHORT":
                signal = short_subminute_signal
                short_signal_source = "subminute"
        if signal.action != "SHORT":
            details = {
                "short_signal_source": short_signal_source,
                "short_trend_bps": round(signal.trend_bps, 2),
                "short_rsi": round(signal.rsi, 2),
                "short_expected_edge_bps": round(signal.expected_edge_bps, 2),
            }
            if short_subminute_signal is not None:
                details.update(
                    {
                        "short_subminute_signal": short_subminute_signal.action,
                        "short_subminute_signal_reason": short_subminute_signal.reason,
                        "short_subminute_ticks": len(short_subminute_prices),
                    }
                )
            return (
                "HOLD",
                signal.reason,
                details,
            )

        expected_gross = order_notional * _d(signal.expected_edge_bps) / Decimal("10000")
        estimated_fees = order_notional * fees.taker_rate * Decimal("2")
        expected_net = expected_gross - estimated_fees
        if expected_net < _d(auto.min_expected_short_profit_usd):
            return (
                "HOLD",
                "auto short edge does not clear min_expected_short_profit_usd",
                {
                    "short_signal_source": short_signal_source,
                    "short_order_usd": str(order_notional),
                    "short_expected_net_usd": _q4(expected_net),
                },
            )

        base_size = order_notional / max(price, Decimal("0.00000001"))
        if not self._live_orders_allowed(execute_live):
            return (
                "HOLD",
                "auto short preview only (set --execute-live and TRADEBOT_ENABLE_LIVE=true)",
                {
                    "short_signal_source": short_signal_source,
                    "short_order_usd": str(order_notional),
                    "short_base_size": str(base_size),
                    "short_expected_net_usd": _q4(expected_net),
                },
            )

        response = self.client.open_short(
            product_id=product_id,
            base_size=base_size,
            leverage=leverage,
            margin_type=auto.short_margin_type,
        )
        if not _is_successful_order(response):
            return "HOLD", "auto short failed at exchange", {"short_response": response}

        short_position = ShortPositionState(base_size=base_size, avg_entry_price=price)
        set_short_position(self.state, product_id, short_position)
        add_daily_short_open_notional(self.state, order_notional)
        increment_daily_trades(self.state)
        set_last_trade_ts(self.state, _now_utc())
        event = {
            "ts": _now_utc().isoformat(),
            "action": "SHORT_OPEN",
            "mode": self.config.mode,
            "product_id": product_id,
            "side": "SHORT_OPEN",
            "price_usd": str(price),
            "base_size": str(base_size),
            "quote_notional_usd": str(order_notional),
            "fee_usd": "0",
            "realized_pnl_usd": "0",
            "signal_reason": signal.reason,
            "trend_bps": round(signal.trend_bps, 4),
            "rsi": round(signal.rsi, 4),
            "expected_edge_bps": round(signal.expected_edge_bps, 4),
        }
        append_trade_log(self.state, {**event, "response": response})
        append_trade(self.metrics, event)
        return (
            "SHORT_OPEN",
            "auto short opened",
            {
                "short_signal_source": short_signal_source,
                "short_order_usd": str(order_notional),
                "short_base_size": str(base_size),
                "short_expected_net_usd": _q4(expected_net),
            },
        )

    def _attempt_auto_close_short(self, execute_live: bool) -> tuple[str, str, dict[str, Any]]:
        auto = self.config.auto_actions
        product_id = auto.short_product_id
        if product_id in self._invalid_products:
            return "HOLD", f"auto close skipped because product {product_id} is disabled", {}
        position = get_short_position(self.state, product_id)
        if position.base_size <= 0 or position.avg_entry_price <= 0:
            return "HOLD", "no short position to close", {}
        if self._auto_check_throttled("auto_close", auto.close_check_interval_seconds):
            return "HOLD", "auto close check throttled", {}

        try:
            price = self._get_mid_price(product_id)
        except RuntimeError as exc:
            if self._is_invalid_product_error(exc):
                self._invalid_products.add(product_id)
                return (
                    "HOLD",
                    f"auto close product {product_id} is invalid or unsupported for this account",
                    {},
                )
            return "HOLD", f"auto close market data unavailable: {exc}", {}
        self._record_tick(product_id, price)
        try:
            closes = self._get_closes(product_id=product_id)
        except RuntimeError as exc:
            if self._is_invalid_product_error(exc):
                self._invalid_products.add(product_id)
                return (
                    "HOLD",
                    f"auto close product {product_id} is invalid or unsupported for this account",
                    {},
                )
            return "HOLD", f"auto close candles unavailable: {exc}", {}
        fees = self._get_fee_rates(product_type=auto.short_product_type)
        roundtrip_fee_bps = float(fees.roundtrip_bps) + self.config.strategy.slippage_buffer_bps
        signal = generate_short_signal(
            closes=closes,
            roundtrip_fee_bps=roundtrip_fee_bps,
            settings=self.config.strategy,
            has_open_short=True,
        )
        short_signal_source = "candle"
        short_subminute_signal: Signal | None = None
        short_subminute_prices: list[float] = []
        if self.config.strategy.enable_subminute_signals:
            short_subminute_prices = self._get_subminute_prices(product_id)
            short_subminute_signal = generate_subminute_short_signal(
                prices=short_subminute_prices,
                roundtrip_fee_bps=roundtrip_fee_bps,
                settings=self.config.strategy,
                has_open_short=True,
            )
            if signal.action == "HOLD" and short_subminute_signal.action == "CLOSE_SHORT":
                signal = short_subminute_signal
                short_signal_source = "subminute"
        signal = self._apply_short_stop_rules(signal, position, price)
        if signal.reason.startswith("short stop loss") or signal.reason.startswith("short take profit"):
            short_signal_source = "risk_stop"
        if signal.action != "CLOSE_SHORT":
            details = {
                "short_signal_source": short_signal_source,
                "short_trend_bps": round(signal.trend_bps, 2),
                "short_rsi": round(signal.rsi, 2),
                "short_expected_edge_bps": round(signal.expected_edge_bps, 2),
            }
            if short_subminute_signal is not None:
                details.update(
                    {
                        "short_subminute_signal": short_subminute_signal.action,
                        "short_subminute_signal_reason": short_subminute_signal.reason,
                        "short_subminute_ticks": len(short_subminute_prices),
                    }
                )
            return (
                "HOLD",
                signal.reason,
                details,
            )

        close_fee = (position.base_size * price) * fees.taker_rate
        estimated_realized = ((position.avg_entry_price - price) * position.base_size) - close_fee
        forced = signal.reason.startswith("short stop loss") or signal.reason.startswith(
            "short take profit"
        )
        if not forced and estimated_realized < _d(auto.min_expected_short_profit_usd):
            return (
                "HOLD",
                "auto short close edge does not clear min_expected_short_profit_usd",
                {
                    "short_signal_source": short_signal_source,
                    "short_estimated_realized_pnl_usd": _q4(estimated_realized),
                },
            )

        if not self._live_orders_allowed(execute_live):
            return (
                "HOLD",
                "auto short close preview only (set --execute-live and TRADEBOT_ENABLE_LIVE=true)",
                {
                    "short_signal_source": short_signal_source,
                    "short_estimated_realized_pnl_usd": _q4(estimated_realized),
                },
            )

        response = self.client.close_position(product_id=product_id, size=position.base_size)
        if not _is_successful_order(response):
            return "HOLD", "auto short close failed at exchange", {"short_response": response}

        clear_short_position(self.state, product_id)
        add_daily_realized_pnl(self.state, estimated_realized)
        add_total_realized_pnl(self.state, estimated_realized)
        add_total_fees(self.state, close_fee)
        add_trade_outcome(self.state, estimated_realized)
        increment_daily_trades(self.state)
        set_last_trade_ts(self.state, _now_utc())
        event = {
            "ts": _now_utc().isoformat(),
            "action": "CLOSE_POSITION",
            "mode": self.config.mode,
            "product_id": product_id,
            "side": "CLOSE_SHORT",
            "price_usd": str(price),
            "base_size": str(position.base_size),
            "quote_notional_usd": str(position.base_size * price),
            "fee_usd": str(close_fee),
            "realized_pnl_usd": str(estimated_realized),
            "signal_reason": signal.reason,
            "trend_bps": round(signal.trend_bps, 4),
            "rsi": round(signal.rsi, 4),
            "expected_edge_bps": round(signal.expected_edge_bps, 4),
        }
        append_trade_log(self.state, {**event, "response": response})
        append_trade(self.metrics, event)
        return (
            "CLOSE_POSITION",
            "auto short closed",
            {
                "short_signal_source": short_signal_source,
                "short_realized_pnl_usd": _q4(estimated_realized),
            },
        )

    def _attempt_auto_convert(self, execute_live: bool) -> tuple[str, str, dict[str, Any]]:
        auto = self.config.auto_actions
        if not self.config.guardrails.allow_convert:
            return "HOLD", "auto convert disabled by guardrails.allow_convert=false", {}
        profit_target = _d(self.config.guardrails.daily_profit_target_usd)
        if profit_target > 0 and get_daily_realized_pnl(self.state) >= profit_target:
            return "HOLD", "auto convert blocked by daily_profit_target_usd", {}
        if self._trade_cooldown_active():
            return "HOLD", "auto convert blocked by cooldown", {}
        if self._auto_check_throttled("auto_convert", auto.convert_check_interval_seconds):
            return "HOLD", "auto convert check throttled", {}
        pair = auto.convert_pair.upper().strip()
        if ":" not in pair:
            return "HOLD", "auto convert pair must be FROM:TO", {}
        if pair not in self.config.guardrails.allowed_convert_pairs:
            return "HOLD", f"auto convert pair {pair} is not allowlisted", {}
        from_currency, to_currency = pair.split(":", 1)
        amount = _d(auto.convert_amount)
        account_map = self.client.currency_account_map()
        from_account = account_map.get(from_currency)
        to_account = account_map.get(to_currency)
        if not from_account or not to_account:
            return "HOLD", "auto convert accounts not found", {}

        quote = self.client.create_convert_quote(
            from_account=from_account,
            to_account=to_account,
            amount=amount,
        )
        trade_id = _find_text_value(quote, ("trade_id", "quote_id", "id"))
        from_amount = _find_decimal_value(quote, ("from_amount", "source_amount")) or amount
        to_amount = _find_decimal_value(
            quote,
            ("to_amount", "destination_amount", "converted_amount", "amount_received"),
        )
        if to_amount is None:
            return "HOLD", "auto convert quote parse failed", {}
        expected_profit = to_amount - from_amount
        if expected_profit < _d(auto.min_convert_profit_usd):
            return (
                "HOLD",
                "auto convert edge does not clear min_convert_profit_usd",
                {"convert_expected_profit_usd": _q4(expected_profit)},
            )
        if not trade_id:
            return "HOLD", "auto convert quote missing trade_id", {}
        if not self._live_orders_allowed(execute_live):
            return (
                "HOLD",
                "auto convert preview only (set --execute-live and TRADEBOT_ENABLE_LIVE=true)",
                {"convert_expected_profit_usd": _q4(expected_profit)},
            )

        commit = self.client.commit_convert(
            trade_id=trade_id,
            from_account=from_account,
            to_account=to_account,
        )
        increment_daily_trades(self.state)
        set_last_trade_ts(self.state, _now_utc())
        event = {
            "ts": _now_utc().isoformat(),
            "action": "CONVERT",
            "mode": self.config.mode,
            "pair": pair,
            "amount": str(amount),
            "fee_usd": "0",
            "realized_pnl_usd": "0",
            "expected_profit_usd": str(expected_profit),
        }
        append_trade_log(self.state, {**event, "response": commit})
        append_trade(self.metrics, event)
        return (
            "CONVERT",
            "auto convert committed",
            {"convert_expected_profit_usd": _q4(expected_profit)},
        )

    def _attempt_buy(
        self,
        signal: Signal,
        position: PositionState,
        price: Decimal,
        balances: dict[str, Decimal],
        maker_fee_rate: Decimal,
        taker_fee_rate: Decimal,
        spread_bps: float | None,
        execute_live: bool,
    ) -> tuple[str, str, dict[str, Any]]:
        usd_available = balances.get(self.quote_currency, Decimal("0"))
        max_order = _d(self.config.guardrails.max_order_usd)
        max_position = _d(self.config.guardrails.max_position_usd)
        remaining_position_cap = max_position - (position.base_size * price)
        reserved = _d(self.config.guardrails.min_usd_reserve)
        max_spendable = usd_available - reserved
        order_usd = min(max_order, remaining_position_cap, max_spendable)
        context = RiskContext(
            now=_now_utc(),
            usd_available=usd_available,
            base_available=balances.get(self.base_currency, Decimal("0")),
            position_base=position.base_size,
            price=price,
        )
        if spread_bps is not None and spread_bps > float(self.config.execution.max_spread_bps):
            return (
                "HOLD",
                (
                    "spread too wide for entry "
                    f"({spread_bps:.2f} bps > {float(self.config.execution.max_spread_bps):.2f} bps)"
                ),
                {
                    "spread_bps": round(spread_bps, 2),
                    "max_spread_bps": round(float(self.config.execution.max_spread_bps), 2),
                },
            )
        reasons = buy_checks(self.state, self.config.guardrails, context, order_usd)
        if reasons:
            return "HOLD", f"blocked by guardrails: {', '.join(reasons)}", {"order_usd": str(order_usd)}
        if get_pending_order(self.state, self.product_id):
            pending = get_pending_order(self.state, self.product_id) or {}
            return (
                "HOLD",
                "spot maker order pending fill",
                {"pending_order_id": str(pending.get("order_id", ""))},
            )

        if self.config.execution.prefer_maker_orders:
            maker_total_fees = order_usd * maker_fee_rate * Decimal("2")
            expected_gross = order_usd * _d(signal.expected_edge_bps) / Decimal("10000")
            expected_net = expected_gross - maker_total_fees
            if expected_net < _d(self.config.strategy.min_expected_profit_usd):
                return (
                    "HOLD",
                    "expected net edge does not clear min_expected_profit_usd",
                    {
                        "order_usd": str(order_usd),
                        "expected_net_usd": str(expected_net.quantize(Decimal("0.0001"))),
                        "estimated_fees_usd": str(maker_total_fees.quantize(Decimal("0.0001"))),
                    },
                )
            offset = _d(self.config.execution.maker_price_offset_bps) / Decimal("10000")
            limit_price = price * (Decimal("1") - offset)
            limit_price = max(limit_price, Decimal("0.00000001"))
            base_size = order_usd / limit_price
            return self._submit_spot_maker_order(
                side="BUY",
                signal=signal,
                base_size=base_size,
                quote_notional=order_usd,
                limit_price=limit_price,
                maker_fee_rate=maker_fee_rate,
                execute_live=execute_live,
            )

        preview_buy = self.client.preview_market_buy(self.product_id, order_usd)
        buy_fee = self.client.preview_commission(preview_buy)
        if buy_fee <= 0 and taker_fee_rate > 0:
            buy_fee = order_usd * taker_fee_rate
        est_base = (order_usd - buy_fee) / max(price, Decimal("0.00000001"))
        if est_base <= 0:
            return (
                "HOLD",
                "estimated buy size non-positive after fees",
                {"order_usd": str(order_usd), "estimated_buy_fee_usd": str(buy_fee)},
            )
        preview_sell = self.client.preview_market_sell(self.product_id, est_base)
        sell_fee = self.client.preview_commission(preview_sell)
        total_fees = buy_fee + sell_fee
        expected_gross = order_usd * _d(signal.expected_edge_bps) / Decimal("10000")
        expected_net = expected_gross - total_fees
        if expected_net < _d(self.config.strategy.min_expected_profit_usd):
            return (
                "HOLD",
                "expected net edge does not clear min_expected_profit_usd",
                {
                    "order_usd": str(order_usd),
                    "expected_net_usd": str(expected_net.quantize(Decimal("0.0001"))),
                    "estimated_fees_usd": str(total_fees.quantize(Decimal("0.0001"))),
                },
            )

        if self.config.mode == "paper":
            fill_price = price * (
                Decimal("1") + (_d(self.config.strategy.slippage_buffer_bps) / Decimal("10000"))
            )
            executed_base = (order_usd - buy_fee) / fill_price
            self._paper_apply_buy(order_usd, executed_base)
            self._update_long_position_after_buy(position, executed_base, order_usd)
            add_daily_buy_notional(self.state, order_usd)
            add_total_fees(self.state, buy_fee)
            increment_daily_trades(self.state)
            set_last_trade_ts(self.state, _now_utc())
            event = {
                "ts": _now_utc().isoformat(),
                "action": "BUY",
                "mode": "paper",
                "product_id": self.product_id,
                "side": "BUY",
                "price_usd": str(fill_price),
                "base_size": str(executed_base),
                "quote_notional_usd": str(order_usd),
                "fee_usd": str(buy_fee),
                "realized_pnl_usd": "0",
                "signal_reason": signal.reason,
                "trend_bps": round(signal.trend_bps, 4),
                "rsi": round(signal.rsi, 4),
                "expected_edge_bps": round(signal.expected_edge_bps, 4),
            }
            append_trade_log(
                self.state,
                {
                    **event,
                },
            )
            append_trade(self.metrics, event)
            return (
                "BUY",
                "paper buy executed",
                {
                    "order_usd": str(order_usd),
                    "base_size": str(executed_base),
                    "estimated_fees_usd": str(total_fees.quantize(Decimal("0.0001"))),
                },
            )

        if not self._live_orders_allowed(execute_live):
            return (
                "HOLD",
                "live buy preview only (set --execute-live and TRADEBOT_ENABLE_LIVE=true)",
                {
                    "order_usd": str(order_usd),
                    "expected_net_usd": str(expected_net.quantize(Decimal("0.0001"))),
                    "estimated_fees_usd": str(total_fees.quantize(Decimal("0.0001"))),
                },
            )

        response = self.client.market_buy(self.product_id, order_usd)
        if not _is_successful_order(response):
            append_trade_log(
                self.state,
                {"action": "BUY_FAILED", "product_id": self.product_id, "response": response},
            )
            return "HOLD", "buy failed at exchange", {"response": response}
        executed_base = (order_usd - buy_fee) / max(price, Decimal("0.00000001"))
        self._update_long_position_after_buy(position, executed_base, order_usd)
        add_daily_buy_notional(self.state, order_usd)
        add_total_fees(self.state, buy_fee)
        event = {
            "ts": _now_utc().isoformat(),
            "action": "BUY",
            "mode": "live",
            "product_id": self.product_id,
            "side": "BUY",
            "price_usd": str(price),
            "base_size": str(executed_base),
            "quote_notional_usd": str(order_usd),
            "fee_usd": str(buy_fee),
            "realized_pnl_usd": "0",
            "signal_reason": signal.reason,
            "trend_bps": round(signal.trend_bps, 4),
            "rsi": round(signal.rsi, 4),
            "expected_edge_bps": round(signal.expected_edge_bps, 4),
        }
        append_trade_log(
            self.state,
            {
                **event,
                "response": response,
            },
        )
        append_trade(self.metrics, event)
        increment_daily_trades(self.state)
        set_last_trade_ts(self.state, _now_utc())
        return (
            "BUY",
            "live buy submitted",
            {
                "order_usd": str(order_usd),
                "base_size_estimated": str(executed_base),
                "estimated_fees_usd": str(total_fees.quantize(Decimal("0.0001"))),
            },
        )

    def _attempt_sell(
        self,
        signal: Signal,
        position: PositionState,
        price: Decimal,
        balances: dict[str, Decimal],
        maker_fee_rate: Decimal,
        taker_fee_rate: Decimal,
        spread_bps: float | None,
        execute_live: bool,
    ) -> tuple[str, str, dict[str, Any]]:
        base_available = balances.get(self.base_currency, Decimal("0"))
        sell_base = min(position.base_size, base_available)
        context = RiskContext(
            now=_now_utc(),
            usd_available=balances.get(self.quote_currency, Decimal("0")),
            base_available=base_available,
            position_base=position.base_size,
            price=price,
        )
        if spread_bps is not None and spread_bps > float(self.config.execution.max_spread_bps):
            return (
                "HOLD",
                (
                    "spread too wide for exit "
                    f"({spread_bps:.2f} bps > {float(self.config.execution.max_spread_bps):.2f} bps)"
                ),
                {
                    "spread_bps": round(spread_bps, 2),
                    "max_spread_bps": round(float(self.config.execution.max_spread_bps), 2),
                },
            )
        reasons = sell_checks(self.state, self.config.guardrails, context, sell_base)
        if reasons:
            return (
                "HOLD",
                f"sell blocked by guardrails: {', '.join(reasons)}",
                {"sell_base": str(sell_base)},
            )
        if get_pending_order(self.state, self.product_id):
            pending = get_pending_order(self.state, self.product_id) or {}
            return (
                "HOLD",
                "spot maker order pending fill",
                {"pending_order_id": str(pending.get("order_id", ""))},
            )

        if self.config.execution.prefer_maker_orders:
            offset = _d(self.config.execution.maker_price_offset_bps) / Decimal("10000")
            limit_price = price * (Decimal("1") + offset)
            maker_fee = (sell_base * limit_price) * maker_fee_rate
            est_realized = ((limit_price - position.avg_entry_price) * sell_base) - maker_fee
            return self._submit_spot_maker_order(
                side="SELL",
                signal=signal,
                base_size=sell_base,
                quote_notional=sell_base * limit_price,
                limit_price=limit_price,
                maker_fee_rate=maker_fee_rate,
                execute_live=execute_live,
                estimated_realized=est_realized,
            )

        preview_sell = self.client.preview_market_sell(self.product_id, sell_base)
        sell_fee = self.client.preview_commission(preview_sell)
        if sell_fee <= 0 and taker_fee_rate > 0:
            sell_fee = (sell_base * price) * taker_fee_rate
        est_realized = ((price - position.avg_entry_price) * sell_base) - sell_fee

        if self.config.mode == "paper":
            fill_price = price * (
                Decimal("1") - (_d(self.config.strategy.slippage_buffer_bps) / Decimal("10000"))
            )
            actual_sell_fee = (sell_base * fill_price) * taker_fee_rate
            self._paper_apply_sell(sell_base, fill_price, actual_sell_fee)
            realized = ((fill_price - position.avg_entry_price) * sell_base) - actual_sell_fee
            self._update_long_position_after_sell(position, sell_base)
            add_daily_realized_pnl(self.state, realized)
            add_total_realized_pnl(self.state, realized)
            add_total_fees(self.state, actual_sell_fee)
            add_trade_outcome(self.state, realized)
            increment_daily_trades(self.state)
            set_last_trade_ts(self.state, _now_utc())
            event = {
                "ts": _now_utc().isoformat(),
                "action": "SELL",
                "mode": "paper",
                "product_id": self.product_id,
                "side": "SELL",
                "price_usd": str(fill_price),
                "base_size": str(sell_base),
                "quote_notional_usd": str(sell_base * fill_price),
                "fee_usd": str(actual_sell_fee),
                "realized_pnl_usd": str(realized),
                "signal_reason": signal.reason,
                "trend_bps": round(signal.trend_bps, 4),
                "rsi": round(signal.rsi, 4),
                "expected_edge_bps": round(signal.expected_edge_bps, 4),
            }
            append_trade_log(
                self.state,
                {
                    **event,
                },
            )
            append_trade(self.metrics, event)
            return (
                "SELL",
                "paper sell executed",
                {
                    "sell_base": str(sell_base),
                    "realized_pnl_usd": str(realized.quantize(Decimal("0.0001"))),
                },
            )

        if not self._live_orders_allowed(execute_live):
            return (
                "HOLD",
                "live sell preview only (set --execute-live and TRADEBOT_ENABLE_LIVE=true)",
                {
                    "sell_base": str(sell_base),
                    "estimated_realized_pnl_usd": str(est_realized.quantize(Decimal("0.0001"))),
                },
            )

        response = self.client.market_sell(self.product_id, sell_base)
        if not _is_successful_order(response):
            append_trade_log(
                self.state,
                {"action": "SELL_FAILED", "product_id": self.product_id, "response": response},
            )
            return "HOLD", "sell failed at exchange", {"response": response}
        self._update_long_position_after_sell(position, sell_base)
        add_daily_realized_pnl(self.state, est_realized)
        add_total_realized_pnl(self.state, est_realized)
        add_total_fees(self.state, sell_fee)
        add_trade_outcome(self.state, est_realized)
        event = {
            "ts": _now_utc().isoformat(),
            "action": "SELL",
            "mode": "live",
            "product_id": self.product_id,
            "side": "SELL",
            "price_usd": str(price),
            "base_size": str(sell_base),
            "quote_notional_usd": str(sell_base * price),
            "fee_usd": str(sell_fee),
            "realized_pnl_usd": str(est_realized),
            "signal_reason": signal.reason,
            "trend_bps": round(signal.trend_bps, 4),
            "rsi": round(signal.rsi, 4),
            "expected_edge_bps": round(signal.expected_edge_bps, 4),
        }
        append_trade_log(
            self.state,
            {
                **event,
                "response": response,
            },
        )
        append_trade(self.metrics, event)
        increment_daily_trades(self.state)
        set_last_trade_ts(self.state, _now_utc())
        return (
            "SELL",
            "live sell submitted",
            {
                "sell_base": str(sell_base),
                "estimated_realized_pnl_usd": str(est_realized.quantize(Decimal("0.0001"))),
            },
        )

    def _submit_spot_maker_order(
        self,
        side: str,
        signal: Signal,
        base_size: Decimal,
        quote_notional: Decimal,
        limit_price: Decimal,
        maker_fee_rate: Decimal,
        execute_live: bool,
        estimated_realized: Decimal | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        side_upper = side.upper()
        if base_size <= 0 or quote_notional <= 0:
            return "HOLD", "maker order size is non-positive", {}
        position = get_position(self.state, self.product_id)
        if side_upper == "BUY" and self.config.mode == "paper":
            buy_fee = quote_notional * maker_fee_rate
            executed_base = (quote_notional - buy_fee) / max(limit_price, Decimal("0.00000001"))
            self._paper_apply_buy(quote_notional, executed_base)
            self._update_long_position_after_buy(position, executed_base, quote_notional)
            add_daily_buy_notional(self.state, quote_notional)
            add_total_fees(self.state, buy_fee)
            increment_daily_trades(self.state)
            set_last_trade_ts(self.state, _now_utc())
            event = {
                "ts": _now_utc().isoformat(),
                "action": "BUY",
                "mode": "paper",
                "product_id": self.product_id,
                "side": "BUY",
                "price_usd": str(limit_price),
                "base_size": str(executed_base),
                "quote_notional_usd": str(quote_notional),
                "fee_usd": str(buy_fee),
                "realized_pnl_usd": "0",
                "signal_reason": signal.reason,
                "signal_source": "maker_post_only",
            }
            append_trade_log(self.state, event)
            append_trade(self.metrics, event)
            return (
                "BUY",
                "paper maker buy executed",
                {
                    "order_usd": str(quote_notional),
                    "base_size": str(executed_base),
                    "limit_price": str(limit_price),
                    "estimated_fees_usd": _q4(buy_fee),
                },
            )
        if side_upper == "SELL" and self.config.mode == "paper":
            sell_fee = quote_notional * maker_fee_rate
            realized = (
                (limit_price - position.avg_entry_price) * base_size - sell_fee
                if position.avg_entry_price > 0
                else Decimal("0")
            )
            self._paper_apply_sell(base_size, limit_price, sell_fee)
            self._update_long_position_after_sell(position, base_size)
            add_daily_realized_pnl(self.state, realized)
            add_total_realized_pnl(self.state, realized)
            add_total_fees(self.state, sell_fee)
            add_trade_outcome(self.state, realized)
            increment_daily_trades(self.state)
            set_last_trade_ts(self.state, _now_utc())
            event = {
                "ts": _now_utc().isoformat(),
                "action": "SELL",
                "mode": "paper",
                "product_id": self.product_id,
                "side": "SELL",
                "price_usd": str(limit_price),
                "base_size": str(base_size),
                "quote_notional_usd": str(quote_notional),
                "fee_usd": str(sell_fee),
                "realized_pnl_usd": str(realized),
                "signal_reason": signal.reason,
                "signal_source": "maker_post_only",
            }
            append_trade_log(self.state, event)
            append_trade(self.metrics, event)
            return (
                "SELL",
                "paper maker sell executed",
                {
                    "sell_base": str(base_size),
                    "limit_price": str(limit_price),
                    "realized_pnl_usd": _q4(realized),
                },
            )

        if not self._live_orders_allowed(execute_live):
            return (
                "HOLD",
                f"live maker {side_upper.lower()} preview only (set --execute-live and TRADEBOT_ENABLE_LIVE=true)",
                {
                    "limit_price": str(limit_price),
                    "base_size": str(base_size),
                    "quote_notional_usd": str(quote_notional),
                    "estimated_realized_pnl_usd": _q4(estimated_realized or Decimal("0")),
                },
            )

        if side_upper == "BUY":
            response = self.client.limit_buy_post_only(self.product_id, base_size, limit_price)
        else:
            response = self.client.limit_sell_post_only(self.product_id, base_size, limit_price)
        if not _is_successful_order(response):
            append_trade_log(
                self.state,
                {
                    "action": f"{side_upper}_FAILED",
                    "product_id": self.product_id,
                    "response": response,
                },
            )
            return "HOLD", f"maker {side_upper.lower()} failed at exchange", {"response": response}

        order_id = _find_text_value(response, ("order_id", "id"))
        if not order_id:
            return "HOLD", f"maker {side_upper.lower()} missing order_id; not tracked", {"response": response}
        set_pending_order(
            self.state,
            self.product_id,
            {
                "order_id": order_id,
                "product_id": self.product_id,
                "side": side_upper,
                "limit_price": str(limit_price),
                "base_size": str(base_size),
                "quote_notional_usd": str(quote_notional),
                "maker_fee_rate": str(maker_fee_rate),
                "accounted_base_size": "0",
                "accounted_quote_notional_usd": "0",
                "accounted_fee_usd": "0",
                "submitted_at": _now_utc().isoformat(),
            },
        )
        append_trade_log(
            self.state,
            {
                "ts": _now_utc().isoformat(),
                "action": f"{side_upper}_MAKER_SUBMITTED",
                "mode": "live",
                "product_id": self.product_id,
                "side": side_upper,
                "price_usd": str(limit_price),
                "base_size": str(base_size),
                "quote_notional_usd": str(quote_notional),
                "fee_usd": "0",
                "realized_pnl_usd": "0",
                "order_id": order_id,
                "signal_reason": signal.reason,
                "signal_source": "maker_post_only",
                "response": response,
            },
        )
        return (
            "HOLD",
            f"maker {side_upper.lower()} submitted (awaiting fill)",
            {
                "pending_order_id": order_id,
                "limit_price": str(limit_price),
                "base_size": str(base_size),
                "quote_notional_usd": str(quote_notional),
            },
        )

    def _reconcile_pending_spot_order(self) -> dict[str, Any]:
        pending = get_pending_order(self.state, self.product_id)
        if not pending:
            return {}
        if self.config.mode != "live":
            clear_pending_order(self.state, self.product_id)
            return {}

        order_id = str(pending.get("order_id", "")).strip()
        if not order_id:
            clear_pending_order(self.state, self.product_id)
            return {}

        details: dict[str, Any] = {
            "pending_order_id": order_id,
            "pending_order_side": str(pending.get("side", "")),
        }
        try:
            payload = self.client.get_order(order_id)
        except Exception as exc:
            details["pending_order_status"] = "UNKNOWN"
            details["pending_order_reason"] = f"status unavailable: {exc}"
            return details

        status = (_find_text_value(payload, ("status", "order_status")) or "UNKNOWN").upper()
        details["pending_order_status"] = status

        filled_base_total = _find_decimal_value(
            payload,
            ("filled_size", "filled_base_size", "filled_quantity", "filled_qty"),
        ) or Decimal("0")
        filled_quote_total = _find_decimal_value(
            payload,
            ("filled_value", "filled_notional", "executed_value", "filled_quote"),
        ) or Decimal("0")
        fee_total = _find_decimal_value(
            payload,
            ("total_fees", "filled_fees", "commission_total", "total_commission"),
        )
        avg_fill_price = _find_decimal_value(
            payload,
            ("average_filled_price", "average_price", "avg_filled_price"),
        )

        accounted_base = _d(pending.get("accounted_base_size", "0"))
        accounted_quote = _d(pending.get("accounted_quote_notional_usd", "0"))
        accounted_fee = _d(pending.get("accounted_fee_usd", "0"))
        delta_base = max(Decimal("0"), filled_base_total - accounted_base)
        delta_quote = max(Decimal("0"), filled_quote_total - accounted_quote)
        maker_fee_rate = _d(pending.get("maker_fee_rate", "0"))
        fill_price = avg_fill_price or _d(pending.get("limit_price", "0"))
        if delta_quote <= 0 and delta_base > 0 and fill_price > 0:
            delta_quote = delta_base * fill_price
        if fee_total is not None:
            delta_fee = max(Decimal("0"), fee_total - accounted_fee)
        else:
            delta_fee = delta_quote * maker_fee_rate if delta_quote > 0 else Decimal("0")

        if delta_base > 0:
            side = str(pending.get("side", "")).upper()
            position = get_position(self.state, self.product_id)
            if side == "BUY":
                self._update_long_position_after_buy(position, delta_base, delta_quote)
                add_daily_buy_notional(self.state, delta_quote)
                add_total_fees(self.state, delta_fee)
                event = {
                    "ts": _now_utc().isoformat(),
                    "action": "BUY",
                    "mode": "live",
                    "product_id": self.product_id,
                    "side": "BUY",
                    "price_usd": str(fill_price),
                    "base_size": str(delta_base),
                    "quote_notional_usd": str(delta_quote),
                    "fee_usd": str(delta_fee),
                    "realized_pnl_usd": "0",
                    "signal_source": "maker_post_only_fill",
                    "order_id": order_id,
                }
                append_trade_log(self.state, event)
                append_trade(self.metrics, event)
                increment_daily_trades(self.state)
                set_last_trade_ts(self.state, _now_utc())
            elif side == "SELL":
                sell_base = min(delta_base, position.base_size)
                if sell_base > 0:
                    ratio = sell_base / delta_base
                    realized_quote = delta_quote * ratio
                    realized_fee = delta_fee * ratio
                    realized = ((fill_price - position.avg_entry_price) * sell_base) - realized_fee
                    self._update_long_position_after_sell(position, sell_base)
                    add_daily_realized_pnl(self.state, realized)
                    add_total_realized_pnl(self.state, realized)
                    add_total_fees(self.state, realized_fee)
                    add_trade_outcome(self.state, realized)
                    event = {
                        "ts": _now_utc().isoformat(),
                        "action": "SELL",
                        "mode": "live",
                        "product_id": self.product_id,
                        "side": "SELL",
                        "price_usd": str(fill_price),
                        "base_size": str(sell_base),
                        "quote_notional_usd": str(realized_quote),
                        "fee_usd": str(realized_fee),
                        "realized_pnl_usd": str(realized),
                        "signal_source": "maker_post_only_fill",
                        "order_id": order_id,
                    }
                    append_trade_log(self.state, event)
                    append_trade(self.metrics, event)
                    increment_daily_trades(self.state)
                    set_last_trade_ts(self.state, _now_utc())

        pending["accounted_base_size"] = str(filled_base_total)
        pending["accounted_quote_notional_usd"] = str(filled_quote_total)
        pending["accounted_fee_usd"] = str(fee_total if fee_total is not None else accounted_fee + delta_fee)
        terminal_statuses = {
            "FILLED",
            "CANCELLED",
            "CANCELED",
            "EXPIRED",
            "FAILED",
            "REJECTED",
            "DONE",
        }
        submitted_at = pending.get("submitted_at")
        if status in terminal_statuses:
            clear_pending_order(self.state, self.product_id)
        else:
            set_pending_order(self.state, self.product_id, pending)
            if submitted_at:
                try:
                    submitted_dt = datetime.fromisoformat(str(submitted_at))
                    age_seconds = (_now_utc() - submitted_dt).total_seconds()
                    details["pending_order_age_seconds"] = int(max(0, age_seconds))
                    if age_seconds > int(self.config.execution.maker_max_order_wait_seconds):
                        details["pending_order_reason"] = "maker order pending longer than maker_max_order_wait_seconds"
                except ValueError:
                    pass

        return details

    def _update_long_position_after_buy(
        self, position: PositionState, bought_base: Decimal, quote_cost_usd: Decimal
    ) -> None:
        old_notional = position.base_size * position.avg_entry_price
        add_notional = quote_cost_usd
        new_base = position.base_size + bought_base
        if new_base <= 0:
            position.base_size = Decimal("0")
            position.avg_entry_price = Decimal("0")
        else:
            position.base_size = new_base
            position.avg_entry_price = (old_notional + add_notional) / new_base
        set_position(self.state, self.product_id, position)

    def _update_long_position_after_sell(self, position: PositionState, sold_base: Decimal) -> None:
        remaining = position.base_size - sold_base
        if remaining <= Decimal("0.00000001"):
            position.base_size = Decimal("0")
            position.avg_entry_price = Decimal("0")
        else:
            position.base_size = remaining
        set_position(self.state, self.product_id, position)

    def _paper_apply_buy(self, order_usd: Decimal, bought_base: Decimal) -> None:
        wallet = self.state.setdefault("paper_wallet", {})
        usd = _d(wallet.get(self.quote_currency, "0"))
        base = _d(wallet.get(self.base_currency, "0"))
        wallet[self.quote_currency] = str(usd - order_usd)
        wallet[self.base_currency] = str(base + bought_base)

    def _paper_apply_sell(self, sell_base: Decimal, fill_price: Decimal, sell_fee: Decimal) -> None:
        wallet = self.state.setdefault("paper_wallet", {})
        usd = _d(wallet.get(self.quote_currency, "0"))
        base = _d(wallet.get(self.base_currency, "0"))
        gross = sell_base * fill_price
        net = gross - sell_fee
        wallet[self.quote_currency] = str(usd + net)
        wallet[self.base_currency] = str(max(Decimal("0"), base - sell_base))

    def _get_balances(self) -> dict[str, Decimal]:
        if self.config.mode == "paper":
            wallet = self.state.setdefault("paper_wallet", {})
            wallet.setdefault(self.quote_currency, wallet.get("USD", "0"))
            wallet.setdefault(self.base_currency, "0")
            return {k: _d(v) for k, v in wallet.items()}
        return self.client.currency_balances()

    def _get_closes(self, product_id: str | None = None) -> list[float]:
        target_product = product_id or self.product_id
        key = (target_product, self.config.granularity, self.config.candles_lookback)
        now = _now_utc()
        cached = self._cached_closes.get(key)
        if cached is not None:
            cached_ts, cached_values = cached
            if (now - cached_ts).total_seconds() < int(self.config.candles_refresh_seconds):
                return cached_values
        try:
            candles = self.client.recent_candles(
                target_product,
                self.config.granularity,
                self.config.candles_lookback,
            )
            closes = [float(candle["close"]) for candle in candles if "close" in candle]
            if not closes:
                raise RuntimeError("Coinbase returned no candles")
            self._cached_closes[key] = (now, closes)
            return closes
        except Exception:
            if cached is not None:
                return cached[1]
            raise

    def _get_fee_rates(self, product_type: str | None = None) -> FeeRates:
        target_product_type = (product_type or self.config.product_type).upper()
        now = _now_utc()
        cached = self._cached_fees.get(target_product_type)
        if cached is not None:
            cached_ts, cached_value = cached
            if (now - cached_ts).total_seconds() < int(self.config.fees_refresh_seconds):
                return cached_value
        try:
            fees = self.client.fee_rates(product_type=target_product_type)
            self._cached_fees[target_product_type] = (now, fees)
            return fees
        except Exception:
            if cached is not None:
                return cached[1]
            raise

    def _get_mid_price(self, product_id: str) -> Decimal:
        self._ensure_ticker_feed()
        now = _now_utc()
        if self._ticker_feed is not None:
            ws_price = self._ticker_feed.latest_mid_price(
                product_id,
                max_age_seconds=float(self.config.websocket.stale_seconds),
            )
            if ws_price is not None:
                self._cached_prices[product_id] = (now, ws_price)
                self._price_source[product_id] = "websocket"
                return ws_price
        cached = self._cached_prices.get(product_id)
        if cached is not None:
            cached_ts, cached_value = cached
            if (now - cached_ts).total_seconds() < float(self.config.price_refresh_seconds):
                self._price_source[product_id] = "cache"
                return cached_value
        try:
            price = self.client.mid_price(product_id)
            self._cached_prices[product_id] = (now, price)
            self._price_source[product_id] = "rest"
            return price
        except Exception:
            if cached is not None:
                self._price_source[product_id] = "cache_fallback"
                return cached[1]
            raise

    @staticmethod
    def _to_decimal_or_none(value: Any) -> Decimal | None:
        if value is None:
            return None
        try:
            return _d(value)
        except Exception:
            return None

    def _current_spread_bps(self, product_id: str) -> float | None:
        self._ensure_ticker_feed()
        now = _now_utc()
        if self._ticker_feed is not None:
            snapshot = self._ticker_feed.latest_snapshot(product_id)
            if isinstance(snapshot, dict):
                bid = self._to_decimal_or_none(snapshot.get("best_bid"))
                ask = self._to_decimal_or_none(snapshot.get("best_ask"))
                if (
                    bid is not None
                    and ask is not None
                    and bid > 0
                    and ask > 0
                    and ask >= bid
                ):
                    self._cached_spreads[product_id] = (now, bid, ask)
                    mid = (bid + ask) / Decimal("2")
                    return float(((ask - bid) / max(mid, Decimal("0.00000001"))) * Decimal("10000"))
        cached = self._cached_spreads.get(product_id)
        if cached is not None:
            cached_ts, bid, ask = cached
            if (now - cached_ts).total_seconds() < float(self.config.price_refresh_seconds):
                mid = (bid + ask) / Decimal("2")
                return float(((ask - bid) / max(mid, Decimal("0.00000001"))) * Decimal("10000"))
        try:
            bid, ask = self.client.best_bid_ask(product_id)
            if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
                return None
            self._cached_spreads[product_id] = (now, bid, ask)
            mid = (bid + ask) / Decimal("2")
            return float(((ask - bid) / max(mid, Decimal("0.00000001"))) * Decimal("10000"))
        except Exception:
            if cached is not None:
                _, bid, ask = cached
                mid = (bid + ask) / Decimal("2")
                return float(((ask - bid) / max(mid, Decimal("0.00000001"))) * Decimal("10000"))
            return None

    def _ensure_ticker_feed(self) -> None:
        if not bool(self.config.websocket.enable):
            return
        if self._ticker_feed_disabled_reason is not None:
            return
        if self._ticker_feed is None:
            try:
                self._ticker_feed = self.client.ticker_websocket_feed(
                    product_ids=[self.product_id],
                    market_data_url=self.config.websocket.market_data_url,
                    channel=self.config.websocket.channel,
                    stale_seconds=float(self.config.websocket.stale_seconds),
                    ping_interval_seconds=int(self.config.websocket.ping_interval_seconds),
                    ping_timeout_seconds=int(self.config.websocket.ping_timeout_seconds),
                    max_reconnect_seconds=int(self.config.websocket.max_reconnect_seconds),
                    subscribe_heartbeats=bool(self.config.websocket.subscribe_heartbeats),
                )
            except Exception as exc:
                self._ticker_feed_disabled_reason = str(exc)
                return
        if not self._ticker_feed.is_running():
            try:
                self._ticker_feed.start()
            except Exception as exc:
                self._ticker_feed_disabled_reason = str(exc)
                self._ticker_feed = None

    def _websocket_status(self, product_id: str) -> dict[str, Any]:
        self._ensure_ticker_feed()
        if not bool(self.config.websocket.enable):
            return {
                "enabled": False,
                "connected": False,
                "reason": "disabled by config.websocket.enable",
            }
        if self._ticker_feed is None:
            return {
                "enabled": True,
                "connected": False,
                "reason": self._ticker_feed_disabled_reason or "not initialized",
            }
        snapshot = self._ticker_feed.latest_snapshot(product_id)
        return {
            "enabled": True,
            "connected": bool(self._ticker_feed.is_connected()),
            "running": bool(self._ticker_feed.is_running()),
            "channel": self.config.websocket.channel,
            "market_data_url": self.config.websocket.market_data_url,
            "stale_seconds": float(self.config.websocket.stale_seconds),
            "last_error": self._ticker_feed.last_error(),
            "last_message_age_seconds": self._ticker_feed.last_message_age_seconds(),
            "latest": snapshot,
        }

    def market_data_marker(self, product_id: str) -> str | None:
        status = self._websocket_status(product_id)
        latest = status.get("latest")
        if isinstance(latest, dict):
            update_id = str(latest.get("update_id") or "").strip()
            if update_id:
                return f"ws:{product_id}:u:{update_id}"
            received_at = str(latest.get("received_at") or "").strip()
            price = str(latest.get("price") or "").strip()
            if received_at:
                return f"ws:{product_id}:{received_at}:{price}"
        return None

    def _record_tick(self, product_id: str, price: Decimal) -> None:
        if not self.config.strategy.enable_subminute_signals:
            return
        now = _now_utc()
        series = self._tick_history.setdefault(product_id, [])
        series.append((now, float(price)))
        cutoff = now - timedelta(seconds=int(self.config.strategy.subminute_window_seconds))
        max_points = max(int(self.config.strategy.subminute_min_samples) * 5, 300)
        kept = [item for item in series if item[0] >= cutoff]
        if len(kept) > max_points:
            kept = kept[-max_points:]
        self._tick_history[product_id] = kept

    def _get_subminute_prices(self, product_id: str) -> list[float]:
        if not self.config.strategy.enable_subminute_signals:
            return []
        now = _now_utc()
        cutoff = now - timedelta(seconds=int(self.config.strategy.subminute_window_seconds))
        series = self._tick_history.get(product_id, [])
        if not series:
            return []
        kept = [item for item in series if item[0] >= cutoff]
        self._tick_history[product_id] = kept
        return [price for _, price in kept]

    def _pnl_summary(self, price: Decimal, position: PositionState) -> dict[str, Any]:
        realized = get_total_realized_pnl(self.state)
        total_fees = get_total_fees(self.state)
        long_unrealized = Decimal("0")
        if position.base_size > 0 and position.avg_entry_price > 0:
            long_unrealized = (price - position.avg_entry_price) * position.base_size
        short_unrealized = Decimal("0")
        short_position = get_short_position(self.state, self.config.auto_actions.short_product_id)
        if short_position.base_size > 0 and short_position.avg_entry_price > 0:
            try:
                short_mark = self._get_mid_price(self.config.auto_actions.short_product_id)
                short_unrealized = (
                    short_position.avg_entry_price - short_mark
                ) * short_position.base_size
            except Exception:
                short_unrealized = Decimal("0")
        unrealized = long_unrealized + short_unrealized
        net_profit = realized + unrealized
        performance = self.state.get("performance", {})
        return {
            "realized_pnl_usd": _q4(realized),
            "unrealized_pnl_usd": _q4(unrealized),
            "net_profit_usd": _q4(net_profit),
            "total_fees_usd": _q4(total_fees),
            "winning_trades": int(performance.get("winning_trades", 0)),
            "losing_trades": int(performance.get("losing_trades", 0)),
        }

    def _daily_buy_snapshot(self) -> dict[str, Any]:
        used = get_daily_buy_notional(self.state)
        limit = _d(self.config.guardrails.max_daily_buy_usd)
        remaining = max(Decimal("0"), limit - used)
        return {
            "daily_buy_used_usd": _q4(used),
            "daily_buy_limit_usd": _q4(limit),
            "daily_buy_remaining_usd": _q4(remaining),
        }

    def _daily_short_snapshot(self) -> dict[str, Any]:
        used = get_daily_short_open_notional(self.state)
        limit = _d(self.config.guardrails.max_daily_short_open_usd)
        remaining = max(Decimal("0"), limit - used)
        return {
            "daily_short_open_used_usd": _q4(used),
            "daily_short_open_limit_usd": _q4(limit),
            "daily_short_open_remaining_usd": _q4(remaining),
        }

    def _record_cycle_point(
        self,
        price: Decimal,
        action: str,
        reason: str,
        position: PositionState,
    ) -> None:
        summary = self._pnl_summary(price, position)
        net = _d(summary["net_profit_usd"])
        peak = _d(self.metrics.get("summary", {}).get("peak_net_profit_usd", "0"))
        peak = max(peak, net)
        drawdown = max(Decimal("0"), peak - net)
        metric_summary = self.metrics.setdefault("summary", {})
        metric_summary["realized_pnl_usd"] = summary["realized_pnl_usd"]
        metric_summary["unrealized_pnl_usd"] = summary["unrealized_pnl_usd"]
        metric_summary["net_profit_usd"] = summary["net_profit_usd"]
        metric_summary["total_fees_usd"] = summary["total_fees_usd"]
        metric_summary["winning_trades"] = summary["winning_trades"]
        metric_summary["losing_trades"] = summary["losing_trades"]
        metric_summary["peak_net_profit_usd"] = _q4(peak)
        metric_summary["max_drawdown_usd"] = _q4(
            max(_d(metric_summary.get("max_drawdown_usd", "0")), drawdown)
        )
        append_equity_point(
            self.metrics,
            {
                "ts": _now_utc().isoformat(),
                "mode": self.config.mode,
                "product_id": self.product_id,
                "action": action,
                "reason": reason,
                "price_usd": str(price),
                "position_base": str(position.base_size),
                "avg_entry_price_usd": str(position.avg_entry_price),
                "realized_pnl_usd": summary["realized_pnl_usd"],
                "unrealized_pnl_usd": summary["unrealized_pnl_usd"],
                "net_profit_usd": summary["net_profit_usd"],
                "total_fees_usd": summary["total_fees_usd"],
            },
        )

    def _persist(self) -> None:
        save_state(self.config.state_file, self.state, self.config.trade_log_limit)
        save_metrics(
            self.config.metrics_file,
            self.metrics,
            trade_limit=self.config.metrics_trade_limit,
            equity_limit=self.config.metrics_equity_limit,
        )

    def _trade_cooldown_active(self) -> bool:
        last = get_last_trade_ts(self.state)
        if last is None:
            return False
        elapsed = (_now_utc() - last).total_seconds()
        return elapsed < int(self.config.guardrails.cooldown_seconds)

    def _auto_check_throttled(self, key: str, interval_seconds: float) -> bool:
        now = _now_utc()
        last = self._last_auto_check_ts.get(key)
        if last is not None and (now - last).total_seconds() < float(interval_seconds):
            return True
        self._last_auto_check_ts[key] = now
        return False

    @staticmethod
    def _is_transient_cycle_error(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        if "coinbase api network error" in message:
            return True
        if "coinbase api error 429" in message:
            return True
        if "coinbase api error 503" in message:
            return True
        if "coinbase api error 502" in message:
            return True
        if "coinbase api error 500" in message:
            return True
        if "coinbase api error 403" in message and "too many errors" in message:
            return True
        return False

    @staticmethod
    def _is_invalid_product_error(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "invalid product_id" in message or "invalid product id" in message

    @staticmethod
    def _live_env_enabled() -> bool:
        raw = os.getenv("TRADEBOT_ENABLE_LIVE")
        if raw is None:
            return False
        normalized = str(raw).strip().lower()
        return normalized in {"true", "1", "yes", "y", "on"}

    def _live_orders_allowed(self, execute_live: bool) -> bool:
        if self.config.mode != "live":
            return False
        if not execute_live:
            return False
        return self._live_env_enabled()

    def _supports_color(self) -> bool:
        if not bool(self.config.use_color_output):
            return False
        if os.getenv("NO_COLOR"):
            return False
        if not sys.stdout.isatty():
            return False
        return True

    def _paint(self, value: object, color: str, enabled: bool) -> str:
        text = str(value)
        if not enabled:
            return text
        return f"{color}{text}{_Ansi.RESET}"

    def _action_color(self, action: str) -> str:
        upper = action.upper()
        if upper == "BUY":
            return _Ansi.GREEN
        if upper == "SELL":
            return _Ansi.RED
        if upper == "SHORT_OPEN":
            return _Ansi.MAGENTA
        if upper == "CLOSE_POSITION":
            return _Ansi.CYAN
        if upper == "CONVERT":
            return _Ansi.BLUE
        return _Ansi.YELLOW

    def _reason_color(self, reason: str) -> str:
        text = str(reason).lower()
        if any(token in text for token in ("failed", "error", "unavailable")):
            return _Ansi.RED
        if any(token in text for token in ("blocked", "limit", "cooldown", "disabled")):
            return _Ansi.YELLOW
        if any(token in text for token in ("opened", "closed", "committed", "executed", "submitted")):
            return _Ansi.GREEN
        if any(token in text for token in ("holding", "no ", "not enough", "throttled")):
            return _Ansi.DIM
        return _Ansi.DIM

    def _detail_key_color(self, key: str) -> str | None:
        if key in {"signal", "candle_signal", "subminute_signal", "short_subminute_signal", "auto_action"}:
            return _Ansi.BOLD
        if "reason" in key:
            return _Ansi.DIM
        if "pnl" in key or "profit" in key:
            return _Ansi.GREEN
        if "fee" in key:
            return _Ansi.RED
        if key.startswith("daily_"):
            return _Ansi.CYAN
        if "signal" in key:
            return _Ansi.MAGENTA
        return None

    def _detail_color(self, key: str, value: object, details: dict[str, Any]) -> str | None:
        if key in {
            "signal",
            "candle_signal",
            "subminute_signal",
            "short_subminute_signal",
            "auto_action",
        }:
            return self._action_color(str(value))

        if key in {"signal_source", "short_signal_source"}:
            source = str(value).lower()
            if source == "subminute":
                return _Ansi.MAGENTA
            if source == "risk_stop":
                return _Ansi.RED
            return _Ansi.CYAN

        if key == "price_source":
            source = str(value).lower()
            if source == "websocket":
                return _Ansi.GREEN
            if source.startswith("cache"):
                return _Ansi.YELLOW
            return _Ansi.CYAN

        if key.endswith("_reason") or key == "signal_reason":
            return self._reason_color(str(value))

        if key.endswith("trend_bps"):
            try:
                trend = _d(value)
            except Exception:
                return None
            if trend > 0:
                return _Ansi.GREEN
            if trend < 0:
                return _Ansi.RED
            return _Ansi.YELLOW

        if key.endswith("rsi") or key == "rsi":
            try:
                rsi_value = _d(value)
            except Exception:
                return None
            if rsi_value >= Decimal("75"):
                return _Ansi.RED
            if rsi_value <= Decimal("25"):
                return _Ansi.BLUE
            if Decimal("45") <= rsi_value <= Decimal("55"):
                return _Ansi.DIM
            return _Ansi.YELLOW

        if key.endswith("expected_edge_bps") or key.endswith("expected_net_usd"):
            try:
                edge = _d(value)
            except Exception:
                return None
            if edge > 0:
                return _Ansi.GREEN
            if edge < 0:
                return _Ansi.RED
            return _Ansi.YELLOW

        if key in {"maker_fee_bps", "taker_fee_bps", "roundtrip_fee_bps"}:
            try:
                fee = _d(value)
            except Exception:
                return None
            if fee >= Decimal("200"):
                return _Ansi.RED
            if fee >= Decimal("80"):
                return _Ansi.YELLOW
            return _Ansi.GREEN

        if key in {"subminute_ticks", "short_subminute_ticks"}:
            try:
                tick_count = int(value)
            except Exception:
                return None
            minimum = int(self.config.strategy.subminute_min_samples)
            if tick_count >= minimum:
                return _Ansi.GREEN
            if tick_count >= max(5, minimum // 2):
                return _Ansi.YELLOW
            return _Ansi.RED

        if key in {"price", "order_usd", "sell_base", "short_order_usd", "short_base_size"}:
            return _Ansi.CYAN

        if key == "short_position_base":
            try:
                size = _d(value)
            except Exception:
                return None
            if size > 0:
                return _Ansi.MAGENTA
            return _Ansi.DIM

        if key == "convert_expected_profit_usd":
            try:
                profit = _d(value)
            except Exception:
                return None
            if profit > 0:
                return _Ansi.GREEN
            if profit < 0:
                return _Ansi.RED
            return _Ansi.YELLOW

        if key in {"realized_pnl_usd", "unrealized_pnl_usd", "net_profit_usd"}:
            try:
                number = _d(value)
            except Exception:
                return None
            if number > 0:
                return _Ansi.GREEN
            if number < 0:
                return _Ansi.RED
            return _Ansi.YELLOW

        if key == "daily_buy_remaining_usd":
            try:
                remaining = _d(value)
            except Exception:
                return None
            if remaining <= 0:
                return _Ansi.RED
            if remaining <= Decimal("1"):
                return _Ansi.YELLOW
            return _Ansi.GREEN

        if key == "daily_buy_used_usd":
            try:
                used = _d(value)
                limit = _d(details.get("daily_buy_limit_usd", "0"))
            except Exception:
                return None
            if limit > 0 and used >= limit:
                return _Ansi.RED
            if used <= 0:
                return _Ansi.DIM
            return _Ansi.YELLOW

        if key == "daily_buy_limit_usd":
            return _Ansi.CYAN

        if key == "daily_short_open_remaining_usd":
            try:
                remaining = _d(value)
            except Exception:
                return None
            if remaining <= 0:
                return _Ansi.RED
            if remaining <= Decimal("1"):
                return _Ansi.YELLOW
            return _Ansi.GREEN

        if key == "daily_short_open_used_usd":
            try:
                used = _d(value)
                limit = _d(details.get("daily_short_open_limit_usd", "0"))
            except Exception:
                return None
            if limit > 0 and used >= limit:
                return _Ansi.RED
            if used <= 0:
                return _Ansi.DIM
            return _Ansi.YELLOW

        if key == "daily_short_open_limit_usd":
            return _Ansi.CYAN

        return None

    def format_cycle_result(self, result: CycleResult) -> str:
        use_color = self._supports_color()
        mode_color = _Ansi.CYAN if result.mode == "live" else _Ansi.MAGENTA
        action_color = self._action_color(result.action)
        mode_text = self._paint(f"[{result.mode}]", mode_color, use_color)
        product_text = self._paint(result.product_id, _Ansi.BLUE, use_color)
        price_text = self._paint(result.price, _Ansi.CYAN, use_color)
        action_text = self._paint(result.action, action_color, use_color)
        reason_text = self._paint(result.reason, self._reason_color(result.reason), use_color)

        detail_parts: list[str] = []
        for key, value in result.details.items():
            key_color = self._detail_key_color(key)
            painted_key = self._paint(key, key_color or "", use_color) if key_color else key
            value_color = self._detail_color(key, value, result.details)
            painted_value = self._paint(value, value_color or "", use_color) if value_color else str(value)
            detail_parts.append(f"{painted_key}={painted_value}")
        details = ", ".join(detail_parts)

        return (
            f"{mode_text} {product_text} price={price_text} "
            f"action={action_text} reason=\"{reason_text}\" {details}"
        )
