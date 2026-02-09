from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from .config import GuardrailSettings
from .state import (
    get_daily_buy_notional,
    get_daily_realized_pnl,
    get_daily_short_open_notional,
    get_daily_trades,
    get_last_trade_ts,
)


def _d(value: object) -> Decimal:
    return Decimal(str(value))


@dataclass(slots=True)
class RiskContext:
    now: datetime
    usd_available: Decimal
    base_available: Decimal
    position_base: Decimal
    price: Decimal


def _seconds_since(older: datetime, newer: datetime) -> float:
    return (newer - older).total_seconds()


def buy_checks(
    state: dict,
    guardrails: GuardrailSettings,
    context: RiskContext,
    order_usd: Decimal,
) -> list[str]:
    reasons: list[str] = []
    if order_usd <= 0:
        reasons.append("order size is zero")
        return reasons

    daily_pnl = get_daily_realized_pnl(state)
    if daily_pnl <= -_d(guardrails.max_daily_loss_usd):
        reasons.append("daily loss limit reached")
    if guardrails.daily_profit_target_usd > 0 and daily_pnl >= _d(guardrails.daily_profit_target_usd):
        reasons.append("daily profit target reached")

    trades_today = get_daily_trades(state)
    if guardrails.max_trades_per_day > 0 and trades_today >= guardrails.max_trades_per_day:
        reasons.append("max trades per day reached")

    daily_buy_notional = get_daily_buy_notional(state)
    if daily_buy_notional + order_usd > _d(guardrails.max_daily_buy_usd):
        reasons.append("max_daily_buy_usd limit reached")

    last_trade = get_last_trade_ts(state)
    if last_trade is not None:
        if _seconds_since(last_trade, context.now) < guardrails.cooldown_seconds:
            reasons.append("cooldown active")

    if order_usd < _d(guardrails.min_order_usd):
        reasons.append("order below min_order_usd")
    if order_usd > _d(guardrails.max_order_usd):
        reasons.append("order above max_order_usd")

    post_trade_reserve = context.usd_available - order_usd
    if post_trade_reserve < _d(guardrails.min_usd_reserve):
        reasons.append("would violate min_usd_reserve")

    projected_position_usd = (context.position_base * context.price) + order_usd
    if projected_position_usd > _d(guardrails.max_position_usd):
        reasons.append("would violate max_position_usd")

    return reasons


def sell_checks(
    state: dict,
    guardrails: GuardrailSettings,
    context: RiskContext,
    sell_base: Decimal,
) -> list[str]:
    reasons: list[str] = []
    if sell_base <= 0:
        reasons.append("sell size is zero")
        return reasons
    if sell_base > context.base_available:
        reasons.append("insufficient base balance")
    notional = sell_base * context.price
    if notional < _d(guardrails.min_order_usd):
        reasons.append("sell notional below min_order_usd")
    return reasons


def short_open_checks(
    state: dict,
    guardrails: GuardrailSettings,
    notional_usd: Decimal,
    leverage: Decimal,
) -> list[str]:
    reasons: list[str] = []
    if notional_usd <= 0:
        reasons.append("short notional is zero")
        return reasons

    daily_pnl = get_daily_realized_pnl(state)
    if daily_pnl <= -_d(guardrails.max_daily_loss_usd):
        reasons.append("daily loss limit reached")
    if guardrails.daily_profit_target_usd > 0 and daily_pnl >= _d(guardrails.daily_profit_target_usd):
        reasons.append("daily profit target reached")

    if notional_usd > _d(guardrails.max_short_notional_usd):
        reasons.append("short notional exceeds max_short_notional_usd")

    if leverage > _d(guardrails.max_short_leverage):
        reasons.append("leverage exceeds max_short_leverage")

    daily_short_open = get_daily_short_open_notional(state)
    if daily_short_open + notional_usd > _d(guardrails.max_daily_short_open_usd):
        reasons.append("max_daily_short_open_usd limit reached")

    return reasons
