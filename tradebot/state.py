from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


def _d(value: Any) -> Decimal:
    return Decimal(str(value))


def _today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


@dataclass(slots=True)
class PositionState:
    base_size: Decimal = Decimal("0")
    avg_entry_price: Decimal = Decimal("0")


@dataclass(slots=True)
class ShortPositionState:
    base_size: Decimal = Decimal("0")
    avg_entry_price: Decimal = Decimal("0")


def default_state(starting_usd: float) -> dict[str, Any]:
    return {
        "version": 2,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "paper_wallet": {"USD": str(_d(starting_usd))},
        "positions": {},
        "short_positions": {},
        "pending_orders": {},
        "risk": {
            "daily_realized_pnl_usd": {},
            "daily_trades": {},
            "daily_buy_notional_usd": {},
            "daily_short_open_notional_usd": {},
            "last_trade_ts": None,
        },
        "performance": {
            "total_realized_pnl_usd": "0",
            "total_fees_usd": "0",
            "winning_trades": 0,
            "losing_trades": 0,
        },
        "trade_log": [],
    }


def load_state(path: str | Path, starting_usd: float) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return default_state(starting_usd)
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        return default_state(starting_usd)
    data = json.loads(raw)
    if "paper_wallet" not in data:
        data["paper_wallet"] = {"USD": str(_d(starting_usd))}
    data.setdefault("positions", {})
    data.setdefault("short_positions", {})
    data.setdefault("pending_orders", {})
    data.setdefault("risk", {})
    data["risk"].setdefault("daily_realized_pnl_usd", {})
    data["risk"].setdefault("daily_trades", {})
    data["risk"].setdefault("daily_buy_notional_usd", {})
    data["risk"].setdefault("daily_short_open_notional_usd", {})
    data["risk"].setdefault("last_trade_ts", None)
    data.setdefault("performance", {})
    data["performance"].setdefault("total_realized_pnl_usd", "0")
    data["performance"].setdefault("total_fees_usd", "0")
    data["performance"].setdefault("winning_trades", 0)
    data["performance"].setdefault("losing_trades", 0)
    data.setdefault("trade_log", [])
    return data


def save_state(path: str | Path, state: dict[str, Any], trade_log_limit: int) -> None:
    p = Path(path)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["trade_log"] = state.get("trade_log", [])[-trade_log_limit:]
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


def get_position(state: dict[str, Any], product_id: str) -> PositionState:
    raw = state.get("positions", {}).get(product_id, {})
    return PositionState(
        base_size=_d(raw.get("base_size", "0")),
        avg_entry_price=_d(raw.get("avg_entry_price", "0")),
    )


def set_position(state: dict[str, Any], product_id: str, position: PositionState) -> None:
    state.setdefault("positions", {})[product_id] = {
        "base_size": str(position.base_size),
        "avg_entry_price": str(position.avg_entry_price),
    }


def get_short_position(state: dict[str, Any], product_id: str) -> ShortPositionState:
    raw = state.get("short_positions", {}).get(product_id, {})
    return ShortPositionState(
        base_size=_d(raw.get("base_size", "0")),
        avg_entry_price=_d(raw.get("avg_entry_price", "0")),
    )


def set_short_position(
    state: dict[str, Any], product_id: str, position: ShortPositionState
) -> None:
    state.setdefault("short_positions", {})[product_id] = {
        "base_size": str(position.base_size),
        "avg_entry_price": str(position.avg_entry_price),
    }


def clear_short_position(state: dict[str, Any], product_id: str) -> None:
    state.setdefault("short_positions", {}).pop(product_id, None)


def get_pending_order(state: dict[str, Any], product_id: str) -> dict[str, Any] | None:
    raw = state.get("pending_orders", {}).get(product_id)
    if isinstance(raw, dict):
        return raw
    return None


def set_pending_order(state: dict[str, Any], product_id: str, payload: dict[str, Any]) -> None:
    state.setdefault("pending_orders", {})[product_id] = payload


def clear_pending_order(state: dict[str, Any], product_id: str) -> None:
    state.setdefault("pending_orders", {}).pop(product_id, None)


def append_trade_log(state: dict[str, Any], event: dict[str, Any]) -> None:
    state.setdefault("trade_log", []).append(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            **event,
        }
    )


def get_daily_realized_pnl(state: dict[str, Any], day: str | None = None) -> Decimal:
    day_key = day or _today_utc()
    raw = state.get("risk", {}).get("daily_realized_pnl_usd", {}).get(day_key, "0")
    return _d(raw)


def add_daily_realized_pnl(state: dict[str, Any], delta: Decimal, day: str | None = None) -> None:
    day_key = day or _today_utc()
    risk = state.setdefault("risk", {})
    pnl = risk.setdefault("daily_realized_pnl_usd", {})
    pnl[day_key] = str(_d(pnl.get(day_key, "0")) + delta)


def get_daily_trades(state: dict[str, Any], day: str | None = None) -> int:
    day_key = day or _today_utc()
    raw = state.get("risk", {}).get("daily_trades", {}).get(day_key, 0)
    return int(raw)


def increment_daily_trades(state: dict[str, Any], day: str | None = None) -> None:
    day_key = day or _today_utc()
    risk = state.setdefault("risk", {})
    trades = risk.setdefault("daily_trades", {})
    trades[day_key] = int(trades.get(day_key, 0)) + 1


def get_daily_buy_notional(state: dict[str, Any], day: str | None = None) -> Decimal:
    day_key = day or _today_utc()
    raw = state.get("risk", {}).get("daily_buy_notional_usd", {}).get(day_key, "0")
    return _d(raw)


def add_daily_buy_notional(state: dict[str, Any], delta: Decimal, day: str | None = None) -> None:
    day_key = day or _today_utc()
    risk = state.setdefault("risk", {})
    spent = risk.setdefault("daily_buy_notional_usd", {})
    spent[day_key] = str(_d(spent.get(day_key, "0")) + delta)


def get_daily_short_open_notional(state: dict[str, Any], day: str | None = None) -> Decimal:
    day_key = day or _today_utc()
    raw = state.get("risk", {}).get("daily_short_open_notional_usd", {}).get(day_key, "0")
    return _d(raw)


def add_daily_short_open_notional(
    state: dict[str, Any], delta: Decimal, day: str | None = None
) -> None:
    day_key = day or _today_utc()
    risk = state.setdefault("risk", {})
    spent = risk.setdefault("daily_short_open_notional_usd", {})
    spent[day_key] = str(_d(spent.get(day_key, "0")) + delta)


def set_last_trade_ts(state: dict[str, Any], ts: datetime) -> None:
    state.setdefault("risk", {})["last_trade_ts"] = ts.isoformat()


def get_last_trade_ts(state: dict[str, Any]) -> datetime | None:
    raw = state.get("risk", {}).get("last_trade_ts")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def get_total_realized_pnl(state: dict[str, Any]) -> Decimal:
    raw = state.get("performance", {}).get("total_realized_pnl_usd", "0")
    return _d(raw)


def add_total_realized_pnl(state: dict[str, Any], delta: Decimal) -> None:
    performance = state.setdefault("performance", {})
    current = _d(performance.get("total_realized_pnl_usd", "0"))
    performance["total_realized_pnl_usd"] = str(current + delta)


def get_total_fees(state: dict[str, Any]) -> Decimal:
    raw = state.get("performance", {}).get("total_fees_usd", "0")
    return _d(raw)


def add_total_fees(state: dict[str, Any], delta: Decimal) -> None:
    performance = state.setdefault("performance", {})
    current = _d(performance.get("total_fees_usd", "0"))
    performance["total_fees_usd"] = str(current + delta)


def add_trade_outcome(state: dict[str, Any], realized_pnl: Decimal) -> None:
    performance = state.setdefault("performance", {})
    if realized_pnl > 0:
        performance["winning_trades"] = int(performance.get("winning_trades", 0)) + 1
    elif realized_pnl < 0:
        performance["losing_trades"] = int(performance.get("losing_trades", 0)) + 1
