from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_metrics() -> dict[str, Any]:
    return {
        "version": 1,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "summary": {
            "realized_pnl_usd": "0",
            "unrealized_pnl_usd": "0",
            "net_profit_usd": "0",
            "total_fees_usd": "0",
            "winning_trades": 0,
            "losing_trades": 0,
            "trade_count": 0,
            "peak_net_profit_usd": "0",
            "max_drawdown_usd": "0",
            "last_trade_ts": None,
        },
        "trades": [],
        "equity_curve": [],
    }


def load_metrics(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return default_metrics()
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        return default_metrics()
    data = json.loads(raw)
    data.setdefault("summary", {})
    data["summary"].setdefault("realized_pnl_usd", "0")
    data["summary"].setdefault("unrealized_pnl_usd", "0")
    data["summary"].setdefault("net_profit_usd", "0")
    data["summary"].setdefault("total_fees_usd", "0")
    data["summary"].setdefault("winning_trades", 0)
    data["summary"].setdefault("losing_trades", 0)
    data["summary"].setdefault("trade_count", 0)
    data["summary"].setdefault("peak_net_profit_usd", "0")
    data["summary"].setdefault("max_drawdown_usd", "0")
    data["summary"].setdefault("last_trade_ts", None)
    data.setdefault("trades", [])
    data.setdefault("equity_curve", [])
    return data


def save_metrics(
    path: str | Path,
    metrics: dict[str, Any],
    trade_limit: int,
    equity_limit: int,
) -> None:
    p = Path(path)
    metrics["updated_at"] = _now_iso()
    metrics["trades"] = metrics.get("trades", [])[-trade_limit:]
    metrics["equity_curve"] = metrics.get("equity_curve", [])[-equity_limit:]
    p.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def append_trade(metrics: dict[str, Any], event: dict[str, Any]) -> None:
    metrics.setdefault("trades", []).append(event)
    summary = metrics.setdefault("summary", {})
    summary["trade_count"] = int(summary.get("trade_count", 0)) + 1
    summary["last_trade_ts"] = event.get("ts")


def append_equity_point(metrics: dict[str, Any], point: dict[str, Any]) -> None:
    metrics.setdefault("equity_curve", []).append(point)
