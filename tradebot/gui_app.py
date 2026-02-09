from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

try:
    from tradebot.bot import TradeBot
    from tradebot.config import BotConfig, load_config
    from tradebot.metrics import load_metrics
    from tradebot.state import load_state
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tradebot.bot import TradeBot
    from tradebot.config import BotConfig, load_config
    from tradebot.metrics import load_metrics
    from tradebot.state import load_state


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="config.json")
    args, _ = parser.parse_known_args()
    return args


def _d(value: object, default: Decimal = Decimal("0")) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return default


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        try:
            return float(Decimal(str(value)))
        except Exception:
            return default


def _ratio(used: float, limit: float) -> float:
    if limit <= 0:
        return 0.0
    return max(0.0, min(1.0, used / limit))


def _live_env_raw() -> str | None:
    raw = os.getenv("TRADEBOT_ENABLE_LIVE")
    if raw is None:
        return None
    return str(raw)


def _live_env_enabled() -> bool:
    raw = _live_env_raw()
    if raw is None:
        return False
    return raw.strip().lower() in {"true", "1", "yes", "y", "on"}


def _today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _format_utc_time(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).strftime("%H:%M:%S")
    except Exception:
        return text[-8:]


def _hide_streamlit_status_widget() -> None:
    # Hide Streamlit's top-right run-status/Stop widget in app mode.
    st.markdown(
        """
        <style>
        header [data-testid="stStatusWidget"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _inject_dashboard_theme() -> None:
    # Intentionally minimal to keep default Streamlit aesthetics.
    return


def _chip_html(label: str, value: object, *, tone: str = "neutral") -> str:
    palette = {
        "neutral": ("#334155", "#eef2f8"),
        "good": ("#0f766e", "#d1fae5"),
        "warn": ("#92400e", "#fef3c7"),
        "bad": ("#b91c1c", "#fee2e2"),
        "info": ("#1d4ed8", "#dbeafe"),
    }
    fg, bg = palette.get(tone, palette["neutral"])
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    return (
        f"<span class='tb-chip' style='color:{fg}; background:{bg};'>"
        f"{safe_label}: <strong>{safe_value}</strong>"
        "</span>"
    )


def _render_section_card(title: str, subtitle: str) -> None:
    st.markdown(
        (
            "<div class='tb-section'>"
            f"<div class='tb-section-title'>{html.escape(title)}</div>"
            f"<div class='tb-section-subtitle'>{html.escape(subtitle)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _load_dotenv_if_present(dotenv_path: str = ".env") -> None:
    path = Path(dotenv_path)
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_key = key.strip()
        if not env_key or env_key in os.environ:
            continue
        env_value = value.strip()
        if len(env_value) >= 2 and (
            (env_value.startswith('"') and env_value.endswith('"'))
            or (env_value.startswith("'") and env_value.endswith("'"))
        ):
            env_value = env_value[1:-1]
        os.environ[env_key] = env_value


def _load_runtime_files(config: BotConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    state = load_state(config.state_file, config.paper.starting_usd)
    metrics = load_metrics(config.metrics_file)
    return state, metrics


def _bot_session_key(config_path: str, config: BotConfig) -> str:
    try:
        p = Path(config_path).resolve()
        if p.exists():
            stamp = p.stat().st_mtime_ns
            return f"{p}:{stamp}:{config.mode}:{config.product_id}"
        return f"{p}:missing:{config.mode}:{config.product_id}"
    except Exception:
        return f"{config_path}:{config.mode}:{config.product_id}"


def _get_or_create_bot(config_path: str, config: BotConfig) -> TradeBot:
    key = _bot_session_key(config_path, config)
    existing_key = st.session_state.get("gui_bot_key")
    existing_bot = st.session_state.get("gui_bot")
    if isinstance(existing_bot, TradeBot) and existing_key == key:
        return existing_bot
    if isinstance(existing_bot, TradeBot):
        try:
            existing_bot.close()
        except Exception:
            pass
    bot = TradeBot(config)
    st.session_state["gui_bot"] = bot
    st.session_state["gui_bot_key"] = key
    return bot


def _build_equity_df(metrics: dict[str, Any]) -> pd.DataFrame:
    points = metrics.get("equity_curve", [])
    if not isinstance(points, list) or not points:
        return pd.DataFrame()
    df = pd.DataFrame(points)
    if "ts" not in df.columns:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    for col in [
        "price_usd",
        "position_base",
        "avg_entry_price_usd",
        "realized_pnl_usd",
        "unrealized_pnl_usd",
        "net_profit_usd",
        "total_fees_usd",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    return df


def _build_trades_df(metrics: dict[str, Any]) -> pd.DataFrame:
    trades = metrics.get("trades", [])
    if not isinstance(trades, list) or not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    for col in [
        "price_usd",
        "base_size",
        "quote_notional_usd",
        "fee_usd",
        "realized_pnl_usd",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "side" not in df.columns and "action" in df.columns:
        df["side"] = df["action"]
    return df.sort_values("ts") if "ts" in df.columns else df


def _cycle_to_dict(result: Any, line: str) -> dict[str, Any]:
    clean_line = _ANSI_RE.sub("", str(line))
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": result.action,
        "reason": result.reason,
        "mode": result.mode,
        "product_id": result.product_id,
        "price": str(result.price),
        "line": clean_line,
        "details": result.details,
    }


def _cycle_value_color(key: str, value: object) -> str:
    text = str(value)
    upper = text.upper()
    decimal_value: Decimal | None = None
    try:
        decimal_value = Decimal(str(value))
    except Exception:
        decimal_value = None

    if key in {"action", "signal", "candle_signal", "subminute_signal"}:
        if upper in {"BUY", "SHORT_OPEN"}:
            return "#22c55e"
        if upper in {"SELL", "CLOSE_POSITION"}:
            return "#ef4444"
        if upper == "HOLD":
            return "#f59e0b"
        return "#a855f7"

    if key == "price_source":
        source = text.lower()
        if source == "websocket":
            return "#22c55e"
        if source.startswith("cache"):
            return "#f59e0b"
        if source == "rest":
            return "#38bdf8"
        return "#e5e7eb"

    if key.endswith("_reason") or key == "reason":
        return "#9ca3af"

    if "fee" in key:
        return "#f87171"

    if "pnl" in key or "profit" in key:
        if decimal_value is not None:
            if decimal_value > 0:
                return "#22c55e"
            if decimal_value < 0:
                return "#ef4444"
        return "#f59e0b"

    if key.endswith("trend_bps") or key.endswith("expected_edge_bps"):
        if decimal_value is not None:
            if decimal_value > 0:
                return "#22c55e"
            if decimal_value < 0:
                return "#ef4444"
        return "#f59e0b"

    if key.endswith("rsi") or key == "rsi":
        if decimal_value is not None:
            if decimal_value >= Decimal("75"):
                return "#ef4444"
            if decimal_value <= Decimal("25"):
                return "#38bdf8"
        return "#f59e0b"

    if key.startswith("daily_"):
        return "#38bdf8"

    return "#e5e7eb"


def _format_cycle_multiline_html(cycle: dict[str, Any]) -> str:
    rows: list[tuple[str, object]] = [
        ("ts", cycle.get("ts", "")),
        ("mode", cycle.get("mode", "")),
        ("product_id", cycle.get("product_id", "")),
        ("price", cycle.get("price", "")),
        ("action", cycle.get("action", "")),
        ("reason", cycle.get("reason", "")),
    ]
    details = cycle.get("details")
    if isinstance(details, dict):
        for key, value in details.items():
            rows.append((str(key), value))

    lines: list[str] = []
    for key, value in rows:
        safe_key = html.escape(str(key))
        safe_value = html.escape(str(value))
        value_color = _cycle_value_color(key, value)
        lines.append(
            (
                "<div>"
                f"<span style='color:#94a3b8;'>{safe_key}</span>"
                "<span style='color:#64748b;'>=</span>"
                f"<span style='color:{value_color};'>{safe_value}</span>"
                "</div>"
            )
        )
    return (
        "<div style='font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; "
        "font-size: 0.85rem; line-height: 1.35; white-space: pre-wrap;'>"
        + "".join(lines)
        + "</div>"
    )


def _render_hero(
    config: BotConfig,
    metrics: dict[str, Any],
    *,
    refresh_interval: int | None,
) -> None:
    del metrics, refresh_interval
    st.caption(f"Mode `{config.mode}` | Product `{config.product_id}`")


def _render_last_cycle_panel(config: BotConfig) -> None:
    if "last_cycle" not in st.session_state:
        st.info("No cycle run yet. Click `Run One Cycle` or start loop mode.")
        return
    raw_last = st.session_state["last_cycle"]
    if not isinstance(raw_last, dict):
        st.info("Last cycle data is unavailable.")
        return
    last = raw_last
    details = last.get("details", {}) if isinstance(last, dict) else {}
    details = details if isinstance(details, dict) else {}
    st.subheader("Last Cycle")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Action", str(last.get("action", "UNKNOWN")))
    c2.metric("Price", f"${_to_float(last.get('price')):,.2f}")
    c3.metric("Mode", str(last.get("mode", config.mode)))
    c4.metric("Time (UTC)", _format_utc_time(last.get("ts", "")))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Signal Source", str(details.get("signal_source", "n/a")))
    k2.metric("RSI", f"{_to_float(details.get('rsi')):.2f}")
    k3.metric("Trend (bps)", f"{_to_float(details.get('trend_bps')):.2f}")
    k4.metric("Expected Edge (bps)", f"{_to_float(details.get('expected_edge_bps')):.2f}")
    st.markdown(_format_cycle_multiline_html(last), unsafe_allow_html=True)
    st.markdown("")
    with st.expander("Cycle Details JSON"):
        st.json(last)


def _render_summary_cards(
    metrics: dict[str, Any],
    state: dict[str, Any],
    config: BotConfig,
) -> None:
    summary = metrics.get("summary", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Net Profit", f"${_to_float(summary.get('net_profit_usd')):,.2f}")
    c2.metric("Realized PnL", f"${_to_float(summary.get('realized_pnl_usd')):,.2f}")
    c3.metric("Unrealized PnL", f"${_to_float(summary.get('unrealized_pnl_usd')):,.2f}")
    c4.metric("Total Fees", f"${_to_float(summary.get('total_fees_usd')):,.2f}")
    c5.metric("Trade Count", f"{int(summary.get('trade_count', 0))}")

    today = _today_utc()
    risk = state.get("risk", {})
    daily_buy_used = _to_float(risk.get("daily_buy_notional_usd", {}).get(today, "0"))
    daily_short_used = _to_float(risk.get("daily_short_open_notional_usd", {}).get(today, "0"))
    daily_realized = _to_float(risk.get("daily_realized_pnl_usd", {}).get(today, "0"))

    buy_limit = float(config.guardrails.max_daily_buy_usd)
    short_limit = float(config.guardrails.max_daily_short_open_usd)
    loss_limit = float(config.guardrails.max_daily_loss_usd)
    target = float(config.guardrails.daily_profit_target_usd)

    st.subheader("Daily Guardrails (UTC)")
    st.progress(
        _ratio(daily_buy_used, buy_limit),
        text=f"Buy budget: ${daily_buy_used:.2f} / ${buy_limit:.2f}",
    )
    st.progress(
        _ratio(daily_short_used, short_limit),
        text=f"Short-open budget: ${daily_short_used:.2f} / ${short_limit:.2f}",
    )
    loss_ratio = _ratio(max(0.0, -daily_realized), loss_limit)
    st.progress(
        loss_ratio,
        text=f"Daily realized PnL: ${daily_realized:.2f} | loss stop ${loss_limit:.2f}",
    )
    if target > 0:
        st.progress(
            _ratio(max(0.0, daily_realized), target),
            text=f"Daily profit target progress: ${max(0.0, daily_realized):.2f} / ${target:.2f}",
        )

    st.caption("Guardrail Snapshot")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric(
        "Buy Budget Used",
        f"${daily_buy_used:.2f}",
        f"{_ratio(daily_buy_used, buy_limit) * 100:.1f}% of limit",
    )
    g2.metric(
        "Short Budget Used",
        f"${daily_short_used:.2f}",
        f"{_ratio(daily_short_used, short_limit) * 100:.1f}% of limit",
    )
    g3.metric(
        "Loss Buffer Used",
        f"${max(0.0, -daily_realized):.2f}",
        f"{_ratio(max(0.0, -daily_realized), loss_limit) * 100:.1f}% of stop",
    )
    if target > 0:
        g4.metric(
            "Profit Target Reached",
            f"${max(0.0, daily_realized):.2f}",
            f"{_ratio(max(0.0, daily_realized), target) * 100:.1f}% of target",
        )
    else:
        g4.metric("Profit Target", "Disabled", "set daily_profit_target_usd > 0")

    table_rows: list[dict[str, float | str]] = [
        {
            "Guardrail": "Buy Budget",
            "Used USD": round(daily_buy_used, 2),
            "Limit USD": round(max(0.0, buy_limit), 2),
            "Remaining USD": round(max(0.0, buy_limit - daily_buy_used), 2),
            "Utilization %": round(_ratio(daily_buy_used, buy_limit) * 100, 2),
        },
        {
            "Guardrail": "Short-open Budget",
            "Used USD": round(daily_short_used, 2),
            "Limit USD": round(max(0.0, short_limit), 2),
            "Remaining USD": round(max(0.0, short_limit - daily_short_used), 2),
            "Utilization %": round(_ratio(daily_short_used, short_limit) * 100, 2),
        },
        {
            "Guardrail": "Daily Loss Stop",
            "Used USD": round(max(0.0, -daily_realized), 2),
            "Limit USD": round(max(0.0, loss_limit), 2),
            "Remaining USD": round(max(0.0, loss_limit - max(0.0, -daily_realized)), 2),
            "Utilization %": round(_ratio(max(0.0, -daily_realized), loss_limit) * 100, 2),
        },
    ]
    if target > 0:
        table_rows.append(
            {
                "Guardrail": "Daily Profit Target",
                "Used USD": round(max(0.0, daily_realized), 2),
                "Limit USD": round(max(0.0, target), 2),
                "Remaining USD": round(max(0.0, target - max(0.0, daily_realized)), 2),
                "Utilization %": round(_ratio(max(0.0, daily_realized), target) * 100, 2),
            }
        )
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch", height=210)


def _render_position_cards(state: dict[str, Any], config: BotConfig) -> None:
    long_position = state.get("positions", {}).get(config.product_id, {})
    short_id = config.auto_actions.short_product_id
    short_position = state.get("short_positions", {}).get(short_id, {})
    pending_order = state.get("pending_orders", {}).get(config.product_id, {})

    c1, c2, c3 = st.columns(3)
    c1.metric(
        f"Long {config.product_id}",
        f"{_to_float(long_position.get('base_size', '0')):.8f}",
        help="Tracked spot base position size.",
    )
    c2.metric(
        f"Short {short_id}",
        f"{_to_float(short_position.get('base_size', '0')):.8f}",
        help="Tracked derivative short base size.",
    )
    c3.metric(
        "Pending Spot Orders",
        "1" if isinstance(pending_order, dict) and pending_order else "0",
        help="Post-only maker order waiting for reconciliation.",
    )

    with st.expander("Position and Pending Order JSON"):
        st.json(
            {
                "long_position": long_position,
                "short_position": short_position,
                "pending_order": pending_order or {},
            }
        )


def _render_equity_charts(metrics: dict[str, Any]) -> None:
    st.subheader("Equity and Price")
    equity_df = _build_equity_df(metrics)
    if equity_df.empty:
        st.info("No equity data yet. Run at least one cycle to start building charts.")
        return
    idx_df = equity_df.set_index("ts")
    primary = [col for col in ["net_profit_usd", "realized_pnl_usd", "unrealized_pnl_usd"] if col in idx_df.columns]
    if primary:
        st.line_chart(idx_df[primary], height=250)

    c1, c2 = st.columns(2)
    with c1:
        if "net_profit_usd" in idx_df.columns:
            st.area_chart(idx_df[["net_profit_usd"]], height=170)
        elif "realized_pnl_usd" in idx_df.columns:
            st.area_chart(idx_df[["realized_pnl_usd"]], height=170)
    with c2:
        if "total_fees_usd" in idx_df.columns:
            st.line_chart(idx_df[["total_fees_usd"]], height=170)
        elif "price_usd" in idx_df.columns:
            st.line_chart(idx_df[["price_usd"]], height=170)

    if "price_usd" in idx_df.columns:
        st.line_chart(idx_df[["price_usd"]], height=170)
    if "position_base" in idx_df.columns:
        st.line_chart(idx_df[["position_base"]], height=160)


def _render_trade_charts(metrics: dict[str, Any]) -> None:
    st.subheader("Trade Activity")
    trade_df = _build_trades_df(metrics)
    if trade_df.empty:
        st.info("No trade events in metrics yet.")
        return
    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        if "side" in trade_df.columns:
            side_counts = trade_df["side"].fillna("UNKNOWN").value_counts()
            st.bar_chart(side_counts, height=210)
    with chart_c2:
        if "realized_pnl_usd" in trade_df.columns and "side" in trade_df.columns:
            pnl_by_side = trade_df.groupby("side", dropna=False)["realized_pnl_usd"].sum()
            st.bar_chart(pnl_by_side, height=210)

    if "ts" in trade_df.columns:
        timeline = trade_df.dropna(subset=["ts"]).sort_values("ts").set_index("ts")
        if not timeline.empty:
            timeline["cumulative_trades"] = 1
            timeline["cumulative_trades"] = timeline["cumulative_trades"].cumsum()
            line_cols = ["cumulative_trades"]
            if "realized_pnl_usd" in timeline.columns:
                timeline["cumulative_realized_pnl_usd"] = timeline["realized_pnl_usd"].fillna(0).cumsum()
                line_cols.append("cumulative_realized_pnl_usd")
            st.line_chart(timeline[line_cols], height=210)

    display_cols = [
        "ts",
        "action",
        "side",
        "price_usd",
        "base_size",
        "quote_notional_usd",
        "fee_usd",
        "realized_pnl_usd",
    ]
    existing_cols = [col for col in display_cols if col in trade_df.columns]
    st.dataframe(trade_df[existing_cols], width="stretch", height=300)


def _render_backtest_panel(bot: TradeBot) -> None:
    with st.expander("Walk-Forward Backtest"):
        c1, c2, c3, c4 = st.columns(4)
        lookback = int(c1.number_input("Lookback Candles", min_value=500, value=3000, step=100))
        train = int(c2.number_input("Train Candles", min_value=100, value=600, step=50))
        test = int(c3.number_input("Test Candles", min_value=30, value=180, step=30))
        step = int(c4.number_input("Step Candles", min_value=10, value=90, step=10))
        include_scenarios = st.checkbox(
            "Include synthetic scenario suite (bull/bear/chop/volatility)",
            value=True,
        )
        scenarios_only = st.checkbox(
            "Scenarios only (skip historical download)",
            value=False,
            help="Runs deterministic synthetic scenarios even if historical data is unavailable or flat.",
        )
        if scenarios_only and not include_scenarios:
            include_scenarios = True
        scenario_length = int(
            st.number_input(
                "Scenario Candles",
                min_value=max(train + test + 10, 300),
                value=max(1800, train + test + 10),
                step=100,
                disabled=not include_scenarios,
            )
        )
        run_backtest = st.button("Run Backtest", key="run_backtest_btn")
        if run_backtest:
            try:
                output = bot.backtest_walk_forward(
                    lookback_candles=lookback,
                    train_candles=train,
                    test_candles=test,
                    step_candles=step,
                    include_scenarios=include_scenarios,
                    scenario_length=scenario_length,
                    scenarios_only=scenarios_only,
                )
                st.session_state["gui_backtest"] = output
                st.success("Backtest complete.")
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
        backtest = st.session_state.get("gui_backtest")
        if not isinstance(backtest, dict):
            return
        summary = backtest.get("summary", {})
        meta = backtest.get("meta", {})
        if bool(meta.get("scenarios_only")):
            st.info("Running in scenarios-only mode (historical replay skipped).")
        st.json(summary)
        if (
            int(summary.get("trades", 0)) == 0
            and int(summary.get("buy_blocked_expected_profit_count", 0)) > 0
        ):
            st.warning(
                "No trades executed because BUY signals were blocked by fee-adjusted "
                "expected-profit checks. Compare `approx_required_edge_bps_for_buy` "
                "vs `max_observed_expected_edge_bps` in summary."
            )
        windows = backtest.get("windows", [])
        if isinstance(windows, list) and windows:
            df = pd.DataFrame(windows)
            for col in ["window_return_pct", "realized_pnl_usd", "equity_end_usd"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "start_index" in df.columns:
                df = df.sort_values("start_index").set_index("start_index")
            if "window_return_pct" in df.columns:
                st.line_chart(df[["window_return_pct"]], height=180)
            if "realized_pnl_usd" in df.columns:
                st.line_chart(df[["realized_pnl_usd"]], height=180)
            st.dataframe(df, width="stretch", height=260)

        scenario_suite = backtest.get("scenario_suite")
        if isinstance(scenario_suite, dict):
            st.subheader("Synthetic Scenario Suite")
            suite_summary_cols = st.columns(3)
            suite_summary_cols[0].metric(
                "Scenarios",
                str(int(scenario_suite.get("scenario_count", 0))),
            )
            suite_summary_cols[1].metric(
                "With Trades",
                str(int(scenario_suite.get("with_trades_count", 0))),
            )
            suite_summary_cols[2].metric(
                "Profitable",
                str(int(scenario_suite.get("profitable_count", 0))),
            )
            scenario_rows = scenario_suite.get("scenarios", [])
            if isinstance(scenario_rows, list) and scenario_rows:
                sdf = pd.DataFrame(scenario_rows)
                for col in [
                    "trades",
                    "executed_buy_count",
                    "executed_sell_count",
                    "signal_buy_count",
                    "signal_sell_count",
                    "buy_blocked_expected_profit_count",
                    "net_profit_usd",
                    "max_drawdown_usd",
                    "win_rate_pct",
                    "approx_required_edge_bps_for_buy",
                    "max_observed_expected_edge_bps",
                ]:
                    if col in sdf.columns:
                        sdf[col] = pd.to_numeric(sdf[col], errors="coerce")
                if "scenario" in sdf.columns:
                    sdf = sdf.set_index("scenario")
                if "trades" in sdf.columns:
                    st.bar_chart(sdf[["trades"]], height=180)
                if "net_profit_usd" in sdf.columns:
                    st.bar_chart(sdf[["net_profit_usd"]], height=180)
                visible_cols = [
                    "trades",
                    "executed_buy_count",
                    "executed_sell_count",
                    "signal_buy_count",
                    "signal_sell_count",
                    "buy_blocked_expected_profit_count",
                    "net_profit_usd",
                    "max_drawdown_usd",
                    "win_rate_pct",
                    "approx_required_edge_bps_for_buy",
                    "max_observed_expected_edge_bps",
                ]
                present = [col for col in visible_cols if col in sdf.columns]
                st.dataframe(sdf[present], width="stretch", height=260)


def _compute_refresh_interval(config: BotConfig) -> int | None:
    if not bool(st.session_state.get("gui_loop_enabled")):
        return None
    if st.session_state.get("gui_loop_mode") == "websocket":
        return 1
    return int(max(1, config.price_refresh_seconds))


def _maybe_run_loop_cycle(bot: TradeBot, config: BotConfig, *, execute_live: bool) -> None:
    if not bool(st.session_state.get("gui_loop_enabled", False)):
        return
    if bool(st.session_state.pop("gui_skip_loop_once", False)):
        return
    marker = bot.market_data_marker(config.product_id)
    last_marker = st.session_state.get("gui_last_loop_data_marker")
    should_run = marker is None or marker != last_marker
    st.session_state["gui_loop_mode"] = "websocket" if marker is not None else "fallback_poll"
    if not should_run:
        return
    try:
        result = bot.run_cycle(execute_live=execute_live)
        line = bot.format_cycle_result(result)
        st.session_state["last_cycle"] = _cycle_to_dict(result, line)
        if marker is not None:
            st.session_state["gui_last_loop_data_marker"] = marker
        st.session_state.pop("gui_loop_error", None)
    except Exception as exc:
        st.session_state["gui_loop_error"] = str(exc)
        st.session_state["gui_loop_enabled"] = False


def _render_runtime_panel(
    bot: TradeBot,
    config: BotConfig,
    *,
    execute_live: bool,
    refresh_interval: int | None,
) -> None:
    _maybe_run_loop_cycle(bot, config, execute_live=execute_live)

    state, metrics = _load_runtime_files(config)
    _render_hero(config, metrics, refresh_interval=refresh_interval)

    if st.session_state.get("gui_loop_enabled"):
        if st.session_state.get("gui_loop_mode") == "websocket":
            st.info("Loop running: websocket-triggered on new market ticks.")
        else:
            st.info("Loop running: websocket unavailable, using fallback polling.")
    if "gui_loop_error" in st.session_state:
        st.error(f"Loop stopped due to error: {st.session_state['gui_loop_error']}")

    tab_overview, tab_live, tab_perf, tab_trades, tab_ops = st.tabs(
        ["Overview", "Live Signal", "Performance", "Trades", "Operations"]
    )

    with tab_overview:
        _render_summary_cards(metrics, state, config)
        _render_position_cards(state, config)

    with tab_live:
        _render_last_cycle_panel(config)

    with tab_perf:
        _render_equity_charts(metrics)

    with tab_trades:
        _render_trade_charts(metrics)

    with tab_ops:
        st.subheader("API Status")
        if "api_status_error" in st.session_state:
            st.warning(f"Status refresh error: {st.session_state['api_status_error']}")
        status = st.session_state.get("api_status")
        if isinstance(status, dict):
            ws = status.get("websocket", {})
            ws = ws if isinstance(ws, dict) else {}
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("WS Enabled", "Yes" if ws.get("enabled") else "No")
            s2.metric("WS Connected", "Yes" if ws.get("connected") else "No")
            s3.metric("WS Running", "Yes" if ws.get("running") else "No")
            age = ws.get("last_message_age_seconds")
            s4.metric("WS Age (s)", f"{_to_float(age):.2f}" if age is not None else "n/a")
            with st.expander("Full API Status JSON"):
                st.json(status)


def main() -> None:
    args = _parse_args()
    st.set_page_config(page_title="Bitcoin Trade Bot Dashboard", layout="wide")
    _hide_streamlit_status_widget()
    _inject_dashboard_theme()
    _load_dotenv_if_present()

    if "gui_loop_enabled" not in st.session_state:
        st.session_state["gui_loop_enabled"] = False
    if "gui_last_loop_data_marker" not in st.session_state:
        st.session_state["gui_last_loop_data_marker"] = None
    if "gui_loop_mode" not in st.session_state:
        st.session_state["gui_loop_mode"] = "websocket"

    st.title("Bitcoin Trade Bot Dashboard")
    st.caption("Optional GUI for non-technical operation and visualization.")

    with st.sidebar:
        st.header("Bot Controls")
        st.caption("1) Choose config and live intent. 2) Run one cycle or loop. 3) Review tabs.")
        config_path = st.text_input("Config Path", value=str(args.config))
        execute_live_request = st.checkbox(
            "Enable execute-live",
            value=False,
            help="Applies to Run One Cycle and Loop. Still requires mode=live and TRADEBOT_ENABLE_LIVE=true.",
        )
        confirm_live = st.checkbox(
            "I understand live orders can execute real trades",
            value=False,
            disabled=not execute_live_request,
        )
        run_cycle_clicked = st.button("Run One Cycle Now", type="primary")
        st.divider()
        st.subheader("Continuous Loop")
        st.caption("Event-driven loop: runs a cycle when new websocket market data arrives.")
        loop_c1, loop_c2 = st.columns(2)
        start_loop_clicked = loop_c1.button("Start Loop")
        stop_loop_clicked = loop_c2.button("Stop Loop")
        if start_loop_clicked:
            st.session_state["gui_loop_enabled"] = True
            st.session_state["gui_last_loop_data_marker"] = None
            st.session_state.pop("gui_loop_error", None)
        if stop_loop_clicked:
            st.session_state["gui_loop_enabled"] = False
        st.caption(f"Loop status: {'ON' if st.session_state.get('gui_loop_enabled') else 'OFF'}")
        st.divider()
        refresh_api_status = st.button("Refresh API Status")

    config_error = None
    try:
        config = load_config(config_path)
    except Exception as exc:
        config = None
        config_error = str(exc)

    if config is None:
        st.error(f"Failed to load config: {config_error}")
        return

    st.caption(
        f"Mode: `{config.mode}` | Product: `{config.product_id}` | Metrics: `{config.metrics_file}`"
    )
    if execute_live_request and not confirm_live:
        st.warning("Live requested but not confirmed. Run Cycle will execute in safe preview mode.")
    if execute_live_request and not _live_env_enabled():
        detected = _live_env_raw()
        st.warning(
            "Live execution is blocked in this app process because "
            "TRADEBOT_ENABLE_LIVE is not truthy. Set it in the same shell before launching "
            "the GUI, then restart. PowerShell example: "
            "`$env:TRADEBOT_ENABLE_LIVE='true'; python -m tradebot --config config.json gui`. "
            f"Current process value: {detected!r}"
        )

    bot = _get_or_create_bot(config_path, config)

    if run_cycle_clicked:
        execute_live = bool(execute_live_request and confirm_live)
        st.session_state["gui_skip_loop_once"] = True
        try:
            result = bot.run_cycle(execute_live=execute_live)
            line = bot.format_cycle_result(result)
            st.session_state["last_cycle"] = _cycle_to_dict(result, line)
            st.success(f"Cycle completed with action={result.action}")
        except Exception as exc:
            st.error(f"Cycle failed: {exc}")

    if refresh_api_status:
        try:
            st.session_state["api_status"] = bot.status()
            st.session_state.pop("api_status_error", None)
        except Exception as exc:
            st.session_state["api_status_error"] = str(exc)

    if "api_status" not in st.session_state and "api_status_error" not in st.session_state:
        try:
            st.session_state["api_status"] = bot.status()
        except Exception as exc:
            st.session_state["api_status_error"] = str(exc)

    refresh_interval = _compute_refresh_interval(config)
    if refresh_interval is not None:
        st.caption(f"Live panel updates every {refresh_interval} seconds.")

    @st.fragment(run_every=refresh_interval)
    def _live_runtime_fragment() -> None:
        execute_live = bool(execute_live_request and confirm_live)
        _render_runtime_panel(
            bot,
            config,
            execute_live=execute_live,
            refresh_interval=refresh_interval,
        )

    _live_runtime_fragment()

    _render_backtest_panel(bot)

    state, metrics = _load_runtime_files(config)
    with st.expander("Download Data"):
        state_json = json.dumps(state, indent=2)
        metrics_json = json.dumps(metrics, indent=2)
        st.download_button(
            label="Download State JSON",
            data=state_json,
            file_name=Path(config.state_file).name,
            mime="application/json",
        )
        st.download_button(
            label="Download Metrics JSON",
            data=metrics_json,
            file_name=Path(config.metrics_file).name,
            mime="application/json",
        )


if __name__ == "__main__":
    main()
