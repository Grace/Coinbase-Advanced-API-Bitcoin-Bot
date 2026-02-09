from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import GuardrailSettings, StrategySettings
from .strategy import Signal, generate_signal


@dataclass(slots=True)
class _SimState:
    cash_usd: float
    base_size: float
    avg_entry_price: float
    realized_pnl_usd: float
    total_fees_usd: float
    wins: int
    losses: int
    trades: int
    peak_equity_usd: float
    max_drawdown_usd: float


@dataclass(slots=True)
class _SimDiagnostics:
    steps: int
    buy_signals: int
    sell_signals: int
    hold_signals: int
    executed_buys: int
    executed_sells: int
    buy_blocked_daily_profit_target: int
    buy_blocked_daily_loss: int
    buy_blocked_order_too_small: int
    buy_blocked_expected_profit: int
    max_observed_expected_edge_bps: float


def _apply_stop_rules(
    signal: Signal,
    settings: StrategySettings,
    has_open_long: bool,
    avg_entry_price: float,
    price: float,
) -> Signal:
    if not has_open_long or avg_entry_price <= 0:
        return signal
    pnl_pct = (price - avg_entry_price) / avg_entry_price
    if pnl_pct <= -float(settings.stop_loss_pct):
        return Signal(
            action="SELL",
            reason=f"stop loss triggered ({pnl_pct:.2%})",
            expected_edge_bps=signal.expected_edge_bps,
            trend_bps=signal.trend_bps,
            rsi=signal.rsi,
        )
    if pnl_pct >= float(settings.take_profit_pct):
        return Signal(
            action="SELL",
            reason=f"take profit triggered ({pnl_pct:.2%})",
            expected_edge_bps=signal.expected_edge_bps,
            trend_bps=signal.trend_bps,
            rsi=signal.rsi,
        )
    return signal


def walk_forward_backtest(
    closes: list[float],
    strategy: StrategySettings,
    guardrails: GuardrailSettings,
    initial_usd: float,
    maker_fee_rate: float,
    taker_fee_rate: float,
    prefer_maker_orders: bool,
    train_candles: int,
    test_candles: int,
    step_candles: int,
) -> dict[str, Any]:
    if train_candles <= 0:
        raise ValueError("train_candles must be > 0")
    if test_candles <= 0:
        raise ValueError("test_candles must be > 0")
    if step_candles <= 0:
        raise ValueError("step_candles must be > 0")
    if len(closes) < (train_candles + test_candles + 2):
        raise ValueError("not enough candles for requested walk-forward windows")

    fee_rate = float(maker_fee_rate if prefer_maker_orders else taker_fee_rate)
    roundtrip_fee_bps = (fee_rate * 2.0 * 10_000.0) + float(strategy.slippage_buffer_bps)
    slippage_frac = 0.0 if prefer_maker_orders else float(strategy.slippage_buffer_bps) / 10_000.0
    max_history = max(int(strategy.slow_ema_period) * 3 + 4, int(strategy.rsi_period) + 4, 100)

    sim = _SimState(
        cash_usd=float(initial_usd),
        base_size=0.0,
        avg_entry_price=0.0,
        realized_pnl_usd=0.0,
        total_fees_usd=0.0,
        wins=0,
        losses=0,
        trades=0,
        peak_equity_usd=float(initial_usd),
        max_drawdown_usd=0.0,
    )
    diagnostics = _SimDiagnostics(
        steps=0,
        buy_signals=0,
        sell_signals=0,
        hold_signals=0,
        executed_buys=0,
        executed_sells=0,
        buy_blocked_daily_profit_target=0,
        buy_blocked_daily_loss=0,
        buy_blocked_order_too_small=0,
        buy_blocked_expected_profit=0,
        max_observed_expected_edge_bps=0.0,
    )

    windows: list[dict[str, Any]] = []
    daily_buy_used = 0.0
    daily_pnl = 0.0
    current_day_bucket = -1

    for start in range(train_candles, len(closes) - test_candles + 1, step_candles):
        end = start + test_candles
        window_trade_start = sim.trades
        window_realized_start = sim.realized_pnl_usd
        window_equity_start = sim.cash_usd + (sim.base_size * closes[start - 1])

        for idx in range(start, end):
            price = float(closes[idx])
            diagnostics.steps += 1
            day_bucket = idx // 1440
            if day_bucket != current_day_bucket:
                current_day_bucket = day_bucket
                daily_buy_used = 0.0
                daily_pnl = 0.0

            history_start = max(0, idx - max_history + 1)
            history = closes[history_start : idx + 1]
            signal = generate_signal(
                closes=history,
                roundtrip_fee_bps=roundtrip_fee_bps,
                settings=strategy,
                has_open_long=sim.base_size > 0.0,
            )
            signal = _apply_stop_rules(
                signal=signal,
                settings=strategy,
                has_open_long=sim.base_size > 0.0,
                avg_entry_price=sim.avg_entry_price,
                price=price,
            )
            diagnostics.max_observed_expected_edge_bps = max(
                diagnostics.max_observed_expected_edge_bps,
                float(signal.expected_edge_bps),
            )

            if signal.action == "BUY":
                diagnostics.buy_signals += 1
            elif signal.action == "SELL":
                diagnostics.sell_signals += 1
            else:
                diagnostics.hold_signals += 1

            if signal.action == "BUY" and sim.base_size <= 0.0:
                if (
                    float(guardrails.daily_profit_target_usd) > 0
                    and daily_pnl >= float(guardrails.daily_profit_target_usd)
                ):
                    diagnostics.buy_blocked_daily_profit_target += 1
                    continue
                if daily_pnl <= -float(guardrails.max_daily_loss_usd):
                    diagnostics.buy_blocked_daily_loss += 1
                    continue
                max_spendable = max(0.0, sim.cash_usd - float(guardrails.min_usd_reserve))
                remaining_pos_cap = float(guardrails.max_position_usd)
                order_usd = min(float(guardrails.max_order_usd), remaining_pos_cap, max_spendable)
                remaining_daily_buy = max(0.0, float(guardrails.max_daily_buy_usd) - daily_buy_used)
                order_usd = min(order_usd, remaining_daily_buy)
                if order_usd < float(guardrails.min_order_usd):
                    diagnostics.buy_blocked_order_too_small += 1
                    continue

                expected_gross = order_usd * (float(signal.expected_edge_bps) / 10_000.0)
                expected_net = expected_gross - (order_usd * fee_rate * 2.0)
                if expected_net < float(strategy.min_expected_profit_usd):
                    diagnostics.buy_blocked_expected_profit += 1
                    continue

                buy_fee = order_usd * fee_rate
                fill_price = price * (1.0 + slippage_frac)
                bought_base = (order_usd - buy_fee) / max(fill_price, 1e-12)
                old_notional = sim.base_size * sim.avg_entry_price
                sim.base_size += bought_base
                sim.avg_entry_price = (
                    (old_notional + order_usd) / sim.base_size if sim.base_size > 0 else 0.0
                )
                sim.cash_usd -= order_usd
                sim.total_fees_usd += buy_fee
                sim.trades += 1
                daily_buy_used += order_usd
                diagnostics.executed_buys += 1

            elif signal.action == "SELL" and sim.base_size > 0.0:
                sell_base = sim.base_size
                fill_price = price * (1.0 - slippage_frac)
                gross = sell_base * fill_price
                sell_fee = gross * fee_rate
                net = gross - sell_fee
                realized = ((fill_price - sim.avg_entry_price) * sell_base) - sell_fee
                sim.cash_usd += net
                sim.base_size = 0.0
                sim.avg_entry_price = 0.0
                sim.realized_pnl_usd += realized
                sim.total_fees_usd += sell_fee
                sim.trades += 1
                daily_pnl += realized
                if realized > 0:
                    sim.wins += 1
                elif realized < 0:
                    sim.losses += 1
                diagnostics.executed_sells += 1

            equity = sim.cash_usd + (sim.base_size * price)
            if equity > sim.peak_equity_usd:
                sim.peak_equity_usd = equity
            drawdown = max(0.0, sim.peak_equity_usd - equity)
            if drawdown > sim.max_drawdown_usd:
                sim.max_drawdown_usd = drawdown

        window_equity_end = sim.cash_usd + (sim.base_size * closes[end - 1])
        windows.append(
            {
                "start_index": start,
                "end_index": end - 1,
                "trades": sim.trades - window_trade_start,
                "realized_pnl_usd": round(sim.realized_pnl_usd - window_realized_start, 4),
                "window_return_pct": round(
                    ((window_equity_end - window_equity_start) / max(window_equity_start, 1e-9)) * 100.0,
                    4,
                ),
                "equity_end_usd": round(window_equity_end, 4),
            }
        )

    final_price = float(closes[-1])
    unrealized = (
        (final_price - sim.avg_entry_price) * sim.base_size
        if sim.base_size > 0 and sim.avg_entry_price > 0
        else 0.0
    )
    net_profit = sim.realized_pnl_usd + unrealized
    reference_order_usd = max(
        1e-9,
        min(
            float(guardrails.max_order_usd),
            float(guardrails.max_position_usd),
            float(guardrails.max_daily_buy_usd),
        ),
    )
    approx_required_edge_bps = (
        (
            float(strategy.min_expected_profit_usd)
            + (reference_order_usd * fee_rate * 2.0)
        )
        / reference_order_usd
    ) * 10_000.0

    return {
        "summary": {
            "initial_usd": round(float(initial_usd), 4),
            "ending_cash_usd": round(sim.cash_usd, 4),
            "ending_base_size": round(sim.base_size, 8),
            "ending_price_usd": round(final_price, 4),
            "realized_pnl_usd": round(sim.realized_pnl_usd, 4),
            "unrealized_pnl_usd": round(unrealized, 4),
            "net_profit_usd": round(net_profit, 4),
            "total_fees_usd": round(sim.total_fees_usd, 4),
            "trades": int(sim.trades),
            "wins": int(sim.wins),
            "losses": int(sim.losses),
            "win_rate_pct": round(
                (sim.wins / max(1, (sim.wins + sim.losses))) * 100.0,
                2,
            ),
            "max_drawdown_usd": round(sim.max_drawdown_usd, 4),
            "windows": len(windows),
            "prefer_maker_orders": bool(prefer_maker_orders),
            "assumed_fee_rate": round(fee_rate, 6),
            "assumed_roundtrip_cost_bps": round((fee_rate * 2.0 * 10_000.0), 2),
            "approx_required_edge_bps_for_buy": round(approx_required_edge_bps, 2),
            "max_observed_expected_edge_bps": round(
                diagnostics.max_observed_expected_edge_bps, 2
            ),
            "signal_buy_count": int(diagnostics.buy_signals),
            "signal_sell_count": int(diagnostics.sell_signals),
            "signal_hold_count": int(diagnostics.hold_signals),
            "executed_buy_count": int(diagnostics.executed_buys),
            "executed_sell_count": int(diagnostics.executed_sells),
            "buy_blocked_expected_profit_count": int(
                diagnostics.buy_blocked_expected_profit
            ),
            "buy_blocked_order_too_small_count": int(
                diagnostics.buy_blocked_order_too_small
            ),
            "buy_blocked_daily_loss_count": int(diagnostics.buy_blocked_daily_loss),
            "buy_blocked_daily_profit_target_count": int(
                diagnostics.buy_blocked_daily_profit_target
            ),
        },
        "windows": windows,
    }


def _scenario_closes(name: str, length: int, start_price: float = 100.0) -> list[float]:
    if length < 50:
        raise ValueError("scenario length must be >= 50")
    price = max(1e-6, float(start_price))
    closes: list[float] = []
    for idx in range(length):
        if name == "bull_trend_pullbacks":
            pattern = [0.012, 0.012, -0.008, 0.012, -0.008, 0.012, -0.008, 0.012]
            change = pattern[idx % len(pattern)]
        elif name == "bear_trend_bounces":
            pattern = [-0.012, -0.012, 0.008, -0.012, 0.008, -0.012, 0.008, -0.012]
            change = pattern[idx % len(pattern)]
        elif name == "sideways_chop":
            pattern = [0.006, -0.005, 0.004, -0.004, 0.005, -0.006]
            change = pattern[idx % len(pattern)]
        elif name == "breakout_then_reversal":
            up = [0.012, 0.012, -0.008, 0.012, -0.008, 0.012, -0.008, 0.012]
            down = [-0.012, -0.012, 0.008, -0.012, 0.008, -0.012, 0.008, -0.012]
            pattern = up if idx < (length // 2) else down
            change = pattern[idx % len(pattern)]
        elif name == "volatility_spike_regime":
            pattern = [0.020, -0.010, 0.018, -0.009, 0.016, -0.008]
            change = pattern[idx % len(pattern)]
        else:
            raise ValueError(f"unknown scenario: {name}")
        price = max(1e-6, price * (1.0 + change))
        closes.append(price)
    return closes


def run_scenario_suite(
    *,
    strategy: StrategySettings,
    guardrails: GuardrailSettings,
    initial_usd: float,
    maker_fee_rate: float,
    taker_fee_rate: float,
    prefer_maker_orders: bool,
    train_candles: int,
    test_candles: int,
    step_candles: int,
    scenario_length: int = 1800,
) -> dict[str, Any]:
    scenario_names = [
        "bull_trend_pullbacks",
        "bear_trend_bounces",
        "sideways_chop",
        "breakout_then_reversal",
        "volatility_spike_regime",
    ]
    per_scenario: list[dict[str, Any]] = []
    for name in scenario_names:
        closes = _scenario_closes(name, length=max(int(scenario_length), train_candles + test_candles + 10))
        result = walk_forward_backtest(
            closes=closes,
            strategy=strategy,
            guardrails=guardrails,
            initial_usd=initial_usd,
            maker_fee_rate=maker_fee_rate,
            taker_fee_rate=taker_fee_rate,
            prefer_maker_orders=prefer_maker_orders,
            train_candles=train_candles,
            test_candles=test_candles,
            step_candles=step_candles,
        )
        summary = result.get("summary", {})
        per_scenario.append(
            {
                "scenario": name,
                "trades": int(summary.get("trades", 0)),
                "executed_buy_count": int(summary.get("executed_buy_count", 0)),
                "executed_sell_count": int(summary.get("executed_sell_count", 0)),
                "signal_buy_count": int(summary.get("signal_buy_count", 0)),
                "signal_sell_count": int(summary.get("signal_sell_count", 0)),
                "buy_blocked_expected_profit_count": int(
                    summary.get("buy_blocked_expected_profit_count", 0)
                ),
                "net_profit_usd": float(summary.get("net_profit_usd", 0.0)),
                "max_drawdown_usd": float(summary.get("max_drawdown_usd", 0.0)),
                "win_rate_pct": float(summary.get("win_rate_pct", 0.0)),
                "approx_required_edge_bps_for_buy": float(
                    summary.get("approx_required_edge_bps_for_buy", 0.0)
                ),
                "max_observed_expected_edge_bps": float(
                    summary.get("max_observed_expected_edge_bps", 0.0)
                ),
                "summary": summary,
            }
        )
    profitable = [row for row in per_scenario if row["net_profit_usd"] > 0]
    with_trades = [row for row in per_scenario if row["trades"] > 0]
    return {
        "scenario_count": len(per_scenario),
        "with_trades_count": len(with_trades),
        "profitable_count": len(profitable),
        "scenarios": per_scenario,
    }
