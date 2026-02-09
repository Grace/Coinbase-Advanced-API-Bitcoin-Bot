from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev

from .config import StrategySettings


@dataclass(slots=True)
class Signal:
    action: str
    reason: str
    expected_edge_bps: float
    trend_bps: float
    rsi: float
    threshold_bps: float = 0.0
    volatility_bps: float = 0.0
    regime: str = "normal"


@dataclass(slots=True)
class _Scores:
    trend_bps: float
    momentum_bps: float
    rsi: float
    bullish_score: float
    bearish_score: float
    expected_long_edge_bps: float
    expected_short_edge_bps: float


def _signal(
    *,
    action: str,
    reason: str,
    expected_edge_bps: float,
    trend_bps: float,
    rsi_value: float,
    threshold_bps: float,
    volatility_bps: float,
    regime: str,
) -> Signal:
    return Signal(
        action=action,
        reason=reason,
        expected_edge_bps=expected_edge_bps,
        trend_bps=trend_bps,
        rsi=rsi_value,
        threshold_bps=threshold_bps,
        volatility_bps=volatility_bps,
        regime=regime,
    )


def ema(values: list[float], period: int) -> float:
    if not values:
        raise ValueError("ema needs at least one value")
    if period <= 0:
        raise ValueError("ema period must be > 0")
    alpha = 2.0 / (period + 1.0)
    out = values[0]
    for value in values[1:]:
        out = (value - out) * alpha + out
    return out


def rsi(values: list[float], period: int = 14) -> float:
    if len(values) < period + 1:
        raise ValueError("rsi needs at least period + 1 values")
    gains = []
    losses = []
    for idx in range(1, period + 1):
        delta = values[idx] - values[idx - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    for idx in range(period + 1, len(values)):
        delta = values[idx] - values[idx - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _realized_volatility_bps(values: list[float], lookback: int) -> float:
    if len(values) < 2:
        return 0.0
    window = values[-max(lookback, 5) :]
    returns_bps: list[float] = []
    for idx in range(1, len(window)):
        prev = max(window[idx - 1], 1e-9)
        ret_bps = ((window[idx] - window[idx - 1]) / prev) * 10_000.0
        returns_bps.append(ret_bps)
    if not returns_bps:
        return 0.0
    if len(returns_bps) == 1:
        return abs(returns_bps[0])
    return float(pstdev(returns_bps))


def _regime_threshold_multiplier(
    *,
    trend_bps: float,
    volatility_bps: float,
    settings: StrategySettings,
) -> tuple[float, str]:
    if not settings.enable_regime_adaptation:
        return 1.0, "normal"
    if volatility_bps >= settings.high_volatility_bps:
        return float(settings.high_vol_threshold_multiplier), "high_volatility"
    if abs(trend_bps) >= settings.trend_regime_threshold_bps:
        return float(settings.trend_threshold_multiplier), "trending"
    if volatility_bps <= settings.low_volatility_bps:
        return float(settings.low_vol_threshold_multiplier), "low_volatility"
    return float(settings.choppy_threshold_multiplier), "choppy"


def _compute_scores(closes: list[float], settings: StrategySettings) -> _Scores:
    fast = ema(closes[-(settings.fast_ema_period * 3) :], settings.fast_ema_period)
    slow = ema(closes[-(settings.slow_ema_period * 3) :], settings.slow_ema_period)
    last_price = closes[-1]
    trend_bps = ((fast - slow) / max(slow, 1e-9)) * 10_000.0
    recent_lookback = min(6, len(closes) - 1)
    momentum_bps = (
        (last_price - closes[-1 - recent_lookback]) / max(closes[-1 - recent_lookback], 1e-9)
    ) * 10_000.0
    rsi_value = rsi(closes[-(settings.rsi_period + 30) :], settings.rsi_period)
    bullish_score = (0.65 * trend_bps) + (0.35 * momentum_bps) + max(0.0, 55.0 - rsi_value)
    bearish_score = (-0.65 * trend_bps) + max(0.0, rsi_value - 68.0)
    return _Scores(
        trend_bps=trend_bps,
        momentum_bps=momentum_bps,
        rsi=rsi_value,
        bullish_score=bullish_score,
        bearish_score=bearish_score,
        expected_long_edge_bps=max(0.0, bullish_score * 0.55),
        expected_short_edge_bps=max(0.0, bearish_score * 0.55),
    )


def _compute_subminute_scores(prices: list[float], settings: StrategySettings) -> _Scores:
    fast = ema(
        prices[-(settings.subminute_fast_ema_period * 3) :],
        settings.subminute_fast_ema_period,
    )
    slow = ema(
        prices[-(settings.subminute_slow_ema_period * 3) :],
        settings.subminute_slow_ema_period,
    )
    last_price = prices[-1]
    trend_bps = ((fast - slow) / max(slow, 1e-9)) * 10_000.0
    recent_lookback = min(8, len(prices) - 1)
    momentum_bps = (
        (last_price - prices[-1 - recent_lookback]) / max(prices[-1 - recent_lookback], 1e-9)
    ) * 10_000.0
    rsi_value = rsi(prices[-(settings.subminute_rsi_period + 30) :], settings.subminute_rsi_period)
    bullish_score = (0.7 * trend_bps) + (0.3 * momentum_bps) + max(0.0, 54.0 - rsi_value)
    bearish_score = (-0.7 * trend_bps) + max(0.0, rsi_value - 66.0)
    return _Scores(
        trend_bps=trend_bps,
        momentum_bps=momentum_bps,
        rsi=rsi_value,
        bullish_score=bullish_score,
        bearish_score=bearish_score,
        expected_long_edge_bps=max(0.0, bullish_score * 0.65),
        expected_short_edge_bps=max(0.0, bearish_score * 0.65),
    )


def generate_signal(
    closes: list[float],
    roundtrip_fee_bps: float,
    settings: StrategySettings,
    has_open_long: bool,
) -> Signal:
    min_needed = max(settings.slow_ema_period + 2, settings.rsi_period + 2)
    if len(closes) < min_needed:
        return _signal(
            action="HOLD",
            reason=f"not enough candles ({len(closes)}/{min_needed})",
            expected_edge_bps=0.0,
            trend_bps=0.0,
            rsi_value=50.0,
            threshold_bps=0.0,
            volatility_bps=0.0,
            regime="insufficient_data",
        )

    scores = _compute_scores(closes, settings)
    volatility_bps = _realized_volatility_bps(
        closes,
        int(settings.regime_volatility_lookback),
    )
    threshold_multiplier, regime = _regime_threshold_multiplier(
        trend_bps=scores.trend_bps,
        volatility_bps=volatility_bps,
        settings=settings,
    )

    threshold = (roundtrip_fee_bps + settings.min_signal_strength_bps) * threshold_multiplier
    exit_threshold = settings.min_signal_strength_bps * threshold_multiplier
    if not has_open_long:
        if scores.bullish_score > threshold and scores.rsi < 70.0:
            return _signal(
                action="BUY",
                reason="trend and momentum are net-positive after fee buffer",
                expected_edge_bps=scores.expected_long_edge_bps,
                trend_bps=scores.trend_bps,
                rsi_value=scores.rsi,
                threshold_bps=threshold,
                volatility_bps=volatility_bps,
                regime=regime,
            )
        return _signal(
            action="HOLD",
            reason="no buy edge after fees",
            expected_edge_bps=scores.expected_long_edge_bps,
            trend_bps=scores.trend_bps,
            rsi_value=scores.rsi,
            threshold_bps=threshold,
            volatility_bps=volatility_bps,
            regime=regime,
        )

    if scores.bearish_score > exit_threshold or scores.rsi > 76.0:
        return _signal(
            action="SELL",
            reason="trend weakened / overbought",
            expected_edge_bps=scores.expected_long_edge_bps,
            trend_bps=scores.trend_bps,
            rsi_value=scores.rsi,
            threshold_bps=exit_threshold,
            volatility_bps=volatility_bps,
            regime=regime,
        )
    return _signal(
        action="HOLD",
        reason="holding long position",
        expected_edge_bps=scores.expected_long_edge_bps,
        trend_bps=scores.trend_bps,
        rsi_value=scores.rsi,
        threshold_bps=exit_threshold,
        volatility_bps=volatility_bps,
        regime=regime,
    )


def generate_short_signal(
    closes: list[float],
    roundtrip_fee_bps: float,
    settings: StrategySettings,
    has_open_short: bool,
) -> Signal:
    min_needed = max(settings.slow_ema_period + 2, settings.rsi_period + 2)
    if len(closes) < min_needed:
        return _signal(
            action="HOLD",
            reason=f"not enough candles ({len(closes)}/{min_needed})",
            expected_edge_bps=0.0,
            trend_bps=0.0,
            rsi_value=50.0,
            threshold_bps=0.0,
            volatility_bps=0.0,
            regime="insufficient_data",
        )

    scores = _compute_scores(closes, settings)
    volatility_bps = _realized_volatility_bps(
        closes,
        int(settings.regime_volatility_lookback),
    )
    threshold_multiplier, regime = _regime_threshold_multiplier(
        trend_bps=scores.trend_bps,
        volatility_bps=volatility_bps,
        settings=settings,
    )
    threshold = (roundtrip_fee_bps + settings.min_signal_strength_bps) * threshold_multiplier
    exit_threshold = settings.min_signal_strength_bps * threshold_multiplier
    if not has_open_short:
        if scores.bearish_score > threshold and scores.rsi > 30.0:
            return _signal(
                action="SHORT",
                reason="downtrend momentum is net-positive after fee buffer",
                expected_edge_bps=scores.expected_short_edge_bps,
                trend_bps=scores.trend_bps,
                rsi_value=scores.rsi,
                threshold_bps=threshold,
                volatility_bps=volatility_bps,
                regime=regime,
            )
        return _signal(
            action="HOLD",
            reason="no short edge after fees",
            expected_edge_bps=scores.expected_short_edge_bps,
            trend_bps=scores.trend_bps,
            rsi_value=scores.rsi,
            threshold_bps=threshold,
            volatility_bps=volatility_bps,
            regime=regime,
        )

    if scores.bullish_score > exit_threshold or scores.rsi < 24.0:
        return _signal(
            action="CLOSE_SHORT",
            reason="short momentum weakened / oversold",
            expected_edge_bps=scores.expected_short_edge_bps,
            trend_bps=scores.trend_bps,
            rsi_value=scores.rsi,
            threshold_bps=exit_threshold,
            volatility_bps=volatility_bps,
            regime=regime,
        )
    return _signal(
        action="HOLD",
        reason="holding short position",
        expected_edge_bps=scores.expected_short_edge_bps,
        trend_bps=scores.trend_bps,
        rsi_value=scores.rsi,
        threshold_bps=exit_threshold,
        volatility_bps=volatility_bps,
        regime=regime,
    )


def generate_subminute_signal(
    prices: list[float],
    roundtrip_fee_bps: float,
    settings: StrategySettings,
    has_open_long: bool,
) -> Signal:
    min_needed = max(
        settings.subminute_min_samples,
        settings.subminute_slow_ema_period + 2,
        settings.subminute_rsi_period + 2,
    )
    if len(prices) < min_needed:
        return _signal(
            action="HOLD",
            reason=f"not enough sub-minute ticks ({len(prices)}/{min_needed})",
            expected_edge_bps=0.0,
            trend_bps=0.0,
            rsi_value=50.0,
            threshold_bps=0.0,
            volatility_bps=0.0,
            regime="insufficient_data",
        )

    scores = _compute_subminute_scores(prices, settings)
    volatility_bps = _realized_volatility_bps(
        prices,
        int(settings.regime_volatility_lookback),
    )
    threshold_multiplier, regime = _regime_threshold_multiplier(
        trend_bps=scores.trend_bps,
        volatility_bps=volatility_bps,
        settings=settings,
    )
    threshold = (roundtrip_fee_bps + settings.subminute_min_signal_strength_bps) * threshold_multiplier
    exit_threshold = settings.subminute_min_signal_strength_bps * threshold_multiplier
    if not has_open_long:
        if scores.bullish_score > threshold and scores.rsi < 78.0:
            return _signal(
                action="BUY",
                reason="sub-minute momentum is net-positive after fee buffer",
                expected_edge_bps=scores.expected_long_edge_bps,
                trend_bps=scores.trend_bps,
                rsi_value=scores.rsi,
                threshold_bps=threshold,
                volatility_bps=volatility_bps,
                regime=regime,
            )
        return _signal(
            action="HOLD",
            reason="no sub-minute buy edge after fees",
            expected_edge_bps=scores.expected_long_edge_bps,
            trend_bps=scores.trend_bps,
            rsi_value=scores.rsi,
            threshold_bps=threshold,
            volatility_bps=volatility_bps,
            regime=regime,
        )

    if scores.bearish_score > exit_threshold or scores.rsi > 82.0:
        return _signal(
            action="SELL",
            reason="sub-minute long momentum weakened / overbought",
            expected_edge_bps=scores.expected_long_edge_bps,
            trend_bps=scores.trend_bps,
            rsi_value=scores.rsi,
            threshold_bps=exit_threshold,
            volatility_bps=volatility_bps,
            regime=regime,
        )
    return _signal(
        action="HOLD",
        reason="holding long position on sub-minute signal",
        expected_edge_bps=scores.expected_long_edge_bps,
        trend_bps=scores.trend_bps,
        rsi_value=scores.rsi,
        threshold_bps=exit_threshold,
        volatility_bps=volatility_bps,
        regime=regime,
    )


def generate_subminute_short_signal(
    prices: list[float],
    roundtrip_fee_bps: float,
    settings: StrategySettings,
    has_open_short: bool,
) -> Signal:
    min_needed = max(
        settings.subminute_min_samples,
        settings.subminute_slow_ema_period + 2,
        settings.subminute_rsi_period + 2,
    )
    if len(prices) < min_needed:
        return _signal(
            action="HOLD",
            reason=f"not enough sub-minute ticks ({len(prices)}/{min_needed})",
            expected_edge_bps=0.0,
            trend_bps=0.0,
            rsi_value=50.0,
            threshold_bps=0.0,
            volatility_bps=0.0,
            regime="insufficient_data",
        )

    scores = _compute_subminute_scores(prices, settings)
    volatility_bps = _realized_volatility_bps(
        prices,
        int(settings.regime_volatility_lookback),
    )
    threshold_multiplier, regime = _regime_threshold_multiplier(
        trend_bps=scores.trend_bps,
        volatility_bps=volatility_bps,
        settings=settings,
    )
    threshold = (roundtrip_fee_bps + settings.subminute_min_signal_strength_bps) * threshold_multiplier
    exit_threshold = settings.subminute_min_signal_strength_bps * threshold_multiplier
    if not has_open_short:
        if scores.bearish_score > threshold and scores.rsi > 22.0:
            return _signal(
                action="SHORT",
                reason="sub-minute downtrend is net-positive after fee buffer",
                expected_edge_bps=scores.expected_short_edge_bps,
                trend_bps=scores.trend_bps,
                rsi_value=scores.rsi,
                threshold_bps=threshold,
                volatility_bps=volatility_bps,
                regime=regime,
            )
        return _signal(
            action="HOLD",
            reason="no sub-minute short edge after fees",
            expected_edge_bps=scores.expected_short_edge_bps,
            trend_bps=scores.trend_bps,
            rsi_value=scores.rsi,
            threshold_bps=threshold,
            volatility_bps=volatility_bps,
            regime=regime,
        )

    if scores.bullish_score > exit_threshold or scores.rsi < 18.0:
        return _signal(
            action="CLOSE_SHORT",
            reason="sub-minute short momentum weakened / oversold",
            expected_edge_bps=scores.expected_short_edge_bps,
            trend_bps=scores.trend_bps,
            rsi_value=scores.rsi,
            threshold_bps=exit_threshold,
            volatility_bps=volatility_bps,
            regime=regime,
        )
    return _signal(
        action="HOLD",
        reason="holding short position on sub-minute signal",
        expected_edge_bps=scores.expected_short_edge_bps,
        trend_bps=scores.trend_bps,
        rsi_value=scores.rsi,
        threshold_bps=exit_threshold,
        volatility_bps=volatility_bps,
        regime=regime,
    )
