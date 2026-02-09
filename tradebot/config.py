from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class StrategySettings:
    fast_ema_period: int = 12
    slow_ema_period: int = 26
    rsi_period: int = 14
    min_signal_strength_bps: float = 20.0
    min_expected_profit_usd: float = 1.0
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    slippage_buffer_bps: float = 12.0
    enable_subminute_signals: bool = False
    subminute_window_seconds: int = 90
    subminute_min_samples: int = 20
    subminute_fast_ema_period: int = 8
    subminute_slow_ema_period: int = 21
    subminute_rsi_period: int = 14
    subminute_min_signal_strength_bps: float = 3.0
    enable_regime_adaptation: bool = True
    regime_volatility_lookback: int = 30
    low_volatility_bps: float = 18.0
    high_volatility_bps: float = 75.0
    trend_regime_threshold_bps: float = 30.0
    low_vol_threshold_multiplier: float = 0.9
    trend_threshold_multiplier: float = 0.85
    choppy_threshold_multiplier: float = 1.1
    high_vol_threshold_multiplier: float = 1.3


@dataclass(slots=True)
class GuardrailSettings:
    max_order_usd: float = 25.0
    min_order_usd: float = 10.0
    max_position_usd: float = 250.0
    min_usd_reserve: float = 100.0
    max_daily_buy_usd: float = 100.0
    max_daily_short_open_usd: float = 100.0
    max_daily_loss_usd: float = 30.0
    daily_profit_target_usd: float = 0.0
    max_trades_per_day: int = 6
    cooldown_seconds: int = 900
    allow_convert: bool = True
    allow_short: bool = False
    max_short_notional_usd: float = 50.0
    max_short_leverage: float = 1.5
    allowed_products: list[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD"])
    allowed_convert_pairs: list[str] = field(
        default_factory=lambda: ["USD:USDC", "USDC:USD"]
    )


@dataclass(slots=True)
class PaperSettings:
    starting_usd: float = 10_000.0


@dataclass(slots=True)
class ExecutionSettings:
    prefer_maker_orders: bool = False
    maker_price_offset_bps: float = 1.0
    maker_max_order_wait_seconds: int = 1200
    max_spread_bps: float = 12.0


@dataclass(slots=True)
class WebsocketSettings:
    enable: bool = True
    market_data_url: str = "wss://advanced-trade-ws.coinbase.com"
    channel: str = "ticker"
    stale_seconds: float = 5.0
    ping_interval_seconds: int = 20
    ping_timeout_seconds: int = 10
    max_reconnect_seconds: int = 30
    subscribe_heartbeats: bool = True


@dataclass(slots=True)
class AutoActionSettings:
    enable_auto_short: bool = False
    enable_auto_close_position: bool = False
    enable_auto_convert: bool = False
    short_check_interval_seconds: float = 5.0
    close_check_interval_seconds: float = 2.0
    convert_check_interval_seconds: float = 60.0
    short_product_id: str = "BTC-PERP"
    short_product_type: str = "FUTURE"
    short_order_usd: float = 10.0
    short_leverage: float = 1.0
    short_margin_type: str = "CROSS"
    min_expected_short_profit_usd: float = 0.03
    convert_pair: str = "USD:USDC"
    convert_amount: float = 5.0
    min_convert_profit_usd: float = 0.01


@dataclass(slots=True)
class BotConfig:
    mode: str = "paper"
    key_file: str = "./cdp_api_key.json"
    product_id: str = "BTC-USD"
    product_type: str = "SPOT"
    use_color_output: bool = True
    granularity: str = "ONE_HOUR"
    candles_lookback: int = 300
    loop_seconds: float = 300.0
    price_refresh_seconds: float = 1.0
    candles_refresh_seconds: int = 15
    fees_refresh_seconds: int = 300
    state_file: str = "./tradebot_state.json"
    metrics_file: str = "./tradebot_metrics.json"
    trade_log_limit: int = 1000
    metrics_trade_limit: int = 5000
    metrics_equity_limit: int = 20000
    strategy: StrategySettings = field(default_factory=StrategySettings)
    guardrails: GuardrailSettings = field(default_factory=GuardrailSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    websocket: WebsocketSettings = field(default_factory=WebsocketSettings)
    auto_actions: AutoActionSettings = field(default_factory=AutoActionSettings)
    paper: PaperSettings = field(default_factory=PaperSettings)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    return json.loads(raw)


def _apply_overrides(default_obj: Any, overrides: dict[str, Any]) -> Any:
    values: dict[str, Any] = {}
    for meta in fields(default_obj):
        name = meta.name
        value = getattr(default_obj, name)
        if name not in overrides:
            values[name] = value
            continue
        override = overrides[name]
        if is_dataclass(value) and isinstance(override, dict):
            values[name] = _apply_overrides(value, override)
        else:
            values[name] = override
    return type(default_obj)(**values)


def load_config(path: str | Path | None = None) -> BotConfig:
    cfg = BotConfig()
    if path is None:
        return _normalize_config(cfg)
    path_obj = Path(path)
    overrides = _read_json(path_obj)
    if not overrides:
        return _normalize_config(cfg)
    merged = _apply_overrides(cfg, overrides)
    return _normalize_config(merged)


def _normalize_config(config: BotConfig) -> BotConfig:
    mode = config.mode.lower().strip()
    if mode not in {"paper", "live"}:
        raise ValueError("config.mode must be 'paper' or 'live'")
    config.mode = mode
    if float(config.loop_seconds) <= 0:
        raise ValueError("config.loop_seconds must be > 0")
    if float(config.price_refresh_seconds) <= 0:
        raise ValueError("config.price_refresh_seconds must be > 0")
    if int(config.candles_refresh_seconds) <= 0:
        raise ValueError("config.candles_refresh_seconds must be > 0")
    if int(config.fees_refresh_seconds) <= 0:
        raise ValueError("config.fees_refresh_seconds must be > 0")
    if float(config.auto_actions.short_check_interval_seconds) <= 0:
        raise ValueError("auto_actions.short_check_interval_seconds must be > 0")
    if float(config.auto_actions.close_check_interval_seconds) <= 0:
        raise ValueError("auto_actions.close_check_interval_seconds must be > 0")
    if float(config.auto_actions.convert_check_interval_seconds) <= 0:
        raise ValueError("auto_actions.convert_check_interval_seconds must be > 0")
    if config.auto_actions.short_order_usd <= 0:
        raise ValueError("auto_actions.short_order_usd must be > 0")
    if config.auto_actions.short_leverage <= 0:
        raise ValueError("auto_actions.short_leverage must be > 0")
    if config.auto_actions.convert_amount <= 0:
        raise ValueError("auto_actions.convert_amount must be > 0")
    if float(config.guardrails.daily_profit_target_usd) < 0:
        raise ValueError("guardrails.daily_profit_target_usd must be >= 0")
    if float(config.execution.maker_price_offset_bps) <= 0:
        raise ValueError("execution.maker_price_offset_bps must be > 0")
    if int(config.execution.maker_max_order_wait_seconds) <= 0:
        raise ValueError("execution.maker_max_order_wait_seconds must be > 0")
    if float(config.execution.max_spread_bps) <= 0:
        raise ValueError("execution.max_spread_bps must be > 0")
    if float(config.websocket.stale_seconds) <= 0:
        raise ValueError("websocket.stale_seconds must be > 0")
    if int(config.websocket.ping_interval_seconds) <= 0:
        raise ValueError("websocket.ping_interval_seconds must be > 0")
    if int(config.websocket.ping_timeout_seconds) <= 0:
        raise ValueError("websocket.ping_timeout_seconds must be > 0")
    if int(config.websocket.max_reconnect_seconds) <= 0:
        raise ValueError("websocket.max_reconnect_seconds must be > 0")
    if not str(config.websocket.market_data_url).strip():
        raise ValueError("websocket.market_data_url must be non-empty")
    channel = str(config.websocket.channel).strip().lower()
    if channel not in {"ticker", "ticker_batch"}:
        raise ValueError("websocket.channel must be one of: ticker, ticker_batch")
    config.websocket.channel = channel
    if int(config.strategy.subminute_window_seconds) <= 0:
        raise ValueError("strategy.subminute_window_seconds must be > 0")
    if int(config.strategy.subminute_min_samples) < 5:
        raise ValueError("strategy.subminute_min_samples must be >= 5")
    if int(config.strategy.subminute_fast_ema_period) <= 0:
        raise ValueError("strategy.subminute_fast_ema_period must be > 0")
    if int(config.strategy.subminute_slow_ema_period) <= 0:
        raise ValueError("strategy.subminute_slow_ema_period must be > 0")
    if (
        int(config.strategy.subminute_fast_ema_period)
        >= int(config.strategy.subminute_slow_ema_period)
    ):
        raise ValueError("strategy.subminute_fast_ema_period must be < subminute_slow_ema_period")
    if int(config.strategy.subminute_rsi_period) <= 1:
        raise ValueError("strategy.subminute_rsi_period must be > 1")
    if float(config.strategy.subminute_min_signal_strength_bps) < 0:
        raise ValueError("strategy.subminute_min_signal_strength_bps must be >= 0")
    if int(config.strategy.regime_volatility_lookback) < 5:
        raise ValueError("strategy.regime_volatility_lookback must be >= 5")
    if float(config.strategy.low_volatility_bps) < 0:
        raise ValueError("strategy.low_volatility_bps must be >= 0")
    if float(config.strategy.high_volatility_bps) <= float(config.strategy.low_volatility_bps):
        raise ValueError("strategy.high_volatility_bps must be > low_volatility_bps")
    if float(config.strategy.trend_regime_threshold_bps) < 0:
        raise ValueError("strategy.trend_regime_threshold_bps must be >= 0")
    if float(config.strategy.low_vol_threshold_multiplier) <= 0:
        raise ValueError("strategy.low_vol_threshold_multiplier must be > 0")
    if float(config.strategy.trend_threshold_multiplier) <= 0:
        raise ValueError("strategy.trend_threshold_multiplier must be > 0")
    if float(config.strategy.choppy_threshold_multiplier) <= 0:
        raise ValueError("strategy.choppy_threshold_multiplier must be > 0")
    if float(config.strategy.high_vol_threshold_multiplier) <= 0:
        raise ValueError("strategy.high_vol_threshold_multiplier must be > 0")
    if config.product_id not in config.guardrails.allowed_products:
        config.guardrails.allowed_products.append(config.product_id)
    return config
