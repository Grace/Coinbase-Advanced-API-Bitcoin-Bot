# Bitcoin Trade Bot

Automatically trades Bitcoin via the Coinbase Advanced API.

<img width="1907" height="949" alt="image" src="https://github.com/user-attachments/assets/45c3e529-a2d4-456e-a8ac-f302997e4ff4" />

## What it does

- Uses Coinbase Advanced API market data + account data.
- Generates buy/sell signals from EMA trend + RSI momentum.
- Accounts for net fees by:
  - Reading your Coinbase Advanced Trade `SPOT` maker/taker fee tier.
  - Previewing order commissions before execution.
  - Requiring expected edge to clear fee + profit thresholds.
- Supports guarded commands for convert and short/close-position actions.
- Includes walk-forward backtest command for config evaluation.
- Supports optional auto-actions in loop mode (`auto_actions`):
  - auto short open
  - auto short close-position
  - auto convert
- Writes graph-ready analytics to `tradebot_metrics.json`:
  - `equity_curve`: time series for realized/unrealized/net PnL.
  - `trades`: normalized trade events with side, size, fees, and realized PnL.

## Important constraints

- Profit is never guaranteed.
- Default mode is `paper` (simulated trading only).
- Live execution is blocked unless all are true:
  - `config.mode = "live"`
  - CLI uses `--execute-live`
  - Environment variable `TRADEBOT_ENABLE_LIVE=true`

## Guardrails implemented

- Max order USD cap.
- Max total position USD cap.
- Max daily buy USD cap.
- Max daily short-open USD cap.
- Minimum USD reserve.
- Daily realized loss stop.
- Optional daily realized profit lock (`daily_profit_target_usd`).
- Max trades per day.
- Cooldown between trades.
- Product allowlist.
- Convert pair allowlist.
- Shorting disabled by default and limited by notional + leverage caps.

## Files

- `tradebot/` bot implementation.
- `config.example.json` safe default settings.
- `tradebot_state.json` runtime state (created automatically).
- `tradebot_metrics.json` analytics output for charting (created automatically).

## Setup

Requirements:

- Python 3.11+.
- Node.js 18+ (used only for ES256 JWT signing helper).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item config.example.json config.json
```

Save your Coinbase API key JSON in a file named `cdp_api_key.json` in the project root folder.

Optional GUI dependencies:

```powershell
pip install -r requirements-gui.txt
```

GUI loop updates use Streamlit fragments, so only live runtime sections rerun.

## Usage

Run one cycle (paper mode):

```powershell
python -m tradebot --config config.json run
```

Run continuously:

```powershell
python -m tradebot --config config.json run --loop
```

Launch optional GUI dashboard:

```powershell
python -m tradebot --config config.json gui
```

GUI options:

- `--host` (default `127.0.0.1`)
- `--port` (default `8501`)
- `--no-browser` to run headless

GUI notes:

- `Run One Cycle Now` in the GUI calls the same `run_cycle` logic as CLI `run`.
- GUI includes Start/Stop loop controls (websocket event-driven; no manual loop interval).
- When GUI loop is ON, a cycle is run when new websocket market data arrives.
- If websocket data is unavailable, GUI loop falls back to conservative polling.
- Backtest panel can include a synthetic scenario suite to test bull/bear/chop/volatility regimes.
- Live execution from GUI still requires:
  - `mode = "live"` in config
  - explicit GUI live checkbox + confirmation
  - `TRADEBOT_ENABLE_LIVE=true`
- GUI reads/writes the same runtime files:
  - `tradebot_state.json`
  - `tradebot_metrics.json`

Loop output behavior:

- In `--loop` mode, TradeBot suppresses duplicate console lines.
- A new line is printed only when cycle output changes (new signal/action-relevant information).

Auto actions in loop:

- Set `auto_actions.enable_auto_short`, `auto_actions.enable_auto_close_position`, and/or
  `auto_actions.enable_auto_convert` to `true` in config.
- Auto actions still obey live-execution safety gate and guardrails.
- Auto short-open and auto-convert also obey `guardrails.cooldown_seconds`.
- Auto checks are throttled by:
  - `auto_actions.short_check_interval_seconds`
  - `auto_actions.close_check_interval_seconds`
  - `auto_actions.convert_check_interval_seconds`

Real-time checks:

- Set `loop_seconds` to `1.0` for one decision cycle per second.
- Websocket market-data feed is enabled by default:
  - `websocket.enable: true`
  - `websocket.market_data_url: wss://advanced-trade-ws.coinbase.com`
  - `websocket.channel: ticker` (or `ticker_batch`)
  - `websocket.stale_seconds`: max age before REST fallback
  - `websocket.ping_interval_seconds` / `websocket.ping_timeout_seconds`
  - `websocket.max_reconnect_seconds`
- `price_refresh_seconds` controls cache/REST fallback behavior when websocket data is stale.
- Keep `granularity` at `ONE_MINUTE` (strategy candles are still 1-minute bars).
- Use:
  - `candles_refresh_seconds` to control how often candle history refreshes (example: `10`).
  - `fees_refresh_seconds` to control how often fee tier refreshes (example: `300`).
- Spot execution mode:
  - `execution.prefer_maker_orders: true` enables post-only maker orders for spot BUY/SELL.
  - `execution.maker_price_offset_bps` controls post-only limit offset.
  - `execution.maker_max_order_wait_seconds` controls stale-pending warning threshold.
  - `execution.max_spread_bps` blocks spot entries/exits when bid/ask spread is too wide.
  - In maker mode, bot output may remain `HOLD` while pending fills are being reconciled.
- Enable sub-minute signal overlay:
  - `strategy.enable_subminute_signals: true`
  - `strategy.subminute_window_seconds` (recent tick window, e.g. `90`)
  - `strategy.subminute_min_samples` (minimum ticks needed)
  - `strategy.subminute_min_signal_strength_bps` (micro-signal threshold)
- Regime adaptation (optional but enabled by default):
  - `strategy.enable_regime_adaptation`
  - `strategy.regime_volatility_lookback`
  - `strategy.low_volatility_bps` / `strategy.high_volatility_bps`
  - `strategy.trend_regime_threshold_bps`
  - `strategy.low_vol_threshold_multiplier`
  - `strategy.trend_threshold_multiplier`
  - `strategy.choppy_threshold_multiplier`
  - `strategy.high_vol_threshold_multiplier`

Show status:

```powershell
python -m tradebot --config config.json status
```

Convert (preview only unless live + `--execute-live`):

```powershell
python -m tradebot --config config.json convert --from-currency USD --to-currency USDC --amount 25
```

Short open (guarded, derivative products only):

```powershell
python -m tradebot --config config.json short-open --product-id BTC-PERP --base-size 0.001 --leverage 1
```

Close derivative position:

```powershell
python -m tradebot --config config.json close-position --product-id BTC-PERP
```

Run walk-forward backtest:

```powershell
python -m tradebot --config config.json backtest --lookback-candles 3000 --train-candles 600 --test-candles 180 --step-candles 90
```

Run backtest with synthetic scenario suite:

```powershell
python -m tradebot --config config.json backtest --lookback-candles 3000 --train-candles 600 --test-candles 180 --step-candles 90 --include-scenarios --scenario-length 1800
```

Run synthetic scenarios only (skip historical download):

```powershell
python -m tradebot --config config.json backtest --scenarios-only --scenario-length 1800 --train-candles 600 --test-candles 180 --step-candles 90
```

Scenario suite includes deterministic market regimes:

- `bull_trend_pullbacks`
- `bear_trend_bounces`
- `sideways_chop`
- `breakout_then_reversal`
- `volatility_spike_regime`

## Metrics JSON

`tradebot_metrics.json` has two chart-friendly arrays:

- `equity_curve`: one point per cycle (`ts`, `price_usd`, `realized_pnl_usd`, `unrealized_pnl_usd`, `net_profit_usd`).
- `trades`: one point per executed trade (`side`, `price_usd`, `base_size`, `quote_notional_usd`, `fee_usd`, `realized_pnl_usd`).

Guardrail note:

- Set `guardrails.max_trades_per_day` to `0` to disable trade-count limiting.
- Set `guardrails.daily_profit_target_usd` to a positive number to stop opening new
  buy/short/auto-convert risk once realized daily PnL reaches that target.

Runtime output note:

- Each `run` cycle now includes:
  - `price_source` (`websocket`, `cache`, `rest`, `cache_fallback`)
  - `signal_source` (`candle` or `subminute`)
  - `spread_bps`, `max_spread_bps`
  - `signal_threshold_bps`, `signal_volatility_bps`, `signal_regime`
  - `candle_signal`, `candle_signal_reason`
  - `candle_threshold_bps`, `candle_volatility_bps`, `candle_regime`
  - `subminute_signal`, `subminute_signal_reason`, `subminute_ticks` (when sub-minute is enabled)
  - `subminute_threshold_bps`, `subminute_volatility_bps`, `subminute_regime` (when sub-minute is enabled)
  - `pending_order_id`, `pending_order_status`, `pending_order_side` (when maker order is awaiting fill)
  - `daily_buy_used_usd`
  - `daily_buy_limit_usd`
  - `daily_buy_remaining_usd`
  - `daily_short_open_used_usd`
  - `daily_short_open_limit_usd`
  - `daily_short_open_remaining_usd`
  so you can monitor progress toward the daily spend cap in real time.
- Spot execution quality checks may return HOLD reasons:
  - `spread too wide for entry (...)`
  - `spread too wide for exit (...)`
- `status` now includes websocket health/connection details under `websocket`.
- Terminal output is colorized by default (`use_color_output: true` in config).
  Disable with `use_color_output: false` or environment variable `NO_COLOR=1`.
  Additional color cues:
  - signal/action fields and signal source (`candle` vs `subminute`)
  - trend/RSI/edge and fee fields
  - auto-action reason fields (`auto_close_reason`, `auto_short_reason`, `auto_convert_reason`)
  Daily spend colors:
  - `daily_buy_remaining_usd`: green when available, yellow when low, red at `0`.
  - `daily_buy_used_usd`: yellow as it accumulates, red when cap is reached.
  - `daily_short_open_remaining_usd`: green/yellow/red by remaining short-open budget.
  - `daily_short_open_used_usd`: yellow while opening shorts, red at cap.

## Going live safely

1. Keep `mode` as `paper` until logs look stable for multiple days.
2. Use very low limits first (`max_order_usd`, `max_position_usd`, `max_daily_loss_usd`).
3. Only then set:

```powershell
$env:TRADEBOT_ENABLE_LIVE="true"
```

and use `run --execute-live`.

## Disclaimer

This software is intended for educational use only and does not constitute financial advice. The author assumes no responsibility for any financial gains or losses resulting from its use.
