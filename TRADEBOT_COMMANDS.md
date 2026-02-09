# TradeBot Commands

Command line entrypoint:

```powershell
python -m tradebot --config <path-to-config.json> <command> [options]
```

If `--config` is omitted and `./config.json` exists, TradeBot uses `./config.json`.

## Live Execution Safety Gate

Live orders are only submitted when all conditions are true:

1. `mode` in config is `"live"`.
2. Command includes `--execute-live` (where applicable).
3. Environment variable is set:

```powershell
$env:TRADEBOT_ENABLE_LIVE="true"
```

Without these, TradeBot performs preview/simulation behavior and returns `HOLD` or preview outputs.

## Commands

## `run`

Run one analysis/execution cycle or loop continuously.

Syntax:

```powershell
python -m tradebot --config config.json run [--loop] [--execute-live]
```

Options:

- `--loop`: run forever, sleeping `loop_seconds` between cycles.
- `--execute-live`: allow live order placement when safety gate is fully satisfied.
- In `--loop` mode, duplicate output lines are suppressed; a line prints only when output changes.

Loop timing controls in config:

- `loop_seconds`: decision cadence (set `1.0` for roughly once per second).
- `websocket.enable`: enable Coinbase market-data websocket feed.
- `websocket.market_data_url`: websocket URL (default `wss://advanced-trade-ws.coinbase.com`).
- `websocket.channel`: `ticker` or `ticker_batch`.
- `websocket.stale_seconds`: max websocket tick age before REST fallback.
- `websocket.ping_interval_seconds` / `websocket.ping_timeout_seconds`: connection keepalive tuning.
- `websocket.max_reconnect_seconds`: reconnect backoff cap.
- `price_refresh_seconds`: cache/REST fallback refresh interval.
- `candles_refresh_seconds`: candle API refresh interval while looping.
- `fees_refresh_seconds`: fee-tier refresh interval while looping.
- `execution.prefer_maker_orders`: submit post-only limit orders for spot buy/sell decisions.
- `execution.maker_price_offset_bps`: price offset used when submitting maker post-only orders.
- `execution.maker_max_order_wait_seconds`: warning threshold for pending maker orders.
- `execution.max_spread_bps`: block spot entries/exits when spread is wider than allowed.
- `strategy.enable_subminute_signals`: enable sub-minute tick-based signal overlay.
- `strategy.subminute_window_seconds`: sub-minute history window kept in memory.
- `strategy.subminute_min_samples`: minimum tick samples required for sub-minute signal.
- `strategy.subminute_min_signal_strength_bps`: sub-minute strength threshold.
- `strategy.enable_regime_adaptation`: enable volatility/trend regime threshold adaptation.
- `strategy.regime_volatility_lookback`: candles/ticks used for volatility regime detection.
- `strategy.low_volatility_bps` / `strategy.high_volatility_bps`: regime boundaries.
- `strategy.trend_regime_threshold_bps`: trend magnitude threshold for trending regime.
- `strategy.low_vol_threshold_multiplier`: threshold multiplier in low-vol regimes.
- `strategy.trend_threshold_multiplier`: threshold multiplier in trending regimes.
- `strategy.choppy_threshold_multiplier`: threshold multiplier in choppy regimes.
- `strategy.high_vol_threshold_multiplier`: threshold multiplier in high-vol regimes.
- `use_color_output`: colorize console output (`true` by default).
  You can also disable colors with `NO_COLOR=1`.
  Spend color cues:
  - `daily_buy_remaining_usd`: green/yellow/red by remaining budget.
  - `daily_buy_used_usd`: yellow while spending, red at cap.

Optional loop auto-actions via config:

- `auto_actions.enable_auto_short`
- `auto_actions.enable_auto_close_position`
- `auto_actions.enable_auto_convert`
- `auto_actions.short_check_interval_seconds`
- `auto_actions.close_check_interval_seconds`
- `auto_actions.convert_check_interval_seconds`

These run inside `run --loop` when their profit/guardrail checks pass.
Auto short-open and auto-convert are also subject to `guardrails.cooldown_seconds`.
When maker mode is enabled, spot actions may show `HOLD` while pending order fills reconcile.

Examples:

```powershell
python -m tradebot --config config.json run
python -m tradebot --config config.json run --loop
python -m tradebot --config config.json run --loop --execute-live
```

## `backtest`

Run walk-forward backtesting over recent candles.

Syntax:

```powershell
python -m tradebot --config config.json backtest [--lookback-candles N] [--train-candles N] [--test-candles N] [--step-candles N] [--include-scenarios] [--scenario-length N] [--scenarios-only]
```

Options:

- `--lookback-candles`: total candles downloaded for simulation (default `3000`).
- `--train-candles`: candles used as lookback context before each test window (default `600`).
- `--test-candles`: candles traded in each window (default `180`).
- `--step-candles`: slide size between windows (default `90`).
- `--include-scenarios`: also run deterministic synthetic market scenarios.
- `--scenario-length`: candles per synthetic scenario (default `1800`).
- `--scenarios-only`: skip historical replay and run synthetic scenarios only.

Output:

- JSON with `summary`, per-window metrics (`windows`), and run metadata (`meta`).
- With `--include-scenarios`, output also includes `scenario_suite` with scenario-level trade and PnL stats.
- With `--scenarios-only`, `summary`/`windows` are empty for historical replay and `meta.scenarios_only=true`.

## `gui`

Launch the optional Streamlit GUI dashboard.

Install GUI dependencies first:

```powershell
pip install -r requirements-gui.txt
```

GUI loop updates use Streamlit fragments, so only live runtime sections rerun.

Syntax:

```powershell
python -m tradebot --config config.json gui [--host 127.0.0.1] [--port 8501] [--no-browser]
```

Options:

- `--host`: bind address (default `127.0.0.1`).
- `--port`: Streamlit port (default `8501`).
- `--no-browser`: start headless.

GUI behavior:

- `Run One Cycle Now` executes one bot cycle using the current config.
- GUI has Start/Stop loop controls with websocket event-driven execution.
- In GUI loop mode, cycles run when new websocket ticks arrive.
- If websocket data is unavailable, GUI loop falls back to conservative polling.
- API status refresh is manual from the dashboard to reduce unnecessary API calls.
- Backtest panel can run synthetic scenario-suite simulations and chart scenario-level outcomes.
- Live order safety gate is still enforced (`mode=live`, explicit live intent, env var enabled).

## `status`

Show current bot/account status snapshot (JSON).

Syntax:

```powershell
python -m tradebot --config config.json status
```

Output includes:

- mode, product, price
- price source (`websocket`, `cache`, `rest`, or `cache_fallback`)
- maker/taker fee rates
- balances
- tracked position
- pending spot maker order (if any)
- websocket status (`connected`, `running`, `last_error`, latest tick snapshot)
- performance summary (realized/unrealized/net/fees)
- daily buy guardrail snapshot (`daily_buy_used_usd`, `daily_buy_limit_usd`, `daily_buy_remaining_usd`)
- daily short-open guardrail snapshot (`daily_short_open_used_usd`, `daily_short_open_limit_usd`, `daily_short_open_remaining_usd`)

## `run` output fields

The printed line includes strategy context, performance fields, and daily spend tracking.
Signal-debug fields:

- `price_source`: where current price came from (`websocket`, `cache`, `rest`, `cache_fallback`).
- `signal_source`: `candle` or `subminute`.
- `spread_bps`, `max_spread_bps`: current spread estimate and execution-quality ceiling.
- `signal_threshold_bps`, `signal_volatility_bps`, `signal_regime`: final threshold and regime diagnostics.
- `candle_signal`, `candle_signal_reason`: baseline 1-minute candle signal.
- `candle_threshold_bps`, `candle_volatility_bps`, `candle_regime`: candle signal diagnostics.
- `subminute_signal`, `subminute_signal_reason`, `subminute_ticks`: micro-signal overlay details (when enabled).
- `subminute_threshold_bps`, `subminute_volatility_bps`, `subminute_regime`: sub-minute diagnostics.
- `auto_close_reason`, `auto_short_reason`, `auto_convert_reason`: why each auto action did or did not execute during HOLD cycles.
- Spread execution-quality blockers:
  - `spread too wide for entry (...)`
  - `spread too wide for exit (...)`

Useful spend fields:

- `daily_buy_used_usd`: total buy notional already used today (UTC day).
- `daily_buy_limit_usd`: configured daily buy cap (`guardrails.max_daily_buy_usd`).
- `daily_buy_remaining_usd`: remaining buy notional available today.
- `daily_short_open_used_usd`: total short-open notional already used today (UTC day).
- `daily_short_open_limit_usd`: configured short-open cap (`guardrails.max_daily_short_open_usd`).
- `daily_short_open_remaining_usd`: remaining short-open notional available today.

Related daily guardrail:

- `guardrails.daily_profit_target_usd` (config): when set > 0, new buy/short/auto-convert
  risk-taking is blocked after that day's realized PnL reaches the target.

## `convert`

Create convert quote and optionally commit it.

Syntax:

```powershell
python -m tradebot --config config.json convert --from-currency <CUR1> --to-currency <CUR2> --amount <AMOUNT> [--execute-live]
```

Required options:

- `--from-currency`: source currency code, e.g. `USD`.
- `--to-currency`: destination currency code, e.g. `USDC`.
- `--amount`: numeric amount to convert.

Optional:

- `--execute-live`: commit convert trade if allowed by safety gate.

Example:

```powershell
python -m tradebot --config config.json convert --from-currency USD --to-currency USDC --amount 25
```

## `short-open`

Open a guarded short position (derivatives only).

Syntax:

```powershell
python -m tradebot --config config.json short-open --product-id <PRODUCT> --base-size <SIZE> [--leverage <LEV>] [--margin-type <TYPE>] [--execute-live]
```

Required options:

- `--product-id`: derivative product, e.g. `BTC-PERP`.
- `--base-size`: base asset size.

Optional:

- `--leverage`: default `1`.
- `--margin-type`: default `CROSS`.
- `--execute-live`: submit live short if safety gate and guardrails allow it.

Example:

```powershell
python -m tradebot --config config.json short-open --product-id BTC-PERP --base-size 0.001 --leverage 1
```

## `close-position`

Close an open derivative position.

Syntax:

```powershell
python -m tradebot --config config.json close-position --product-id <PRODUCT> [--size <SIZE>] [--execute-live]
```

Required options:

- `--product-id`: derivative product, e.g. `BTC-PERP`.

Optional:

- `--size`: partial size to close. Omit to close all.
- `--execute-live`: submit live close if safety gate allows it.

Example:

```powershell
python -m tradebot --config config.json close-position --product-id BTC-PERP
```

## Useful Helpers

Show built-in help:

```powershell
python -m tradebot --help
python -m tradebot run --help
python -m tradebot convert --help
python -m tradebot short-open --help
python -m tradebot close-position --help
python -m tradebot backtest --help
python -m tradebot gui --help
python -m tradebot status --help
```

## Related Files

- Runtime state: `tradebot_state.json`
- Analytics for graphing: `tradebot_metrics.json`
- Main settings: `config.json`
