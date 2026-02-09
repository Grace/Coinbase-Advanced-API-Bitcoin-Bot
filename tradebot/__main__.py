from __future__ import annotations

import argparse
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

from .bot import TradeBot
from .config import load_config


def _decimal(value: str) -> Decimal:
    return Decimal(value)


def _default_config_path() -> str | None:
    candidate = Path("config.json")
    return str(candidate) if candidate.exists() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safety-first Coinbase Advanced Trade bot")
    parser.add_argument("--config", default=_default_config_path(), help="Path to JSON config")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run bot cycle (or loop)")
    run_parser.add_argument("--loop", action="store_true", help="Run continuously")
    run_parser.add_argument(
        "--execute-live",
        action="store_true",
        help="Allow live order placement (requires mode=live + TRADEBOT_ENABLE_LIVE=true)",
    )

    convert_parser = subparsers.add_parser("convert", help="Create/commit convert quote")
    convert_parser.add_argument("--from-currency", required=True)
    convert_parser.add_argument("--to-currency", required=True)
    convert_parser.add_argument("--amount", required=True, type=_decimal)
    convert_parser.add_argument("--execute-live", action="store_true")

    short_parser = subparsers.add_parser("short-open", help="Open a guarded short position")
    short_parser.add_argument("--product-id", required=True)
    short_parser.add_argument("--base-size", required=True, type=_decimal)
    short_parser.add_argument("--leverage", default=Decimal("1"), type=_decimal)
    short_parser.add_argument("--margin-type", default="CROSS")
    short_parser.add_argument("--execute-live", action="store_true")

    close_parser = subparsers.add_parser("close-position", help="Close derivative position")
    close_parser.add_argument("--product-id", required=True)
    close_parser.add_argument("--size", type=_decimal)
    close_parser.add_argument("--execute-live", action="store_true")

    backtest_parser = subparsers.add_parser("backtest", help="Run walk-forward backtest")
    backtest_parser.add_argument("--lookback-candles", type=int, default=3000)
    backtest_parser.add_argument("--train-candles", type=int, default=600)
    backtest_parser.add_argument("--test-candles", type=int, default=180)
    backtest_parser.add_argument("--step-candles", type=int, default=90)
    backtest_parser.add_argument(
        "--include-scenarios",
        action="store_true",
        help="Also run synthetic scenario suite (bull/bear/chop/volatility regimes)",
    )
    backtest_parser.add_argument(
        "--scenario-length",
        type=int,
        default=1800,
        help="Candles per synthetic scenario when --include-scenarios is enabled",
    )
    backtest_parser.add_argument(
        "--scenarios-only",
        action="store_true",
        help="Run only synthetic scenarios (skip historical candle download)",
    )

    gui_parser = subparsers.add_parser("gui", help="Launch optional GUI dashboard")
    gui_parser.add_argument("--host", default="127.0.0.1")
    gui_parser.add_argument("--port", type=int, default=8501)
    gui_parser.add_argument("--no-browser", action="store_true")

    subparsers.add_parser("status", help="Show balances, fee tier, and tracked position")
    return parser


def _launch_gui(config_path: str | None, host: str, port: int, no_browser: bool) -> None:
    gui_script = Path(__file__).with_name("gui_app.py")
    effective_config = config_path or _default_config_path() or "config.json"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(gui_script),
        "--server.address",
        str(host),
        "--server.port",
        str(port),
        "--server.headless",
        "true" if no_browser else "false",
        "--",
        "--config",
        str(effective_config),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to launch GUI. Install optional dependencies with "
            "'pip install -r requirements-gui.txt'"
        ) from exc


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "gui":
            _launch_gui(
                config_path=args.config,
                host=args.host,
                port=int(args.port),
                no_browser=bool(args.no_browser),
            )
            return

        config = load_config(args.config)
        bot = TradeBot(config)

        if args.command == "run":
            if args.loop:
                bot.run_loop(execute_live=args.execute_live)
                return
            result = bot.run_cycle(execute_live=args.execute_live)
            print(bot.format_cycle_result(result))
            return

        if args.command == "convert":
            output = bot.convert(
                from_currency=args.from_currency,
                to_currency=args.to_currency,
                amount=args.amount,
                execute_live=args.execute_live,
            )
            print(json.dumps(output, indent=2))
            return

        if args.command == "short-open":
            output = bot.open_short(
                product_id=args.product_id,
                base_size=args.base_size,
                leverage=args.leverage,
                execute_live=args.execute_live,
                margin_type=args.margin_type,
            )
            print(json.dumps(output, indent=2))
            return

        if args.command == "close-position":
            output = bot.close_position(
                product_id=args.product_id,
                size=args.size,
                execute_live=args.execute_live,
            )
            print(json.dumps(output, indent=2))
            return

        if args.command == "status":
            print(json.dumps(bot.status(), indent=2))
            return

        if args.command == "backtest":
            output = bot.backtest_walk_forward(
                lookback_candles=args.lookback_candles,
                train_candles=args.train_candles,
                test_candles=args.test_candles,
                step_candles=args.step_candles,
                include_scenarios=bool(args.include_scenarios or args.scenarios_only),
                scenario_length=int(args.scenario_length),
                scenarios_only=bool(args.scenarios_only),
            )
            print(json.dumps(output, indent=2))
            return

        raise RuntimeError(f"Unsupported command: {args.command}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
