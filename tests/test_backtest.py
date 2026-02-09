import unittest

from tradebot.backtest import run_scenario_suite, walk_forward_backtest
from tradebot.config import GuardrailSettings, StrategySettings


class BacktestTests(unittest.TestCase):
    def test_walk_forward_returns_summary_and_windows(self) -> None:
        closes = [100.0 + (i * 0.05) for i in range(1200)]
        settings = StrategySettings(
            min_signal_strength_bps=5.0,
            min_expected_profit_usd=0.01,
            slippage_buffer_bps=2.0,
        )
        guardrails = GuardrailSettings(
            max_order_usd=25.0,
            min_order_usd=1.0,
            max_position_usd=100.0,
            min_usd_reserve=0.0,
            max_daily_buy_usd=500.0,
            max_daily_loss_usd=999.0,
            max_trades_per_day=0,
        )
        result = walk_forward_backtest(
            closes=closes,
            strategy=settings,
            guardrails=guardrails,
            initial_usd=1000.0,
            maker_fee_rate=0.006,
            taker_fee_rate=0.012,
            prefer_maker_orders=False,
            train_candles=300,
            test_candles=120,
            step_candles=60,
        )
        self.assertIn("summary", result)
        self.assertIn("windows", result)
        self.assertGreaterEqual(len(result["windows"]), 1)
        self.assertIn("net_profit_usd", result["summary"])
        self.assertIn("signal_buy_count", result["summary"])
        self.assertIn("buy_blocked_expected_profit_count", result["summary"])

    def test_walk_forward_rejects_small_series(self) -> None:
        closes = [100.0 + i for i in range(20)]
        with self.assertRaises(ValueError):
            walk_forward_backtest(
                closes=closes,
                strategy=StrategySettings(),
                guardrails=GuardrailSettings(),
                initial_usd=1000.0,
                maker_fee_rate=0.006,
                taker_fee_rate=0.012,
                prefer_maker_orders=False,
                train_candles=30,
                test_candles=10,
                step_candles=5,
            )

    def test_scenario_suite_returns_expected_shape(self) -> None:
        suite = run_scenario_suite(
            strategy=StrategySettings(
                min_signal_strength_bps=3.0,
                min_expected_profit_usd=0.0,
                slippage_buffer_bps=1.0,
            ),
            guardrails=GuardrailSettings(
                max_order_usd=50.0,
                min_order_usd=1.0,
                max_position_usd=50.0,
                min_usd_reserve=0.0,
                max_daily_buy_usd=1000.0,
                max_daily_loss_usd=999.0,
                max_trades_per_day=0,
            ),
            initial_usd=1000.0,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0,
            prefer_maker_orders=True,
            train_candles=300,
            test_candles=120,
            step_candles=60,
            scenario_length=1200,
        )
        self.assertIn("scenario_count", suite)
        self.assertIn("scenarios", suite)
        self.assertEqual(int(suite["scenario_count"]), 5)
        self.assertGreaterEqual(int(suite["with_trades_count"]), 1)
        scenarios = suite["scenarios"]
        self.assertIsInstance(scenarios, list)
        self.assertGreaterEqual(len(scenarios), 5)
        for row in scenarios:
            self.assertIn("scenario", row)
            self.assertIn("trades", row)
            self.assertIn("net_profit_usd", row)


if __name__ == "__main__":
    unittest.main()
