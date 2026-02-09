import unittest

from tradebot.config import StrategySettings
from tradebot.strategy import (
    ema,
    generate_short_signal,
    generate_signal,
    generate_subminute_short_signal,
    generate_subminute_signal,
    rsi,
)


class StrategyTests(unittest.TestCase):
    def test_ema_basic(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        out = ema(values, period=2)
        self.assertGreater(out, 3.0)

    def test_rsi_bounds(self) -> None:
        values = [100.0 + (i * 0.5) for i in range(40)]
        out = rsi(values, period=14)
        self.assertGreaterEqual(out, 0.0)
        self.assertLessEqual(out, 100.0)

    def test_generate_signal_buy_when_uptrend(self) -> None:
        closes = [100.0 + (i * 0.4) for i in range(150)]
        settings = StrategySettings(min_signal_strength_bps=5.0)
        signal = generate_signal(
            closes=closes,
            roundtrip_fee_bps=5.0,
            settings=settings,
            has_open_long=False,
        )
        self.assertIn(signal.action, {"BUY", "HOLD"})
        self.assertGreaterEqual(signal.expected_edge_bps, 0.0)

    def test_generate_signal_sell_when_bearish_with_position(self) -> None:
        closes = [200.0 - (i * 0.6) for i in range(150)]
        settings = StrategySettings(min_signal_strength_bps=5.0)
        signal = generate_signal(
            closes=closes,
            roundtrip_fee_bps=5.0,
            settings=settings,
            has_open_long=True,
        )
        self.assertIn(signal.action, {"SELL", "HOLD"})

    def test_generate_short_signal_short_when_downtrend(self) -> None:
        closes = [200.0 - (i * 0.5) for i in range(150)]
        settings = StrategySettings(min_signal_strength_bps=5.0)
        signal = generate_short_signal(
            closes=closes,
            roundtrip_fee_bps=5.0,
            settings=settings,
            has_open_short=False,
        )
        self.assertIn(signal.action, {"SHORT", "HOLD"})

    def test_generate_subminute_signal_shapes(self) -> None:
        prices = [100.0 + (i * 0.01) + (0.03 if i % 4 == 0 else -0.01) for i in range(180)]
        settings = StrategySettings(
            subminute_min_samples=20,
            subminute_min_signal_strength_bps=1.0,
        )
        signal = generate_subminute_signal(
            prices=prices,
            roundtrip_fee_bps=2.0,
            settings=settings,
            has_open_long=False,
        )
        self.assertIn(signal.action, {"BUY", "HOLD"})
        self.assertGreaterEqual(signal.expected_edge_bps, 0.0)

    def test_generate_subminute_short_signal_shapes(self) -> None:
        prices = [200.0 - (i * 0.012) + (0.03 if i % 5 == 0 else -0.01) for i in range(180)]
        settings = StrategySettings(
            subminute_min_samples=20,
            subminute_min_signal_strength_bps=1.0,
        )
        signal = generate_subminute_short_signal(
            prices=prices,
            roundtrip_fee_bps=2.0,
            settings=settings,
            has_open_short=False,
        )
        self.assertIn(signal.action, {"SHORT", "HOLD"})
        self.assertGreaterEqual(signal.expected_edge_bps, 0.0)


if __name__ == "__main__":
    unittest.main()
