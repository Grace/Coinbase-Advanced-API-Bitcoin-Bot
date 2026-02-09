import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from tradebot.config import GuardrailSettings
from tradebot.risk import RiskContext, buy_checks, sell_checks, short_open_checks


def _ctx() -> RiskContext:
    return RiskContext(
        now=datetime.now(timezone.utc),
        usd_available=Decimal("500"),
        base_available=Decimal("1"),
        position_base=Decimal("0.2"),
        price=Decimal("100"),
    )


class RiskTests(unittest.TestCase):
    def test_buy_checks_blocks_over_max_order(self) -> None:
        guardrails = GuardrailSettings(max_order_usd=25.0)
        reasons = buy_checks(state={}, guardrails=guardrails, context=_ctx(), order_usd=Decimal("30"))
        self.assertIn("order above max_order_usd", reasons)

    def test_buy_checks_blocks_cooldown(self) -> None:
        guardrails = GuardrailSettings(cooldown_seconds=600)
        state = {
            "risk": {
                "last_trade_ts": (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
            }
        }
        reasons = buy_checks(
            state=state,
            guardrails=guardrails,
            context=_ctx(),
            order_usd=Decimal("20"),
        )
        self.assertIn("cooldown active", reasons)

    def test_buy_checks_blocks_daily_buy_notional_limit(self) -> None:
        guardrails = GuardrailSettings(max_daily_buy_usd=10.0)
        state = {"risk": {"daily_buy_notional_usd": {datetime.now(timezone.utc).date().isoformat(): "9"}}}
        reasons = buy_checks(
            state=state,
            guardrails=guardrails,
            context=_ctx(),
            order_usd=Decimal("2"),
        )
        self.assertIn("max_daily_buy_usd limit reached", reasons)

    def test_buy_checks_ignores_trade_limit_when_zero(self) -> None:
        guardrails = GuardrailSettings(max_trades_per_day=0)
        state = {"risk": {"daily_trades": {datetime.now(timezone.utc).date().isoformat(): 999}}}
        reasons = buy_checks(
            state=state,
            guardrails=guardrails,
            context=_ctx(),
            order_usd=Decimal("2"),
        )
        self.assertNotIn("max trades per day reached", reasons)

    def test_sell_checks_blocks_when_no_balance(self) -> None:
        guardrails = GuardrailSettings(min_order_usd=10.0)
        context = RiskContext(
            now=datetime.now(timezone.utc),
            usd_available=Decimal("100"),
            base_available=Decimal("0.01"),
            position_base=Decimal("0.01"),
            price=Decimal("100"),
        )
        reasons = sell_checks(
            state={},
            guardrails=guardrails,
            context=context,
            sell_base=Decimal("0.1"),
        )
        self.assertIn("insufficient base balance", reasons)

    def test_short_open_checks_blocks_daily_short_limit(self) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        state = {"risk": {"daily_short_open_notional_usd": {today: "9"}}}
        guardrails = GuardrailSettings(max_daily_short_open_usd=10.0, max_short_leverage=2.0)
        reasons = short_open_checks(
            state=state,
            guardrails=guardrails,
            notional_usd=Decimal("2"),
            leverage=Decimal("1"),
        )
        self.assertIn("max_daily_short_open_usd limit reached", reasons)

    def test_buy_checks_blocks_when_daily_profit_target_reached(self) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        state = {"risk": {"daily_realized_pnl_usd": {today: "5"}}}
        guardrails = GuardrailSettings(daily_profit_target_usd=5.0)
        reasons = buy_checks(
            state=state,
            guardrails=guardrails,
            context=_ctx(),
            order_usd=Decimal("2"),
        )
        self.assertIn("daily profit target reached", reasons)

    def test_short_open_checks_blocks_when_daily_profit_target_reached(self) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        state = {"risk": {"daily_realized_pnl_usd": {today: "2"}}}
        guardrails = GuardrailSettings(daily_profit_target_usd=2.0)
        reasons = short_open_checks(
            state=state,
            guardrails=guardrails,
            notional_usd=Decimal("1"),
            leverage=Decimal("1"),
        )
        self.assertIn("daily profit target reached", reasons)


if __name__ == "__main__":
    unittest.main()
