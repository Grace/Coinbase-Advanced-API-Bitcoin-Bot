import unittest
from datetime import datetime, timezone
from decimal import Decimal

from tradebot.coinbase_client import CoinbaseTickerWebSocketFeed


class WebsocketFeedTests(unittest.TestCase):
    def test_ingest_ticker_event_updates_latest_mid_price(self) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        feed = CoinbaseTickerWebSocketFeed(
            market_data_url="wss://advanced-trade-ws.coinbase.com",
            product_ids=["BTC-USD"],
            channel="ticker",
            stale_seconds=30.0,
            ping_interval_seconds=20,
            ping_timeout_seconds=10,
            max_reconnect_seconds=30,
            subscribe_heartbeats=True,
        )
        feed.ingest_message(
            {
                "channel": "ticker",
                "events": [
                    {
                        "type": "snapshot",
                        "tickers": [
                            {
                                "product_id": "BTC-USD",
                                "price": "70000.00",
                                "best_bid": "69999.00",
                                "best_ask": "70001.00",
                                "time": now_iso,
                            }
                        ],
                    }
                ],
            }
        )
        mid = feed.latest_mid_price("BTC-USD", max_age_seconds=60.0)
        self.assertEqual(mid, Decimal("70000.00"))

    def test_ingest_ticker_event_uses_last_price_when_bid_ask_missing(self) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        feed = CoinbaseTickerWebSocketFeed(
            market_data_url="wss://advanced-trade-ws.coinbase.com",
            product_ids=["ETH-USD"],
            channel="ticker",
            stale_seconds=30.0,
            ping_interval_seconds=20,
            ping_timeout_seconds=10,
            max_reconnect_seconds=30,
            subscribe_heartbeats=True,
        )
        feed.ingest_message(
            {
                "channel": "ticker",
                "events": [
                    {
                        "type": "update",
                        "tickers": [
                            {
                                "product_id": "ETH-USD",
                                "price": "3500.25",
                                "time": now_iso,
                            }
                        ],
                    }
                ],
            }
        )
        mid = feed.latest_mid_price("ETH-USD", max_age_seconds=60.0)
        self.assertEqual(mid, Decimal("3500.25"))


if __name__ == "__main__":
    unittest.main()
