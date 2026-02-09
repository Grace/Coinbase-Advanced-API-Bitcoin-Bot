import unittest
from typing import Any

from tradebot.coinbase_client import CoinbaseAdvancedClient


class _FakeCoinbaseClient(CoinbaseAdvancedClient):
    def __init__(self) -> None:
        # Intentionally skip parent init (no key/jwt needed for unit tests).
        self.calls: list[dict[str, Any]] = []

    def _request(  # type: ignore[override]
        self,
        method: str,
        path: str,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "method": method,
                "path": path,
                "query": dict(query or {}),
                "body": body,
            }
        )
        query = query or {}
        start = int(query.get("start", 0))
        end = int(query.get("end", start))
        granularity = str(query.get("granularity", "ONE_MINUTE"))
        seconds_per = {
            "ONE_MINUTE": 60,
            "FIVE_MINUTE": 300,
            "FIFTEEN_MINUTE": 900,
            "THIRTY_MINUTE": 1800,
            "ONE_HOUR": 3600,
            "TWO_HOUR": 7200,
            "SIX_HOUR": 21600,
            "ONE_DAY": 86400,
        }.get(granularity, 3600)

        candles: list[dict[str, Any]] = []
        ts = start
        while ts <= end:
            candles.append({"start": ts, "close": "100.0"})
            ts += seconds_per
        return {"candles": candles}


class CoinbaseClientCandleTests(unittest.TestCase):
    def test_recent_candles_chunks_large_lookback_and_caps_request_span(self) -> None:
        client = _FakeCoinbaseClient()
        lookback = 1500
        candles = client.recent_candles(
            product_id="BTC-USD",
            granularity="ONE_MINUTE",
            lookback=lookback,
        )
        self.assertGreater(len(client.calls), 1)
        self.assertEqual(len(candles), lookback)
        starts = [int(c["start"]) for c in candles]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(len(starts), len(set(starts)))
        max_window_seconds = 60 * 300
        for call in client.calls:
            query = call["query"]
            span = int(query["end"]) - int(query["start"])
            self.assertLessEqual(span, max_window_seconds)


if __name__ == "__main__":
    unittest.main()
