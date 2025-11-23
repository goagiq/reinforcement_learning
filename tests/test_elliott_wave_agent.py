from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pandas as pd
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="strands.experimental.hooks",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agentic_swarm.shared_context import SharedContext
from src.agentic_swarm.agents.elliott_wave_agent import ElliottWaveAgent


class DummyMarketDataProvider:
    def __init__(self, data_map):
        self._data_map = data_map

    def get_historical_data(self, instrument, timeframe, lookback_bars):
        return self._data_map.get(timeframe)


def _make_price_dataframe(prices, start=None, freq_minutes=1):
    start = start or datetime(2024, 1, 1, 9, 30)
    rows = []
    for idx, price in enumerate(prices):
        ts = start + timedelta(minutes=freq_minutes * idx)
        rows.append(
            {
                "timestamp": ts,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1000 + idx,
            }
        )
    return pd.DataFrame(rows)


def test_elliott_wave_agent_detects_bullish_wave():
    prices = [
        100.0,
        103.0,
        105.5,
        99.5,
        108.0,
        115.0,
        110.0,
        119.0,
        123.0,
        117.5,
        126.5,
        128.0,
    ]
    df_1m = _make_price_dataframe(prices)
    provider = DummyMarketDataProvider({1: df_1m})
    shared_context = SharedContext(ttl_seconds=60)

    agent = ElliottWaveAgent(
        shared_context=shared_context,
        market_data_provider=provider,
        config={
            "instrument": "ES",
            "timeframes": [1],
            "lookback_bars": 200,
            "min_bars": 8,
            "min_confidence": 0.3,
            "swing_threshold": 0.006,
            "position_multiplier": 0.7,
            "max_position_size": 0.9,
        },
    )

    result = agent.analyze({"timestamp": datetime.now(timezone.utc).isoformat()})

    assert result["action"] == "BUY"
    assert result["confidence"] >= 0.4
    assert result["position_size"] > 0
    assert result["bias"] == "bullish"
    breakdown = result["timeframe_breakdown"][1]
    assert breakdown["status"] == "signal"
    assert breakdown["phase"] in {"wave3", "wave5_setup", "wave5_breakout"}


def test_elliott_wave_agent_returns_hold_when_no_signal():
    flat_prices = [100.0 + (idx * 0.05) for idx in range(20)]
    df = _make_price_dataframe(flat_prices)
    provider = DummyMarketDataProvider({1: df})
    shared_context = SharedContext(ttl_seconds=60)

    agent = ElliottWaveAgent(
        shared_context=shared_context,
        market_data_provider=provider,
        config={
            "instrument": "ES",
            "timeframes": [1],
            "lookback_bars": 200,
            "min_bars": 10,
            "min_confidence": 0.5,
            "swing_threshold": 0.02,
        },
    )

    result = agent.analyze({"timestamp": datetime.now(timezone.utc).isoformat()})

    assert result["action"] == "HOLD"
    assert result["confidence"] == 0
    assert result["position_size"] == 0
    breakdown = result["timeframe_breakdown"][1]
    assert breakdown["status"] in {"no_signal", "insufficient_data"}

