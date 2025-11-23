"""
Elliott Wave Analysis Utilities

Provides lightweight swing detection and impulse wave heuristics for real-time
Wave 3 / Wave 5 identification across intraday timeframes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Pivot:
    """Represents a swing high/low pivot."""

    index: int
    timestamp: pd.Timestamp
    price: float
    kind: str  # "high" or "low"


def detect_pivots(
    df: pd.DataFrame,
    swing_threshold: float = 0.003,
) -> List[Pivot]:
    """
    Detect swing highs/lows using a simple zig-zag style threshold.

    Args:
        df: Price dataframe with columns ["timestamp", "close"].
        swing_threshold: Minimum percent change required to mark a new pivot.

    Returns:
        List of Pivot objects alternating between highs and lows.
    """
    if df.empty or "close" not in df:
        return []

    closes = df["close"].values
    timestamps = pd.to_datetime(df["timestamp"]).values
    pivots: List[Pivot] = []

    # Seed with the first bar as an initial low pivot
    last_pivot_idx = 0
    last_pivot_price = closes[0]
    last_pivot_kind = "low"
    pivots.append(Pivot(0, pd.Timestamp(timestamps[0]), float(last_pivot_price), last_pivot_kind))

    for idx in range(1, len(closes)):
        price = closes[idx]
        change = (price - last_pivot_price) / last_pivot_price if last_pivot_price != 0 else 0.0

        if last_pivot_kind == "low":
            if price < last_pivot_price:
                last_pivot_price = price
                last_pivot_idx = idx
                pivots[-1] = Pivot(idx, pd.Timestamp(timestamps[idx]), float(price), "low")
            elif change >= swing_threshold:
                last_pivot_kind = "high"
                last_pivot_price = price
                last_pivot_idx = idx
                pivots.append(Pivot(idx, pd.Timestamp(timestamps[idx]), float(price), "high"))
        else:  # last pivot was high
            if price > last_pivot_price:
                last_pivot_price = price
                last_pivot_idx = idx
                pivots[-1] = Pivot(idx, pd.Timestamp(timestamps[idx]), float(price), "high")
            elif -change >= swing_threshold:
                last_pivot_kind = "low"
                last_pivot_price = price
                last_pivot_idx = idx
                pivots.append(Pivot(idx, pd.Timestamp(timestamps[idx]), float(price), "low"))

    # Ensure alternation starts with the dominant trend
    if len(pivots) >= 2 and pivots[0].kind == pivots[1].kind:
        pivots.pop(0)

    return pivots


def _find_impulse_window(pivots: List[Pivot]) -> Optional[Tuple[List[Pivot], str]]:
    """Return latest 5-pivot sequence forming a bullish or bearish impulse."""
    for start in range(len(pivots) - 5, -1, -1):
        window = pivots[start : start + 5]
        kinds = [p.kind for p in window]
        if kinds == ["low", "high", "low", "high", "low"]:
            return window, "bullish"
        if kinds == ["high", "low", "high", "low", "high"]:
            return window, "bearish"
    return None


def _find_wave3_window(pivots: List[Pivot]) -> Optional[Tuple[List[Pivot], str]]:
    """Return latest 4-pivot sequence indicative of a Wave 3 impulse."""
    for start in range(len(pivots) - 4, -1, -1):
        window = pivots[start : start + 4]
        kinds = [p.kind for p in window]
        if kinds == ["low", "high", "low", "high"]:
            return window, "bullish"
        if kinds == ["high", "low", "high", "low"]:
            return window, "bearish"
    return None


def _score_extension(length_ratio: float, target: float) -> float:
    """Score how well a wave length ratio meets a Fibonacci target."""
    if length_ratio <= 0:
        return 0.0
    diff = abs(length_ratio - target)
    if diff < 0.1:
        return 0.35
    if diff < 0.25:
        return 0.25
    if diff < 0.4:
        return 0.15
    return 0.0


def analyze_timeframe(
    df: pd.DataFrame,
    timeframe: int,
    swing_threshold: float = 0.003,
    min_bars: int = 120,
) -> Optional[Dict[str, any]]:
    """
    Analyze a timeframe for Elliott Wave impulses.

    Args:
        df: Price dataframe with OHLC and timestamp columns.
        timeframe: Timeframe in minutes (for metadata).
        swing_threshold: Zig-zag threshold as decimal (0.003 = 0.3%).
        min_bars: Minimum bars required.

    Returns:
        Dict with signal metadata or None if no impulse detected.
    """
    if len(df) < min_bars:
        return None

    pivots = detect_pivots(df, swing_threshold=swing_threshold)
    if len(pivots) < 4:
        return None

    current_price = float(df["close"].iloc[-1])
    current_time = pd.to_datetime(df["timestamp"].iloc[-1])

    impulse_window = _find_impulse_window(pivots)
    wave5_signal: Optional[Dict[str, any]] = None

    if impulse_window:
        window, direction = impulse_window
        prices = [p.price for p in window]

        if direction == "bullish":
            low0, high1, low2, high3, low4 = prices
            wave1 = high1 - low0
            wave2 = high1 - low2
            wave3 = high3 - low2
            wave4 = high3 - low4
            extension_ratio = wave3 / wave1 if wave1 != 0 else 0.0
            retrace2_ratio = wave2 / wave1 if wave1 != 0 else 0.0
            retrace4_ratio = wave4 / wave3 if wave3 != 0 else 0.0

            confidence = 0.0
            confidence += _score_extension(extension_ratio, 1.618)
            if 0.2 <= retrace2_ratio <= 0.62:
                confidence += 0.2
            if retrace4_ratio <= 0.5:
                confidence += 0.15
            if high3 > high1 and low4 > low2:
                confidence += 0.1

            phase = "wave5_setup"
            if current_price > high3:
                confidence += 0.15
                phase = "wave5_breakout"
            elif current_price > low4:
                confidence += 0.05

            confidence = min(1.0, max(0.0, confidence))

            wave5_signal = {
                "timeframe": timeframe,
                "direction": "bullish",
                "action": "BUY",
                "phase": phase,
                "confidence": confidence,
                "current_price": current_price,
                "timestamp": current_time.isoformat(),
                "levels": {
                    "wave1_high": high1,
                    "wave2_low": low2,
                    "wave3_high": high3,
                    "wave4_low": low4,
                },
                "metrics": {
                    "wave1": wave1,
                    "wave3": wave3,
                    "extension_ratio": extension_ratio,
                    "retrace2_ratio": retrace2_ratio,
                    "retrace4_ratio": retrace4_ratio,
                },
                "pivots": [
                    {
                        "timestamp": p.timestamp.isoformat(),
                        "price": p.price,
                        "kind": p.kind,
                    }
                    for p in window
                ],
            }

        else:  # bearish
            high0, low1, high2, low3, high4 = prices
            wave1 = high0 - low1
            wave2 = high2 - low1
            wave3 = high2 - low3
            wave4 = high4 - low3
            extension_ratio = wave3 / wave1 if wave1 != 0 else 0.0
            retrace2_ratio = wave2 / wave1 if wave1 != 0 else 0.0
            retrace4_ratio = wave4 / wave3 if wave3 != 0 else 0.0

            confidence = 0.0
            confidence += _score_extension(extension_ratio, 1.618)
            if 0.2 <= retrace2_ratio <= 0.62:
                confidence += 0.2
            if retrace4_ratio <= 0.5:
                confidence += 0.15
            if low3 < low1 and high4 < high2:
                confidence += 0.1

            phase = "wave5_setup"
            if current_price < low3:
                confidence += 0.15
                phase = "wave5_breakout"
            elif current_price < high4:
                confidence += 0.05

            confidence = min(1.0, max(0.0, confidence))

            wave5_signal = {
                "timeframe": timeframe,
                "direction": "bearish",
                "action": "SELL",
                "phase": phase,
                "confidence": confidence,
                "current_price": current_price,
                "timestamp": current_time.isoformat(),
                "levels": {
                    "wave1_low": low1,
                    "wave2_high": high2,
                    "wave3_low": low3,
                    "wave4_high": high4,
                },
                "metrics": {
                    "wave1": wave1,
                    "wave3": wave3,
                    "extension_ratio": extension_ratio,
                    "retrace2_ratio": retrace2_ratio,
                    "retrace4_ratio": retrace4_ratio,
                },
                "pivots": [
                    {
                        "timestamp": p.timestamp.isoformat(),
                        "price": p.price,
                        "kind": p.kind,
                    }
                    for p in window
                ],
            }

    wave3_window = _find_wave3_window(pivots)
    wave3_signal: Optional[Dict[str, any]] = None

    if wave3_window and (wave5_signal is None or wave5_signal["confidence"] < 0.65):
        window, direction = wave3_window
        prices = [p.price for p in window]

        if direction == "bullish":
            low0, high1, low2, high3 = prices
            wave1 = high1 - low0
            wave2 = high1 - low2
            wave3 = high3 - low2
            extension_ratio = wave3 / wave1 if wave1 != 0 else 0.0
            retrace2_ratio = wave2 / wave1 if wave1 != 0 else 0.0

            confidence = 0.0
            confidence += _score_extension(extension_ratio, 1.618)
            if 0.2 <= retrace2_ratio <= 0.62:
                confidence += 0.2
            if high3 > high1:
                confidence += 0.15
            if current_price > high1:
                confidence += 0.1

            confidence = min(1.0, max(0.0, confidence))

            wave3_signal = {
                "timeframe": timeframe,
                "direction": "bullish",
                "action": "BUY",
                "phase": "wave3",
                "confidence": confidence,
                "current_price": current_price,
                "timestamp": current_time.isoformat(),
                "levels": {
                    "wave1_high": high1,
                    "wave2_low": low2,
                    "wave3_high": high3,
                },
                "metrics": {
                    "wave1": wave1,
                    "wave3": wave3,
                    "extension_ratio": extension_ratio,
                    "retrace2_ratio": retrace2_ratio,
                },
                "pivots": [
                    {
                        "timestamp": p.timestamp.isoformat(),
                        "price": p.price,
                        "kind": p.kind,
                    }
                    for p in window
                ],
            }

        else:
            high0, low1, high2, low3 = prices
            wave1 = high0 - low1
            wave2 = high2 - low1
            wave3 = high2 - low3
            extension_ratio = wave3 / wave1 if wave1 != 0 else 0.0
            retrace2_ratio = wave2 / wave1 if wave1 != 0 else 0.0

            confidence = 0.0
            confidence += _score_extension(extension_ratio, 1.618)
            if 0.2 <= retrace2_ratio <= 0.62:
                confidence += 0.2
            if low3 < low1:
                confidence += 0.15
            if current_price < low1:
                confidence += 0.1

            confidence = min(1.0, max(0.0, confidence))

            wave3_signal = {
                "timeframe": timeframe,
                "direction": "bearish",
                "action": "SELL",
                "phase": "wave3",
                "confidence": confidence,
                "current_price": current_price,
                "timestamp": current_time.isoformat(),
                "levels": {
                    "wave1_low": low1,
                    "wave2_high": high2,
                    "wave3_low": low3,
                },
                "metrics": {
                    "wave1": wave1,
                    "wave3": wave3,
                    "extension_ratio": extension_ratio,
                    "retrace2_ratio": retrace2_ratio,
                },
                "pivots": [
                    {
                        "timestamp": p.timestamp.isoformat(),
                        "price": p.price,
                        "kind": p.kind,
                    }
                    for p in window
                ],
            }

    candidates = [signal for signal in [wave5_signal, wave3_signal] if signal]
    if not candidates:
        return None

    best_signal = max(candidates, key=lambda s: s["confidence"])
    best_signal["timeframe"] = timeframe
    best_signal["swing_threshold"] = swing_threshold
    return best_signal


