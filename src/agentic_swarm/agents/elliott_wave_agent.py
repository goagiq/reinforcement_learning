"""
Elliott Wave Agent

Real-time identification of impulsive Wave 3 and Wave 5 structures across
intraday timeframes. Produces deterministic trade signals and confidence
scores that feed the DecisionGate workflow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from copy import deepcopy

import pandas as pd

from src.agentic_swarm.shared_context import SharedContext
from src.analysis.elliott_wave import analyze_timeframe
from src.data_sources.market_data import MarketDataProvider


class ElliottWaveAgent:
    """
    Lightweight analytical agent (no LLM) that evaluates incoming price action
    for Elliott Wave impulse patterns.
    """

    def __init__(
        self,
        shared_context: SharedContext,
        market_data_provider: MarketDataProvider,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.shared_context = shared_context
        self.market_data_provider = market_data_provider
        self.config = deepcopy(config) if config else {}

        env_defaults = self.config.get("environment_defaults", {})
        self.instrument = self.config.get("instrument", env_defaults.get("instrument", "ES"))
        self.timeframes = sorted(
            set(self.config.get("timeframes", env_defaults.get("timeframes", [1, 5, 15])))
        )

        self.lookback_bars = self.config.get("lookback_bars", 400)
        self.swing_threshold = self.config.get("swing_threshold", 0.003)
        self.swing_thresholds = self.config.get("swing_thresholds", {})
        self.min_confidence = self.config.get("min_confidence", 0.55)
        self.position_multiplier = self.config.get("position_multiplier", 0.6)
        self.max_position_size = self.config.get("max_position_size", 0.8)
        self.min_bars = self.config.get("min_bars", 120)

        self.description = (
            "Detects Elliott Wave impulsive structures (Wave 3 / Wave 5) "
            "across multi-timeframe data to provide directional bias and confidence."
        )

    def analyze(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the latest market data for Elliott Wave signals.

        Args:
            market_state: Current market snapshot provided by the orchestrator.

        Returns:
            Dict with aggregated Elliott Wave recommendation.
        """
        timeframe_breakdown: Dict[int, Dict[str, Any]] = {}
        bullish_score = 0.0
        bearish_score = 0.0
        best_signal: Optional[Dict[str, Any]] = None

        for tf in self.timeframes:
            threshold = self._resolve_threshold(tf)
            df = self._load_timeframe_data(tf)
            if df is None or len(df) < self.min_bars:
                timeframe_breakdown[tf] = {
                    "status": "insufficient_data",
                    "confidence": 0.0,
                    "message": f"Only {0 if df is None else len(df)} bars available (min {self.min_bars})",
                }
                continue

            signal = analyze_timeframe(
                df=df,
                timeframe=tf,
                swing_threshold=threshold,
                min_bars=self.min_bars,
            )

            if signal is None:
                timeframe_breakdown[tf] = {
                    "status": "no_signal",
                    "confidence": 0.0,
                    "message": "No qualifying impulse structure detected",
                }
                continue

            timeframe_breakdown[tf] = {
                "status": "signal",
                "phase": signal.get("phase"),
                "direction": signal.get("direction"),
                "action": signal.get("action"),
                "confidence": signal.get("confidence", 0.0),
                "levels": signal.get("levels", {}),
                "timestamp": signal.get("timestamp"),
            }

            confidence = signal.get("confidence", 0.0)
            if signal.get("direction") == "bullish":
                bullish_score += confidence
            elif signal.get("direction") == "bearish":
                bearish_score += confidence

            if best_signal is None or confidence > best_signal.get("confidence", 0.0):
                best_signal = signal

        aggregated = self._aggregate_signals(
            timeframe_breakdown=timeframe_breakdown,
            bullish_score=bullish_score,
            bearish_score=bearish_score,
            best_signal=best_signal,
            market_state=market_state,
        )

        self.shared_context.set("elliott_wave_analysis", aggregated, "analysis_results")
        self.shared_context.add_agent_history(
            "elliott_wave",
            aggregated["action"],
            f"confidence={aggregated['confidence']:.2f}, bias={aggregated['bias']}",
        )

        return aggregated

    def _load_timeframe_data(self, timeframe: int) -> Optional[pd.DataFrame]:
        """Fetch historical bars for the configured instrument/timeframe."""
        df = self.market_data_provider.get_historical_data(
            instrument=self.instrument,
            timeframe=timeframe,
            lookback_bars=self.lookback_bars,
        )
        if df is None or df.empty:
            return None

        # Ensure timestamp column exists and is datetime
        if "timestamp" not in df:
            df = df.reset_index().rename(columns={"index": "timestamp"})
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def _resolve_threshold(self, timeframe: int) -> float:
        """Return swing threshold for a given timeframe."""
        key = str(timeframe)
        if key in self.swing_thresholds:
            return float(self.swing_thresholds[key])
        return float(self.swing_threshold)

    def _aggregate_signals(
        self,
        timeframe_breakdown: Dict[int, Dict[str, Any]],
        bullish_score: float,
        bearish_score: float,
        best_signal: Optional[Dict[str, Any]],
        market_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Combine per-timeframe analyses into a single recommendation."""
        action = "HOLD"
        confidence = 0.0
        position_size = 0.0
        phase = None
        bias = "neutral"

        if best_signal and best_signal.get("confidence", 0.0) >= self.min_confidence:
            confidence = float(best_signal["confidence"])
            action = best_signal.get("action", "HOLD")
            phase = best_signal.get("phase")
            if action == "BUY":
                bias = "bullish"
            elif action == "SELL":
                bias = "bearish"

        net_bias = bullish_score - bearish_score
        bias_strength = abs(net_bias)
        if bias_strength < 0.15:
            bias = "neutral"

        if action != "HOLD":
            position_size = min(
                self.max_position_size,
                max(0.0, confidence * self.position_multiplier),
            )

        aggregated = {
            "source": "elliott_wave",
            "instrument": self.instrument,
            "action": action,
            "confidence": round(confidence, 4),
            "position_size": round(position_size, 4),
            "phase": phase,
            "bias": bias,
            "bias_score": round(net_bias, 4),
            "timeframe_breakdown": timeframe_breakdown,
            "bullish_score": round(bullish_score, 4),
            "bearish_score": round(bearish_score, 4),
            "min_confidence": self.min_confidence,
            "timestamp": market_state.get("timestamp"),
        }

        return aggregated


