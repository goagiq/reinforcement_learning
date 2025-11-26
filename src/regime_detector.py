"""
Real-time Market Regime Detector

Simple, lightweight regime detection for use in trading environment.
Detects trending, ranging, and volatile regimes from recent price data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class RealTimeRegimeDetector:
    """
    Real-time regime detector for trading environment.
    
    Detects three regimes:
    - trending: Strong directional movement
    - ranging: Sideways movement
    - volatile: High volatility, choppy price action
    """
    
    def __init__(self, lookback_window: int = 50):
        """
        Initialize regime detector.
        
        Args:
            lookback_window: Number of bars to use for regime detection
        """
        self.lookback_window = lookback_window
        self.last_regime = "ranging"
        self.last_confidence = 0.5
        self.last_duration = 0
    
    def detect_regime(
        self,
        price_data: pd.DataFrame,
        current_step: int
    ) -> Dict[str, any]:
        """
        Detect current market regime from price data.
        
        Args:
            price_data: DataFrame with OHLCV data
            current_step: Current step index in data
        
        Returns:
            Dict with regime information:
            - regime: "trending", "ranging", or "volatile"
            - confidence: 0.0 to 1.0
            - duration: Normalized duration (0.0 to 1.0)
        """
        # Ensure we have enough data
        if current_step < self.lookback_window:
            return {
                "regime": "ranging",
                "confidence": 0.5,
                "duration": 0.0
            }
        
        # Get recent price data
        start_idx = max(0, current_step - self.lookback_window)
        end_idx = current_step + 1
        recent_data = price_data.iloc[start_idx:end_idx].copy()
        
        if len(recent_data) < 20:
            return {
                "regime": "ranging",
                "confidence": 0.5,
                "duration": 0.0
            }
        
        # Calculate features
        returns = recent_data["close"].pct_change().dropna()
        abs_returns = returns.abs()
        
        # Volatility (rolling std of returns)
        volatility = returns.rolling(window=min(20, len(returns))).std().iloc[-1]
        if pd.isna(volatility):
            volatility = abs_returns.mean()
        
        # Trend strength (ADX-like measure)
        # Calculate directional movement
        high_diff = recent_data["high"].diff()
        low_diff = -recent_data["low"].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = -low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Normalize by price
        avg_price = recent_data["close"].mean()
        plus_dm_norm = plus_dm.sum() / avg_price if avg_price > 0 else 0
        minus_dm_norm = minus_dm.sum() / avg_price if avg_price > 0 else 0
        
        # Trend strength (higher = more trending)
        trend_strength = abs(plus_dm_norm - minus_dm_norm) / (abs(plus_dm_norm) + abs(minus_dm_norm) + 1e-6)
        
        # Range detection (price oscillation)
        price_range = recent_data["high"].max() - recent_data["low"].min()
        price_range_pct = price_range / avg_price if avg_price > 0 else 0
        
        # Mean absolute return (for volatility)
        mean_abs_return = abs_returns.mean()
        
        # Regime classification
        # Normalize features
        volatility_norm = min(1.0, volatility * 100)  # Scale volatility
        trend_strength_norm = min(1.0, trend_strength * 2)  # Scale trend strength
        range_norm = min(1.0, price_range_pct * 10)  # Scale range
        
        # Classify regime
        if trend_strength_norm > 0.6 and volatility_norm < 0.5:
            # Strong trend, low volatility = trending
            regime = "trending"
            confidence = min(1.0, trend_strength_norm * 1.2)
        elif volatility_norm > 0.7 or mean_abs_return > 0.02:
            # High volatility = volatile
            regime = "volatile"
            confidence = min(1.0, volatility_norm * 1.2)
        elif range_norm < 0.3 and trend_strength_norm < 0.4:
            # Low range, low trend = ranging
            regime = "ranging"
            confidence = min(1.0, (1.0 - range_norm) * 1.2)
        else:
            # Mixed signals = default to ranging
            regime = "ranging"
            confidence = 0.5
        
        # Track regime duration (simplified)
        if regime == self.last_regime:
            self.last_duration += 1
        else:
            self.last_duration = 1
        
        # Normalize duration (0.0 to 1.0, assuming max duration of 100 steps)
        duration_norm = min(1.0, self.last_duration / 100.0)
        
        self.last_regime = regime
        self.last_confidence = confidence
        
        return {
            "regime": regime,
            "confidence": float(confidence),
            "duration": float(duration_norm)
        }
    
    def get_regime_features(self, price_data: pd.DataFrame, current_step: int) -> np.ndarray:
        """
        Get regime features as numpy array for RL state.
        
        Returns:
            Array of 5 features:
            - [0]: Trending indicator (1.0 if trending, else 0.0)
            - [1]: Ranging indicator (1.0 if ranging, else 0.0)
            - [2]: Volatile indicator (1.0 if volatile, else 0.0)
            - [3]: Confidence (0.0 to 1.0)
            - [4]: Duration (normalized, 0.0 to 1.0)
        """
        regime_info = self.detect_regime(price_data, current_step)
        
        # One-hot encoding for regime
        regime_one_hot = [0.0, 0.0, 0.0]
        if regime_info["regime"] == "trending":
            regime_one_hot[0] = 1.0
        elif regime_info["regime"] == "ranging":
            regime_one_hot[1] = 1.0
        elif regime_info["regime"] == "volatile":
            regime_one_hot[2] = 1.0
        
        return np.array([
            regime_one_hot[0],  # Trending
            regime_one_hot[1],  # Ranging
            regime_one_hot[2],  # Volatile
            regime_info["confidence"],  # Confidence
            regime_info["duration"]  # Duration
        ], dtype=np.float32)

