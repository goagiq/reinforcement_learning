"""
Signal Calculator

Maps RL agent actions and swarm agent recommendations to NinjaScript signals:
- Signal_Trend: -2 (downtrend strong), -1 (downtrend weak), 1 (uptrend weak), 2 (uptrend strong)
- Signal_Trade: -3 (downtrend strengthening), -2 (downtrend pullback), -1 (downtrend start),
                0 (no signal), 1 (uptrend start), 2 (uptrend pullback), 3 (uptrend strengthening)

Integrates:
- RL Agent (primary)
- Warren Buffett Contrarian Agent (fear/greed)
- Markov Regime Analyzer
- Elliott Wave Agent
- Other swarm agents
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class SignalState:
    """Tracks signal state for pullback detection"""
    previous_action: float = 0.0
    previous_trend: int = 0
    previous_trade: int = 0
    action_history: list = None
    trend_history: list = None
    
    def __post_init__(self):
        if self.action_history is None:
            self.action_history = []
        if self.trend_history is None:
            self.trend_history = []


class SignalCalculator:
    """
    Calculates Signal_Trend and Signal_Trade from RL + Swarm recommendations.
    """
    
    def __init__(
        self,
        action_change_threshold: float = 0.15,  # Significant change threshold
        pullback_detection_bars: int = 3,  # Bars to look back for pullback
        trend_strength_threshold: float = 0.5  # Threshold for strong vs weak trend
    ):
        """
        Initialize signal calculator.
        
        Args:
            action_change_threshold: Minimum action change to trigger signal update
            pullback_detection_bars: Number of bars to look back for pullback detection
            trend_strength_threshold: Action magnitude threshold for strong trend
        """
        self.action_change_threshold = action_change_threshold
        self.pullback_detection_bars = pullback_detection_bars
        self.trend_strength_threshold = trend_strength_threshold
        self.state = SignalState()
    
    def calculate_signals(
        self,
        rl_action: float,
        rl_confidence: float,
        swarm_recommendation: Optional[Dict] = None,
        current_position: float = 0.0,
        markov_regime: Optional[str] = None,
        markov_regime_confidence: float = 0.0
    ) -> Tuple[int, int]:
        """
        Calculate Signal_Trend and Signal_Trade.
        
        Args:
            rl_action: RL agent action (-1.0 to 1.0)
            rl_confidence: RL agent confidence (0.0 to 1.0)
            swarm_recommendation: Swarm recommendation dict (optional)
            current_position: Current position size (-1.0 to 1.0)
            markov_regime: Current Markov regime (e.g., "BULL", "BEAR", "NEUTRAL")
            markov_regime_confidence: Confidence in Markov regime (0.0 to 1.0)
        
        Returns:
            Tuple of (Signal_Trend, Signal_Trade)
        """
        # Extract swarm signals
        contrarian_signal = None
        contrarian_confidence = 0.0
        market_condition = "NEUTRAL"
        elliott_signal = None
        elliott_confidence = 0.0
        
        if swarm_recommendation:
            contrarian_signal = swarm_recommendation.get("contrarian_signal")
            contrarian_confidence = swarm_recommendation.get("contrarian_confidence", 0.0)
            market_condition = swarm_recommendation.get("market_condition", "NEUTRAL")
            elliott_signal = swarm_recommendation.get("elliott_wave_action")
            elliott_confidence = swarm_recommendation.get("elliott_wave_confidence", 0.0)
        
        # Calculate combined action (RL + Swarm weighted)
        # Use RL as primary, but adjust based on swarm signals
        combined_action = rl_action
        combined_confidence = rl_confidence
        
        # Adjust for Warren Buffett contrarian signal
        if contrarian_confidence >= 0.6:
            if contrarian_signal == "BUY" and market_condition == "FEARFUL":
                # Fearful market - contrarian says BUY (strong signal)
                combined_action = max(combined_action, 0.3)  # Boost upward
                combined_confidence = min(1.0, combined_confidence + contrarian_confidence * 0.2)
            elif contrarian_signal == "SELL" and market_condition == "GREEDY":
                # Greedy market - contrarian says SELL (strong signal)
                combined_action = min(combined_action, -0.3)  # Boost downward
                combined_confidence = min(1.0, combined_confidence + contrarian_confidence * 0.2)
        
        # Adjust for Elliott Wave signal
        if elliott_confidence >= 0.6:
            if elliott_signal == "BUY":
                combined_action = max(combined_action, 0.2)
                combined_confidence = min(1.0, combined_confidence + elliott_confidence * 0.15)
            elif elliott_signal == "SELL":
                combined_action = min(combined_action, -0.2)
                combined_confidence = min(1.0, combined_confidence + elliott_confidence * 0.15)
        
        # Adjust for Markov regime
        if markov_regime and markov_regime_confidence >= 0.6:
            if markov_regime in ["BULL", "TRENDING_UP"]:
                # Bull regime - favor long positions
                if combined_action > 0:
                    combined_confidence = min(1.0, combined_confidence + markov_regime_confidence * 0.1)
                else:
                    combined_action = max(combined_action, -0.1)  # Reduce short bias
            elif markov_regime in ["BEAR", "TRENDING_DOWN"]:
                # Bear regime - favor short positions
                if combined_action < 0:
                    combined_confidence = min(1.0, combined_confidence + markov_regime_confidence * 0.1)
                else:
                    combined_action = min(combined_action, 0.1)  # Reduce long bias
        
        # Calculate Signal_Trend based on action direction and magnitude
        signal_trend = self._calculate_trend_signal(combined_action, combined_confidence)
        
        # Calculate Signal_Trade based on action change, position state, and pullback detection
        signal_trade = self._calculate_trade_signal(
            combined_action,
            combined_confidence,
            current_position,
            signal_trend
        )
        
        # Update state
        self.state.previous_action = combined_action
        self.state.previous_trend = signal_trend
        self.state.previous_trade = signal_trade
        self.state.action_history.append(combined_action)
        self.state.trend_history.append(signal_trend)
        
        # Keep history limited
        if len(self.state.action_history) > self.pullback_detection_bars * 2:
            self.state.action_history.pop(0)
        if len(self.state.trend_history) > self.pullback_detection_bars * 2:
            self.state.trend_history.pop(0)
        
        return signal_trend, signal_trade
    
    def _calculate_trend_signal(self, action: float, confidence: float) -> int:
        """
        Calculate Signal_Trend:
        - 2 = uptrend strong
        - 1 = uptrend weak
        - -1 = downtrend weak
        - -2 = downtrend strong
        """
        if abs(action) < 0.05:  # No significant action
            return 0
        
        action_magnitude = abs(action)
        is_strong = action_magnitude >= self.trend_strength_threshold and confidence >= 0.7
        
        if action > 0:  # Uptrend
            return 2 if is_strong else 1
        else:  # Downtrend
            return -2 if is_strong else -1
    
    def _calculate_trade_signal(
        self,
        action: float,
        confidence: float,
        current_position: float,
        trend_signal: int
    ) -> int:
        """
        Calculate Signal_Trade:
        - 3 = uptrend strengthening
        - 2 = uptrend pullback
        - 1 = uptrend start
        - 0 = no signal
        - -1 = downtrend start
        - -2 = downtrend pullback
        - -3 = downtrend strengthening
        """
        # Check if action change is significant
        action_change = abs(action - self.state.previous_action)
        if action_change < self.action_change_threshold and abs(action) < 0.05:
            # No significant change - return previous signal or 0
            if abs(self.state.previous_trade) > 0:
                return self.state.previous_trade
            return 0
        
        # Determine if this is a pullback
        is_pullback = self._detect_pullback(action, trend_signal)
        
        # Determine if trend is strengthening
        is_strengthening = self._detect_strengthening(action, trend_signal)
        
        # Consider position state
        position_factor = self._get_position_factor(current_position, action)
        
        # Calculate signal based on trend direction
        if trend_signal > 0:  # Uptrend
            if is_strengthening:
                return 3  # Uptrend strengthening
            elif is_pullback:
                return 2  # Uptrend pullback
            elif position_factor > 0 or abs(action) >= 0.1:
                return 1  # Uptrend start
            else:
                return 0  # No signal
        elif trend_signal < 0:  # Downtrend
            if is_strengthening:
                return -3  # Downtrend strengthening
            elif is_pullback:
                return -2  # Downtrend pullback
            elif position_factor < 0 or abs(action) >= 0.1:
                return -1  # Downtrend start
            else:
                return 0  # No signal
        else:  # No trend
            return 0
    
    def _detect_pullback(self, current_action: float, current_trend: int) -> bool:
        """
        Detect if current action is a pullback (temporary reversal).
        
        Pullback = trend is positive but action temporarily goes negative (or vice versa),
        but recent history shows the trend direction.
        """
        if len(self.state.action_history) < 2:
            return False
        
        # Check if current action is opposite to trend
        if current_trend > 0 and current_action < -0.1:
            # Uptrend but action is negative - possible pullback
            # Check if recent history shows uptrend
            recent_actions = self.state.action_history[-self.pullback_detection_bars:]
            if len(recent_actions) >= 2:
                avg_recent = np.mean(recent_actions)
                if avg_recent > 0.1:  # Recent average is positive
                    return True
        elif current_trend < 0 and current_action > 0.1:
            # Downtrend but action is positive - possible pullback
            # Check if recent history shows downtrend
            recent_actions = self.state.action_history[-self.pullback_detection_bars:]
            if len(recent_actions) >= 2:
                avg_recent = np.mean(recent_actions)
                if avg_recent < -0.1:  # Recent average is negative
                    return True
        
        return False
    
    def _detect_strengthening(self, current_action: float, current_trend: int) -> bool:
        """
        Detect if trend is strengthening (action magnitude increasing).
        """
        if len(self.state.action_history) < 2:
            return False
        
        # Check if action magnitude is increasing in trend direction
        if current_trend > 0:  # Uptrend
            if current_action > self.state.previous_action and current_action > 0.3:
                return True
        elif current_trend < 0:  # Downtrend
            if current_action < self.state.previous_action and current_action < -0.3:
                return True
        
        return False
    
    def _get_position_factor(self, current_position: float, action: float) -> float:
        """
        Get position factor to consider current position state.
        
        Returns:
            Positive if should enter/expand position, negative if should exit/reduce
        """
        if abs(current_position) < 0.01:  # Flat
            # No position - factor based on action direction
            return 1.0 if action > 0.1 else (-1.0 if action < -0.1 else 0.0)
        elif current_position > 0:  # Long position
            # Long position - factor positive if action is positive (expand), negative if action is negative (exit)
            return 1.0 if action > 0.1 else (-1.0 if action < -0.1 else 0.0)
        else:  # Short position
            # Short position - factor negative if action is negative (expand), positive if action is positive (exit)
            return -1.0 if action < -0.1 else (1.0 if action > 0.1 else 0.0)
    
    def reset(self):
        """Reset signal state (useful for new trading session)."""
        self.state = SignalState()

