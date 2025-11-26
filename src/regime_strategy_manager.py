"""
Regime-Specific Strategy Manager

Manages different trading strategies for different market regimes.
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


@dataclass
class RegimeStrategy:
    """Strategy configuration for a specific regime"""
    regime: MarketRegime
    position_size_multiplier: float  # Multiplier for position sizing (0-1)
    entry_threshold: float  # Threshold for entry signals (higher = stricter)
    exit_threshold: float  # Threshold for exit signals
    stop_loss_multiplier: float  # Stop loss multiplier (1.0 = normal)
    take_profit_multiplier: float  # Take profit multiplier
    max_position_size: float  # Maximum position size for this regime
    min_confidence: float  # Minimum confidence required for trades


@dataclass
class RegimeTransition:
    """Regime transition event"""
    from_regime: MarketRegime
    to_regime: MarketRegime
    timestamp: datetime
    confidence: float
    transition_type: str  # "gradual", "sudden"


class RegimeStrategyManager:
    """
    Manages regime-specific strategies:
    - Different strategies per regime
    - Regime-specific position sizing
    - Regime transition detection
    - Strategy switching
    """
    
    def __init__(
        self,
        default_strategy: Optional[RegimeStrategy] = None
    ):
        """
        Initialize regime strategy manager.
        
        Args:
            default_strategy: Default strategy (used when regime is unknown)
        """
        self.strategies: Dict[MarketRegime, RegimeStrategy] = {}
        self.current_regime: MarketRegime = MarketRegime.UNKNOWN
        self.regime_history: List[MarketRegime] = []
        self.transitions: List[RegimeTransition] = []
        
        # Default strategy
        if default_strategy is None:
            default_strategy = RegimeStrategy(
                regime=MarketRegime.UNKNOWN,
                position_size_multiplier=1.0,
                entry_threshold=0.5,
                exit_threshold=0.3,
                stop_loss_multiplier=1.0,
                take_profit_multiplier=1.0,
                max_position_size=1.0,
                min_confidence=0.5
            )
        self.default_strategy = default_strategy
        
        # Initialize default strategies
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self):
        """Initialize default strategies for each regime"""
        # Trending Up Strategy
        self.strategies[MarketRegime.TRENDING_UP] = RegimeStrategy(
            regime=MarketRegime.TRENDING_UP,
            position_size_multiplier=1.2,  # Larger positions in trends
            entry_threshold=0.4,  # Lower threshold (easier entry)
            exit_threshold=0.2,  # Lower threshold (hold longer)
            stop_loss_multiplier=1.5,  # Wider stops in trends
            take_profit_multiplier=2.0,  # Larger targets
            max_position_size=1.0,
            min_confidence=0.5
        )
        
        # Trending Down Strategy
        self.strategies[MarketRegime.TRENDING_DOWN] = RegimeStrategy(
            regime=MarketRegime.TRENDING_DOWN,
            position_size_multiplier=0.8,  # Smaller positions in downtrends
            entry_threshold=0.6,  # Higher threshold (stricter entry)
            exit_threshold=0.4,  # Higher threshold (exit faster)
            stop_loss_multiplier=0.8,  # Tighter stops
            take_profit_multiplier=1.5,  # Smaller targets
            max_position_size=0.8,
            min_confidence=0.6
        )
        
        # Ranging Strategy
        self.strategies[MarketRegime.RANGING] = RegimeStrategy(
            regime=MarketRegime.RANGING,
            position_size_multiplier=0.6,  # Smaller positions in ranges
            entry_threshold=0.7,  # Higher threshold (mean reversion)
            exit_threshold=0.5,  # Higher threshold (quick exits)
            stop_loss_multiplier=0.6,  # Tighter stops
            take_profit_multiplier=1.0,  # Normal targets
            max_position_size=0.6,
            min_confidence=0.7
        )
        
        # High Volatility Strategy
        self.strategies[MarketRegime.HIGH_VOLATILITY] = RegimeStrategy(
            regime=MarketRegime.HIGH_VOLATILITY,
            position_size_multiplier=0.5,  # Much smaller positions
            entry_threshold=0.8,  # Very strict entry
            exit_threshold=0.6,  # Quick exits
            stop_loss_multiplier=1.2,  # Wider stops for volatility
            take_profit_multiplier=1.5,  # Larger targets
            max_position_size=0.5,
            min_confidence=0.8
        )
        
        # Low Volatility Strategy
        self.strategies[MarketRegime.LOW_VOLATILITY] = RegimeStrategy(
            regime=MarketRegime.LOW_VOLATILITY,
            position_size_multiplier=1.0,  # Normal positions
            entry_threshold=0.5,  # Normal threshold
            exit_threshold=0.3,  # Normal threshold
            stop_loss_multiplier=1.0,  # Normal stops
            take_profit_multiplier=1.0,  # Normal targets
            max_position_size=1.0,
            min_confidence=0.5
        )
    
    def detect_regime(
        self,
        price_data: np.ndarray,
        volume_data: np.ndarray,
        volatility: float,
        trend_strength: float = 0.0  # -1 to 1, negative = down, positive = up
    ) -> MarketRegime:
        """
        Detect current market regime from market data.
        
        Args:
            price_data: Recent price data
            volume_data: Recent volume data
            volatility: Current volatility
            trend_strength: Trend strength (-1 to 1)
        
        Returns:
            Detected market regime
        """
        # High volatility detection
        if volatility > 0.03:
            regime = MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.01:
            regime = MarketRegime.LOW_VOLATILITY
        else:
            # Trend detection
            if trend_strength > 0.3:
                regime = MarketRegime.TRENDING_UP
            elif trend_strength < -0.3:
                regime = MarketRegime.TRENDING_DOWN
            else:
                regime = MarketRegime.RANGING
        
        # Update regime history
        if regime != self.current_regime:
            # Regime transition detected
            transition = RegimeTransition(
                from_regime=self.current_regime,
                to_regime=regime,
                timestamp=datetime.now(),
                confidence=0.7,  # Default confidence
                transition_type="gradual"
            )
            self.transitions.append(transition)
            self.current_regime = regime
        
        self.regime_history.append(regime)
        if len(self.regime_history) > 100:
            self.regime_history.pop(0)
        
        return regime
    
    def get_strategy(self, regime: Optional[MarketRegime] = None) -> RegimeStrategy:
        """
        Get strategy for a specific regime.
        
        Args:
            regime: Regime (uses current if None)
        
        Returns:
            RegimeStrategy for the regime
        """
        if regime is None:
            regime = self.current_regime
        
        return self.strategies.get(regime, self.default_strategy)
    
    def adjust_position_size(
        self,
        base_position_size: float,
        regime: Optional[MarketRegime] = None
    ) -> float:
        """
        Adjust position size based on regime.
        
        Args:
            base_position_size: Base position size
            regime: Regime (uses current if None)
        
        Returns:
            Adjusted position size
        """
        strategy = self.get_strategy(regime)
        
        adjusted_size = base_position_size * strategy.position_size_multiplier
        
        # Apply max position size limit
        adjusted_size = min(adjusted_size, strategy.max_position_size)
        
        return adjusted_size
    
    def should_enter_trade(
        self,
        signal_strength: float,
        confidence: float,
        regime: Optional[MarketRegime] = None
    ) -> bool:
        """
        Check if trade should be entered based on regime strategy.
        
        Args:
            signal_strength: Signal strength (0-1)
            confidence: Trade confidence (0-1)
            regime: Regime (uses current if None)
        
        Returns:
            True if trade should be entered
        """
        strategy = self.get_strategy(regime)
        
        # Check thresholds
        if signal_strength < strategy.entry_threshold:
            return False
        
        if confidence < strategy.min_confidence:
            return False
        
        return True
    
    def should_exit_trade(
        self,
        signal_strength: float,
        regime: Optional[MarketRegime] = None
    ) -> bool:
        """
        Check if trade should be exited based on regime strategy.
        
        Args:
            signal_strength: Signal strength (0-1)
            regime: Regime (uses current if None)
        
        Returns:
            True if trade should be exited
        """
        strategy = self.get_strategy(regime)
        
        return signal_strength < strategy.exit_threshold
    
    def get_stop_loss_distance(
        self,
        base_stop_loss: float,
        regime: Optional[MarketRegime] = None
    ) -> float:
        """
        Get stop loss distance adjusted for regime.
        
        Args:
            base_stop_loss: Base stop loss distance
            regime: Regime (uses current if None)
        
        Returns:
            Adjusted stop loss distance
        """
        strategy = self.get_strategy(regime)
        return base_stop_loss * strategy.stop_loss_multiplier
    
    def get_take_profit_distance(
        self,
        base_take_profit: float,
        regime: Optional[MarketRegime] = None
    ) -> float:
        """
        Get take profit distance adjusted for regime.
        
        Args:
            base_take_profit: Base take profit distance
            regime: Regime (uses current if None)
        
        Returns:
            Adjusted take profit distance
        """
        strategy = self.get_strategy(regime)
        return base_take_profit * strategy.take_profit_multiplier
    
    def get_regime_summary(self) -> Dict:
        """
        Get summary of current regime and transitions.
        
        Returns:
            Dictionary with regime summary
        """
        return {
            "current_regime": self.current_regime.value,
            "regime_history": [r.value for r in self.regime_history[-10:]],  # Last 10
            "num_transitions": len(self.transitions),
            "recent_transitions": [
                {
                    "from": t.from_regime.value,
                    "to": t.to_regime.value,
                    "timestamp": t.timestamp.isoformat(),
                    "type": t.transition_type
                }
                for t in self.transitions[-5:]  # Last 5 transitions
            ],
            "current_strategy": {
                "position_size_multiplier": self.get_strategy().position_size_multiplier,
                "entry_threshold": self.get_strategy().entry_threshold,
                "exit_threshold": self.get_strategy().exit_threshold,
                "max_position_size": self.get_strategy().max_position_size
            }
        }

