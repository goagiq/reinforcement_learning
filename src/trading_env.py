"""
Trading Environment for Reinforcement Learning

Gymnasium-compatible trading environment with multi-timeframe support.
Uses continuous action space for position sizing.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.utils.colors import error, warn, info

# Import execution quality modules
try:
    from src.slippage_model import SlippageModel
    from src.execution_quality import ExecutionQualityTracker
    from src.market_impact import MarketImpactModel
    EXECUTION_QUALITY_AVAILABLE = True
except ImportError:
    EXECUTION_QUALITY_AVAILABLE = False
    SlippageModel = None
    ExecutionQualityTracker = None
    MarketImpactModel = None


@dataclass
class TradeState:
    """Current trading state"""
    position: float  # Current position (-1.0 to 1.0)
    entry_price: Optional[float]
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    trades_count: int
    winning_trades: int
    losing_trades: int
    consecutive_losses: int = 0  # Track consecutive losses for loss limit
    trading_paused: bool = False  # Track if trading is paused due to consecutive losses
    
    # Track average win/loss for risk/reward monitoring
    total_win_pnl: float = 0.0  # Sum of all winning trade PnLs
    total_loss_pnl: float = 0.0  # Sum of all losing trade PnLs (absolute values)
    
    # Trailing stop tracking (NEW - ATR-adaptive)
    highest_price: Optional[float] = None  # Highest price since entry (for long positions)
    lowest_price: Optional[float] = None  # Lowest price since entry (for short positions)
    trailing_stop_price: Optional[float] = None  # Current trailing stop price
    
    # Take profit target tracking (NEW - ATR-adaptive)
    take_profit_target_price: Optional[float] = None  # ATR-based take profit target
    take_profit_hit: bool = False  # Track if target was hit (for reward bonus)


class TradingEnvironment(gym.Env):
    """
    Trading environment for RL.
    
    State space: Multi-timeframe market features
    Action space: Continuous position size [-1.0, 1.0]
    Reward: Risk-adjusted returns
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 1}
    
    def __init__(
        self,
        data: Dict[int, pd.DataFrame],  # Multi-timeframe data
        timeframes: List[int] = [1, 5, 15],
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.0003,  # Increased from 0.0001 to 0.0003 (0.03%) for realistic costs
        lookback_bars: int = 20,
        reward_config: Optional[Dict] = None,
        max_episode_steps: Optional[int] = None,  # Optional limit on episode length
        action_threshold: float = 0.05  # Configurable action threshold (default 5%)
    ):
        super().__init__()
        
        self.data = data
        self.timeframes = sorted(timeframes)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.commission_rate = transaction_cost  # Commission rate (0.03% = 0.0003)
        self.lookback_bars = lookback_bars
        self.action_threshold = action_threshold  # Minimum position change to trigger trade (default 5%)
        
        # Reward configuration with profitability-focused defaults
        default_reward_config = {
            "pnl_weight": 1.0,
            "risk_penalty": 0.5,
            "drawdown_penalty": 0.3,
            "exploration_bonus_enabled": True,  # Can be disabled
            "exploration_bonus_scale": 0.00001,  # Reduced from 0.0001 (10x reduction)
            "loss_mitigation": 0.05,  # Reduced from 0.3 to 0.05 (5% mitigation)
            "overtrading_penalty_enabled": True,
            "optimal_trades_per_episode": 50,  # Target trades per episode
            "profit_factor_required": 1.0  # Minimum profit factor to reward
        }
        
        if reward_config:
            default_reward_config.update(reward_config)
        self.reward_config = default_reward_config
        
        # Validate data alignment
        self._validate_data()
        
        # Regime detector (optional, for regime-aware RL)
        self.regime_detector = None
        self.include_regime_features = reward_config.get("include_regime_features", False) if reward_config else False
        
        if self.include_regime_features:
            try:
                from src.regime_detector import RealTimeRegimeDetector
                self.regime_detector = RealTimeRegimeDetector(lookback_window=50)
                print("[OK] Regime detector initialized")
            except Exception as e:
                print(f"[WARN] Could not initialize regime detector: {e}")
                self.regime_detector = None
                self.include_regime_features = False
        
        # Forecast predictor (optional, for forecast-enhanced RL)
        self.forecast_predictor = None
        self.include_forecast_features = reward_config.get("include_forecast_features", False) if reward_config else False
        
        if self.include_forecast_features:
            try:
                from src.forecasting.simple_forecast_predictor import ForecastPredictor
                forecast_horizon = reward_config.get("forecast_horizon", 5) if reward_config else 5
                forecast_cache_steps = reward_config.get("forecast_cache_steps", 20) if reward_config else 20
                self.forecast_predictor = ForecastPredictor(
                    lookback_window=20,
                    forecast_horizon=forecast_horizon,
                    cache_steps=forecast_cache_steps
                )
                print(f"[OK] Forecast predictor initialized (cache_steps={forecast_cache_steps})")
            except Exception as e:
                print(f"[WARN] Could not initialize forecast predictor: {e}")
                self.forecast_predictor = None
                self.include_forecast_features = False
        
        # Calculate state dimension
        # Features per timeframe: OHLCV (5) + volume_ratio (1) + returns (1) + indicators (estimated 8)
        features_per_tf = 5 + 1 + 1 + 8  # ~15 features per timeframe
        base_state_dim = features_per_tf * len(self.timeframes) * self.lookback_bars
        
        # Add regime features if enabled (5 features: trending, ranging, volatile, confidence, duration)
        regime_features_dim = 5 if self.include_regime_features else 0
        # Add forecast features if enabled (3 features: direction, confidence, expected_return)
        forecast_features_dim = 3 if self.include_forecast_features else 0
        self.state_dim = base_state_dim + regime_features_dim + forecast_features_dim
        
        # Action space: continuous position size [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # State space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # Trading state
        self.state: Optional[TradeState] = None
        self.current_step = 0
        # Calculate max steps from data length
        data_max_steps = len(self.data[min(self.timeframes)]) - self.lookback_bars - 1
        # Use configured episode length limit if provided, otherwise use data length
        # This allows episodes to complete in reasonable time (e.g., 10K steps) even with very long data
        self.max_steps = max_episode_steps if max_episode_steps is not None and max_episode_steps < data_max_steps else data_max_steps
        self.data_max_steps = data_max_steps  # Store original data length for reference
        self._original_max_steps = self.max_steps  # Store original max_steps for restoration after adjustments
        self._original_max_steps = self.max_steps  # Store original max_steps for restoration after adjustments
        
        # Performance tracking
        self.equity_curve = [initial_capital]
        self.max_equity = initial_capital
        self.max_drawdown = 0.0
        
        # Quality filter configuration (simplified for training - mirrors DecisionGate)
        # These are applied during training to filter low-quality trades
        quality_config = reward_config.get("quality_filters", {})
        if quality_config.get("enabled", True):
            self.min_action_confidence = quality_config.get("min_action_confidence", 0.3)  # Minimum action magnitude to consider (proxy for confidence)
            self.min_quality_score = quality_config.get("min_quality_score", 0.5)  # Minimum quality score (simplified)
            self.require_positive_expected_value = quality_config.get("require_positive_expected_value", True)  # Reject trades with EV <= 0
        else:
            # Quality filters disabled - set permissive defaults
            self.min_action_confidence = 0.0
            self.min_quality_score = 0.0
            self.require_positive_expected_value = False
        
        # Track recent performance for expected value calculation
        self.recent_trades_pnl = []  # Track PnL of recent trades for EV calculation
        self.recent_trades_window = 50  # Use last N trades for EV calculation
        
        # Per-trade R:R tracking (for immediate feedback on trade quality)
        self.recent_trades_rr = []  # Track R:R of each trade at exit
        self.recent_trades_entry_prices = []  # Track entry prices for R:R calculation
        self.recent_trades_stop_losses = []  # Track stop loss levels for R:R calculation
        
        # CRITICAL FIX #1: Initialize bid-ask spread configuration
        # Bid-ask spread is a fundamental cost component that must be modeled for realistic training
        spread_config = reward_config.get("bid_ask_spread", {}) if reward_config else {}
        self.spread_enabled = spread_config.get("enabled", True)
        self.spread_pct = spread_config.get("spread_pct", 0.002)  # Default 0.2% spread (conservative for futures)
        if self.spread_enabled:
            print(f"  [CRITICAL FIX] Bid-ask spread: ENABLED ({self.spread_pct:.3%}) - realistic execution prices")
        else:
            print(f"  [WARN] Bid-ask spread: DISABLED - execution prices may overstate returns")
        
        # CRITICAL FIX #2: Initialize volatility-normalized position sizing configuration
        # Normalizes position sizes based on volatility so risk per trade is consistent
        vol_sizing_config = reward_config.get("volatility_position_sizing", {}) if reward_config else {}
        self.vol_sizing_enabled = vol_sizing_config.get("enabled", False)  # Disabled by default (opt-in)
        self.risk_per_trade_pct = vol_sizing_config.get("risk_per_trade_pct", 0.01)  # 1% risk per trade
        self.atr_period = vol_sizing_config.get("atr_period", 14)  # ATR period
        self.atr_multiplier = vol_sizing_config.get("atr_multiplier", 2.0)  # Stop loss in ATR multiples
        self.min_position_size = vol_sizing_config.get("min_position_size", 0.01)  # Minimum 1%
        self.max_position_size = vol_sizing_config.get("max_position_size", 1.0)  # Maximum 100%
        if self.vol_sizing_enabled:
            print(f"  [FIX #2] Volatility position sizing: ENABLED ({self.risk_per_trade_pct:.1%} risk per trade, ATR={self.atr_period})")
        else:
            print(f"  [INFO] Volatility position sizing: DISABLED - using fixed position sizes")
        
        # Initialize slippage model, market impact model, and execution quality tracker (Priority 1 optimization)
        slippage_config = reward_config.get("slippage", {}) if reward_config else {}
        market_impact_config = reward_config.get("market_impact", {}) if reward_config else {}
        
        if EXECUTION_QUALITY_AVAILABLE and SlippageModel:
            self.slippage_model = SlippageModel(slippage_config)
            self.execution_tracker = ExecutionQualityTracker()
            self.slippage_enabled = slippage_config.get("enabled", True)
            
            # Initialize market impact model
            if MarketImpactModel:
                self.market_impact_model = MarketImpactModel(market_impact_config)
                self.market_impact_enabled = market_impact_config.get("enabled", True)
            else:
                self.market_impact_model = None
                self.market_impact_enabled = False
            
            # Print initialization status
            import sys
            print(f"  [PRIORITY 1] Slippage model: {'Enabled' if self.slippage_enabled else 'Disabled'}")
            print(f"  [PRIORITY 1] Market impact model: {'Enabled' if self.market_impact_enabled else 'Disabled'}")
            print(f"  [PRIORITY 1] Execution quality tracker: Available")
            sys.stdout.flush()  # Force flush to ensure messages appear
        else:
            self.slippage_model = None
            self.execution_tracker = None
            self.market_impact_model = None
            self.slippage_enabled = False
            self.market_impact_enabled = False
            import sys
            print(f"  [WARN] Execution quality features: Not available (modules not imported)")
            sys.stdout.flush()  # Force flush to ensure messages appear
        
        # Reset episode tracking
        self._reset_episode_tracking()
        
        # Optional callback for trade logging (non-intrusive)
        # Initialize if not already set (allows external setup)
        self.trade_callback = None  # Set externally if journaling is enabled
        self._last_entry_price = None
        self._last_entry_step = None
        self._last_entry_position = None
        self._last_entry_action_confidence = None  # NEW: Store action confidence for journal
    
    def _reset_episode_tracking(self):
        """Reset episode-specific tracking variables"""
        self.episode_trades = 0
        self.total_commission_cost = 0.0
        self.last_position_change = 0.0
        self._steps_since_pause = 0  # Reset pause counter
        self.action_history = []  # Track actions for diversity calculation
        self.action_history_window = 50  # Use last N actions for diversity
        # Note: recent_trades_pnl persists across episodes for EV calculation
        # Reset per-trade R:R tracking at episode start
        if hasattr(self, 'recent_trades_rr'):
            # Keep last 10 trades for context, but clear older ones
            if len(self.recent_trades_rr) > 10:
                self.recent_trades_rr = self.recent_trades_rr[-10:]
        
        # Stop loss configuration (fixed - not adaptive)
        self.stop_loss_pct = self.reward_config.get("stop_loss_pct", 0.02)  # Default 2% stop loss
        
        # Trailing stop configuration (NEW - ATR-adaptive)
        trailing_stop_config = self.reward_config.get("trailing_stop", {}) if self.reward_config else {}
        self.trailing_stop_enabled = trailing_stop_config.get("enabled", True)  # Enabled by default
        self.trailing_stop_atr_multiplier = trailing_stop_config.get("atr_multiplier", 2.0)  # ATR multiplier for trailing stop
        self.trailing_stop_pct_fallback = trailing_stop_config.get("pct_fallback", 0.02)  # Fallback to 2% if ATR unavailable
        self.trailing_stop_min_distance_pct = trailing_stop_config.get("min_distance_pct", 0.005)  # Minimum 0.5% distance
        self.trailing_stop_max_distance_pct = trailing_stop_config.get("max_distance_pct", 0.05)  # Maximum 5% distance
        self.trailing_stop_activation_pct = trailing_stop_config.get("activation_pct", 0.01)  # Activate after 1% favorable move
        
        if self.trailing_stop_enabled:
            print(f"  [TRAILING STOP] ENABLED (ATR multiplier: {self.trailing_stop_atr_multiplier}x, fallback: {self.trailing_stop_pct_fallback:.1%})")
        else:
            print(f"  [TRAILING STOP] DISABLED - using fixed stop loss only")
        
        # Take profit configuration (NEW - ATR-adaptive soft targets)
        take_profit_config = self.reward_config.get("take_profit", {}) if self.reward_config else {}
        self.take_profit_enabled = take_profit_config.get("enabled", True)  # Enabled by default
        self.take_profit_mode = take_profit_config.get("mode", "soft_targets")  # soft_targets, partial_exits, hard_targets
        # Use R:R ratio to calculate target (e.g., if stop = 1 ATR, target = 2 ATR for 2:1 R:R)
        self.take_profit_rr_ratio = take_profit_config.get("risk_reward_ratio", self.reward_config.get("min_risk_reward_ratio", 2.0) if self.reward_config else 2.0)
        # OR use direct ATR multiplier (alternative approach)
        self.take_profit_atr_multiplier = take_profit_config.get("atr_multiplier", None)  # If None, use R:R ratio
        self.take_profit_pct_fallback = take_profit_config.get("pct_fallback", 0.04)  # Fallback to 4% if ATR unavailable (2x stop loss)
        self.take_profit_bonus_multiplier = take_profit_config.get("bonus_multiplier", 1.5)  # Reward bonus when target hit
        
        if self.take_profit_enabled:
            if self.take_profit_atr_multiplier:
                print(f"  [TAKE PROFIT] ENABLED (Mode: {self.take_profit_mode}, ATR multiplier: {self.take_profit_atr_multiplier}x)")
            else:
                print(f"  [TAKE PROFIT] ENABLED (Mode: {self.take_profit_mode}, R:R ratio: {self.take_profit_rr_ratio}:1)")
        else:
            print(f"  [TAKE PROFIT] DISABLED")
        
        # NEW: Read adaptive profitability parameters from adaptive training config
        import json
        from pathlib import Path
        
        adaptive_config_path = Path("logs/adaptive_training/current_reward_config.json")
        if adaptive_config_path.exists():
            try:
                with open(adaptive_config_path, 'r') as f:
                    adaptive_config = json.load(f)
                    # Read adaptive min_risk_reward_ratio
                    self.min_risk_reward_ratio = adaptive_config.get("min_risk_reward_ratio", self.reward_config.get("min_risk_reward_ratio", 1.5))
                    
                    # Read adaptive quality filters
                    quality_filters = adaptive_config.get("quality_filters", {})
                    if quality_filters:
                        self.min_action_confidence = quality_filters.get("min_action_confidence", self.reward_config.get("quality_filters", {}).get("min_action_confidence", 0.15))
                        self.min_quality_score = quality_filters.get("min_quality_score", self.reward_config.get("quality_filters", {}).get("min_quality_score", 0.4))
                    else:
                        # Fallback to config defaults
                        quality_filters_config = self.reward_config.get("quality_filters", {})
                        self.min_action_confidence = quality_filters_config.get("min_action_confidence", 0.15)
                        self.min_quality_score = quality_filters_config.get("min_quality_score", 0.4)
                    
                    # Read adaptive stop loss (NEW)
                    adaptive_stop_loss = adaptive_config.get("stop_loss_pct")
                    if adaptive_stop_loss is not None:
                        self.stop_loss_pct = adaptive_stop_loss
                        print(f"[ADAPTIVE] Using adaptive stop loss: {self.stop_loss_pct:.3f} ({self.stop_loss_pct*100:.1f}%)")
            except Exception as e:
                # Fallback to config defaults if adaptive config read fails
                self.min_risk_reward_ratio = self.reward_config.get("min_risk_reward_ratio", 1.5)
                quality_filters_config = self.reward_config.get("quality_filters", {})
                self.min_action_confidence = quality_filters_config.get("min_action_confidence", 0.15)
                self.min_quality_score = quality_filters_config.get("min_quality_score", 0.4)
        else:
            # No adaptive config - use defaults from reward_config
            self.min_risk_reward_ratio = self.reward_config.get("min_risk_reward_ratio", 1.5)
            quality_filters_config = self.reward_config.get("quality_filters", {})
            self.min_action_confidence = quality_filters_config.get("min_action_confidence", 0.15)
            self.min_quality_score = quality_filters_config.get("min_quality_score", 0.4)
    
    def _validate_data(self):
        """Validate that data timeframes align"""
        primary_df = self.data[min(self.timeframes)]
        
        for tf in self.timeframes[1:]:
            tf_df = self.data[tf]
            # Check that timeframes align (higher TF should have fewer bars)
            if len(tf_df) > len(primary_df):
                raise ValueError(f"Timeframe {tf} has more bars than primary {min(self.timeframes)}")
    
    def _get_regime_features(self, step: int) -> np.ndarray:
        """
        Get regime features for RL state.
        
        Returns:
            Array of 5 features: [trending, ranging, volatile, confidence, duration]
            Or zeros if regime detector not available
        """
        if not self.include_regime_features or self.regime_detector is None:
            return np.zeros(5, dtype=np.float32)
        
        try:
            primary_data = self.data[min(self.timeframes)]
            return self.regime_detector.get_regime_features(primary_data, step)
        except Exception as e:
            # If regime detection fails, return zeros
            return np.zeros(5, dtype=np.float32)
    
    def _get_forecast_features(self, step: int) -> np.ndarray:
        """
        Get forecast features for RL state.
        
        Returns:
            Array of 3 features: [direction, confidence, expected_return]
            - direction: -1 to +1 (bearish to bullish)
            - confidence: 0-1 (confidence in forecast)
            - expected_return: Expected price change % (negative = price down, positive = price up)
            Or zeros if forecast predictor not available
        """
        if not self.include_forecast_features or self.forecast_predictor is None:
            return np.zeros(3, dtype=np.float32)
        
        try:
            primary_data = self.data[min(self.timeframes)]
            safe_step = min(step, len(primary_data) - 1)
            
            # Get forecast features
            forecast_features = self.forecast_predictor.get_forecast_features(
                primary_data,
                current_step=safe_step
            )
            
            return np.array(forecast_features, dtype=np.float32)
        except Exception as e:
            # If forecast fails, return zeros
            print(warn(f"[WARN] Forecast feature extraction failed at step {step}: {e}"), flush=True)
            return np.zeros(3, dtype=np.float32)
    
    def _get_state_features(self, step: int) -> np.ndarray:
        """
        Extract state features for current step.
        
        Combines features from all timeframes.
        """
        features = []
        
        for tf in self.timeframes:
            tf_data = self.data[tf]
            
            # Get current bar index for this timeframe
            primary_step = step
            tf_step = min(primary_step, len(tf_data) - 1)
            
            # Get lookback window
            start_idx = max(0, tf_step - self.lookback_bars + 1)
            window = tf_data.iloc[start_idx:tf_step + 1].copy()
            
            # Extract features
            tf_features = self._extract_timeframe_features(window, tf_data, tf_step)
            features.extend(tf_features)
        
        # Pad if necessary
        feature_array = np.array(features, dtype=np.float32)
        if len(feature_array) < self.state_dim:
            padding = np.zeros(self.state_dim - len(feature_array), dtype=np.float32)
            feature_array = np.concatenate([feature_array, padding])
        
        # Add regime features if enabled (at the end of state vector)
        if self.include_regime_features:
            regime_features = self._get_regime_features(step)
            feature_array = np.concatenate([feature_array, regime_features])
        
        # Add forecast features if enabled (after regime features)
        if self.include_forecast_features:
            forecast_features = self._get_forecast_features(step)
            feature_array = np.concatenate([feature_array, forecast_features])
        
        # Normalize
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return feature_array[:self.state_dim]
    
    def _extract_timeframe_features(
        self,
        window: pd.DataFrame,
        full_data: pd.DataFrame,
        current_idx: int
    ) -> List[float]:
        """Extract features from a timeframe window"""
        features = []
        
        if len(window) == 0:
            return [0.0] * 15 * self.lookback_bars
        
        # Price features
        prices = window[["open", "high", "low", "close"]].values.flatten()
        features.extend(prices.tolist())
        
        # Volume features
        volumes = window["volume"].values
        features.extend(volumes.tolist())
        
        # Returns
        if len(window) > 1:
            returns = window["close"].pct_change().dropna().values
            features.extend(returns.tolist())
        else:
            features.extend([0.0])
        
        # Volume ratio (current vs average)
        if current_idx >= 20:
            # CRITICAL FIX: Add boundary check to prevent IndexError
            start_vol_idx = max(0, current_idx - 20)
            end_vol_idx = min(current_idx, len(full_data))
            if end_vol_idx > start_vol_idx:
                avg_volume = full_data["volume"].iloc[start_vol_idx:end_vol_idx].mean()
            else:
                avg_volume = window["volume"].iloc[-1] if len(window) > 0 else 1.0
            current_volume = window["volume"].iloc[-1] if len(window) > 0 else 0
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            features.append(volume_ratio)
        else:
            features.append(1.0)
        
        # Simple moving averages (if enough data)
        if len(window) >= 5:
            sma_5 = window["close"].iloc[-5:].mean()
            sma_10 = window["close"].iloc[-min(10, len(window)):].mean() if len(window) >= 10 else sma_5
            features.extend([sma_5, sma_10])
        else:
            features.extend([window["close"].iloc[-1], window["close"].iloc[-1]])
        
        # Price relative to range
        if len(window) > 1:
            high_low_range = window["high"].max() - window["low"].min()
            price_position = (window["close"].iloc[-1] - window["low"].min()) / high_low_range if high_low_range > 0 else 0.5
            features.append(price_position)
        else:
            features.append(0.5)
        
        # Pad to expected size
        expected_size = 15 * self.lookback_bars
        while len(features) < expected_size:
            features.append(0.0)
        
        return features[:expected_size]
    
    def _calculate_atr(self, safe_step: int, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR) for volatility measurement.
        
        CRITICAL FIX #2: ATR is used for volatility-normalized position sizing.
        
        Args:
            safe_step: Current step index (safe, within bounds)
            period: ATR period (default 14)
        
        Returns:
            ATR value (in price units)
        """
        primary_data = self.data[min(self.timeframes)]
        
        if safe_step < period:
            # Not enough data - return default ATR based on price
            if safe_step < len(primary_data):
                current_price = primary_data.iloc[safe_step]["close"]
                return current_price * 0.02  # Default 2% ATR
            return 0.0
        
        # Get data window for ATR calculation
        start_idx = max(0, safe_step - period + 1)
        window = primary_data.iloc[start_idx:safe_step + 1].copy()
        
        if len(window) < 2:
            return 0.0
        
        # Calculate True Range components
        high_low = window["high"] - window["low"]
        high_close_prev = abs(window["high"] - window["close"].shift(1))
        low_close_prev = abs(window["low"] - window["close"].shift(1))
        
        # True Range = max(high-low, |high-close_prev|, |low-close_prev|)
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # ATR = Simple Moving Average of True Range
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        # Fill NaN with default
        if pd.isna(atr) or atr <= 0:
            if safe_step < len(primary_data):
                current_price = primary_data.iloc[safe_step]["close"]
                return current_price * 0.02  # Default 2% ATR
            return 0.0
        
        return float(atr)
    
    def _normalize_position_by_volatility(
        self, 
        action_value: float, 
        current_price: float, 
        safe_step: int
    ) -> float:
        """
        Normalize position size based on volatility to ensure consistent risk per trade.
        
        CRITICAL FIX #2: Volatility-normalized position sizing.
        
        Formula:
        - Risk amount = capital * risk_per_trade_pct
        - Stop distance = ATR * atr_multiplier (or price * stop_loss_pct if ATR unavailable)
        - Normalized position = (risk_amount / stop_distance) / capital
        - Final position = action_value * normalized_multiplier
        
        Args:
            action_value: Raw action value from agent (-1.0 to 1.0)
            current_price: Current market price
            safe_step: Current step index
        
        Returns:
            Volatility-normalized position size (-1.0 to 1.0)
        """
        if not self.vol_sizing_enabled or current_price <= 0:
            return action_value  # Return unchanged if disabled or invalid price
        
        # Calculate ATR for volatility
        atr = self._calculate_atr(safe_step, period=self.atr_period)
        
        # Use ATR-based stop distance, fallback to percentage-based
        if atr > 0:
            stop_distance = atr * self.atr_multiplier
        else:
            # Fallback to percentage-based stop loss
            stop_distance = current_price * self.stop_loss_pct
        
        if stop_distance <= 0:
            return action_value  # Invalid stop distance - return unchanged
        
        # Calculate risk amount per trade
        risk_amount = self.initial_capital * self.risk_per_trade_pct
        
        # Calculate position size that would risk exactly risk_amount
        # Position size = (risk_amount / stop_distance) / (price * contract_size)
        # For normalized position: divide by capital to get 0-1 range
        # Simplified: normalized_position = risk_amount / stop_distance / capital * (capital / price)
        # = risk_amount / (stop_distance * price)
        
        # Calculate normalized position multiplier
        # If stop_distance = 2% of price, and we want to risk 1% of capital:
        # normalized_size = (1% capital) / (2% price * capital/price per unit)
        # For normalized positions (0-1), we scale by risk_per_trade_pct / (stop_distance / price)
        
        stop_distance_pct = stop_distance / current_price  # Stop loss as % of price
        normalized_multiplier = self.risk_per_trade_pct / stop_distance_pct
        
        # Apply multiplier to action value, then clamp
        normalized_position = action_value * normalized_multiplier
        
        # Clamp to min/max position size limits
        if normalized_position > 0:
            normalized_position = max(self.min_position_size, min(self.max_position_size, normalized_position))
        else:
            normalized_position = max(-self.max_position_size, min(-self.min_position_size, normalized_position))
        
        return float(normalized_position)
    
    def _calculate_trailing_stop(
        self,
        current_price: float,
        safe_step: int,
        entry_price: float,
        position: float
    ) -> Optional[float]:
        """
        Calculate ATR-adaptive trailing stop price.
        
        As a quant trader, ATR-based trailing stops are preferred because:
        - They adapt to volatility (wider in high vol, tighter in low vol)
        - Prevent getting stopped out by normal market noise
        - Allow tighter stops in calm markets to protect profits better
        
        Args:
            current_price: Current market price
            safe_step: Current step index
            entry_price: Entry price of position
            position: Current position size (positive = long, negative = short)
        
        Returns:
            Trailing stop price, or None if trailing stop not active
        """
        if not self.trailing_stop_enabled or position == 0:
            return None
        
        # Calculate ATR for volatility-adaptive stop distance
        atr = self._calculate_atr(safe_step, period=self.atr_period)
        
        # Calculate stop distance (ATR-based preferred, fallback to percentage)
        if atr > 0:
            stop_distance = atr * self.trailing_stop_atr_multiplier
        else:
            # Fallback to percentage-based if ATR unavailable
            stop_distance = current_price * self.trailing_stop_pct_fallback
        
        # Enforce min/max distance limits
        min_distance = current_price * self.trailing_stop_min_distance_pct
        max_distance = current_price * self.trailing_stop_max_distance_pct
        stop_distance = max(min_distance, min(max_distance, stop_distance))
        
        # Check if position has moved favorably enough to activate trailing stop
        if position > 0:  # Long position
            favorable_move = (current_price - entry_price) / entry_price
            if favorable_move < self.trailing_stop_activation_pct:
                # Not enough favorable move - trailing stop not active yet
                return None
            
            # Update highest price
            if self.state.highest_price is None or current_price > self.state.highest_price:
                self.state.highest_price = current_price
            
            # Trailing stop = highest price - stop_distance
            trailing_stop = self.state.highest_price - stop_distance
            
            # Never move stop against position (only tighten, never widen)
            if self.state.trailing_stop_price is not None:
                trailing_stop = max(trailing_stop, self.state.trailing_stop_price)
            
            # Ensure stop is below entry (don't allow negative risk)
            trailing_stop = min(trailing_stop, entry_price * (1 - self.trailing_stop_min_distance_pct))
            
        else:  # Short position
            favorable_move = (entry_price - current_price) / entry_price
            if favorable_move < self.trailing_stop_activation_pct:
                # Not enough favorable move - trailing stop not active yet
                return None
            
            # Update lowest price
            if self.state.lowest_price is None or current_price < self.state.lowest_price:
                self.state.lowest_price = current_price
            
            # Trailing stop = lowest price + stop_distance
            trailing_stop = self.state.lowest_price + stop_distance
            
            # Never move stop against position (only tighten, never widen)
            if self.state.trailing_stop_price is not None:
                trailing_stop = min(trailing_stop, self.state.trailing_stop_price)
            
            # Ensure stop is above entry (don't allow negative risk)
            trailing_stop = max(trailing_stop, entry_price * (1 + self.trailing_stop_min_distance_pct))
        
        return trailing_stop
    
    def _calculate_take_profit_target(
        self,
        current_price: float,
        safe_step: int,
        entry_price: float,
        position: float
    ) -> Optional[float]:
        """
        Calculate ATR-adaptive take profit target price.
        
        As a quant trader, ATR-based targets maintain consistent R:R ratios:
        - If stop = 1 ATR, target = 2 ATR → 2:1 R:R
        - If stop = 2 ATR, target = 4 ATR → 2:1 R:R
        - Adapts to volatility automatically
        
        Args:
            current_price: Current market price
            safe_step: Current step index
            entry_price: Entry price of position
            position: Current position size (positive = long, negative = short)
        
        Returns:
            Take profit target price, or None if not enabled
        """
        if not self.take_profit_enabled or position == 0:
            return None
        
        # Calculate ATR for volatility-adaptive target distance
        atr = self._calculate_atr(safe_step, period=self.atr_period)
        
        # Calculate target distance
        if atr > 0:
            # Use ATR multiplier if specified, otherwise use R:R ratio with stop loss
            if self.take_profit_atr_multiplier is not None:
                target_distance = atr * self.take_profit_atr_multiplier
            else:
                # Use R:R ratio: target = stop_distance × R:R_ratio
                # Stop distance is typically ATR × trailing_stop_atr_multiplier (or stop_loss_pct)
                stop_distance = atr * self.trailing_stop_atr_multiplier if self.trailing_stop_enabled else (current_price * self.stop_loss_pct)
                target_distance = stop_distance * self.take_profit_rr_ratio
        else:
            # Fallback to percentage-based if ATR unavailable
            if self.take_profit_atr_multiplier is not None:
                target_distance = current_price * (self.take_profit_atr_multiplier * 0.02)  # Assume 2% per ATR
            else:
                # Use R:R ratio with percentage stop loss
                target_distance = (current_price * self.stop_loss_pct) * self.take_profit_rr_ratio
        
        # Calculate target price
        if position > 0:  # Long position
            target_price = entry_price + target_distance
        else:  # Short position
            target_price = entry_price - target_distance
        
        return target_price
    
    def _apply_bid_ask_spread(self, price: float, is_buy: bool) -> float:
        """
        Apply bid-ask spread to execution price.
        
        CRITICAL FIX #1: Models realistic execution prices for training.
        - Buy orders execute at ASK (higher price)
        - Sell orders execute at BID (lower price)
        
        Args:
            price: Base price (close price)
            is_buy: True for buy orders, False for sell orders
        
        Returns:
            Execution price with spread applied
        """
        if not self.spread_enabled:
            return price
        
        spread_half = self.spread_pct / 2.0
        if is_buy:
            # Buy at ASK (higher price)
            return price * (1.0 + spread_half)
        else:
            # Sell at BID (lower price)
            return price * (1.0 - spread_half)
    
    def _calculate_commission_cost(self, position_change: float, old_position: float = 0.0, new_position: float = 0.0) -> float:
        """
        Calculate commission cost for a trade.
        
        CRITICAL FIX: Charge commission only ONCE per round trip trade.
        Standard practice: Charge on ENTRY only (when opening a new position).
        This prevents double-charging (entry + exit).
        """
        if abs(position_change) < self.action_threshold:
            return 0.0
        
        # CRITICAL FIX: Only charge commission when OPENING a position, not when closing
        # Opening: old_position == 0 and new_position != 0
        # Closing: old_position != 0 and new_position == 0
        # Reversing: old_position != 0 and new_position * old_position < 0 (handled separately)
        
        is_opening = abs(old_position) < 0.01 and abs(new_position) >= self.action_threshold
        is_reversing = abs(old_position) >= 0.01 and abs(new_position) >= 0.01 and (old_position * new_position < 0)
        
        if is_opening or is_reversing:
            # Charge commission only on entry (opening new position or reversing)
            # For reversal, charge based on the NEW position size
            position_size_for_commission = abs(new_position) if is_reversing else abs(position_change)
            commission_cost = position_size_for_commission * self.initial_capital * self.commission_rate
            return commission_cost
        else:
            # Closing position - no commission (already charged on entry)
            return 0.0
    
    def _get_market_metrics(self, safe_step: int) -> Dict:
        """
        Get market metrics for slippage calculation.
        
        Args:
            safe_step: Current step index (safe, within bounds)
        
        Returns:
            Dictionary with volatility, volume, avg_volume, timestamp
        """
        primary_data = self.data[min(self.timeframes)]
        
        # Get current volume
        current_volume = primary_data.iloc[safe_step]["volume"] if safe_step < len(primary_data) else 0.0
        
        # Calculate average volume (last 20 bars)
        lookback_start = max(0, safe_step - 20)
        avg_volume = primary_data.iloc[lookback_start:safe_step]["volume"].mean() if safe_step > 0 else current_volume
        
        # Calculate volatility (std of returns over last 20 bars)
        if safe_step > 0:
            lookback_data = primary_data.iloc[lookback_start:safe_step]
            if len(lookback_data) > 1:
                returns = lookback_data["close"].pct_change().dropna()
                volatility = returns.std() if len(returns) > 0 else 0.0
            else:
                volatility = 0.0
        else:
            volatility = 0.0
        
        # Get timestamp if available
        timestamp = None
        if "timestamp" in primary_data.columns and safe_step < len(primary_data):
            try:
                timestamp = primary_data.iloc[safe_step]["timestamp"]
                if isinstance(timestamp, pd.Timestamp):
                    timestamp = timestamp.to_pydatetime()
            except:
                timestamp = None
        
        return {
            "volatility": volatility,
            "volume": current_volume,
            "avg_volume": avg_volume,
            "timestamp": timestamp
        }
    
    def _calculate_overtrading_penalty(self) -> float:
        """Calculate penalty for overtrading"""
        if not self.reward_config.get("overtrading_penalty_enabled", True):
            return 0.0
        
        optimal_trades = self.reward_config.get("optimal_trades_per_episode", 50)
        if self.episode_trades <= optimal_trades:
            return 0.0
        
        # Penalty for each trade above optimal
        excess_trades = self.episode_trades - optimal_trades
        penalty_per_trade = 0.0001  # Small penalty per excess trade
        return excess_trades * penalty_per_trade
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if self.state is None:
            return 1.0
        
        total_trades = self.state.trades_count
        if total_trades < 10:  # Not enough data
            return 1.0
        
        # Calculate gross profit and loss from realized PnL
        # Note: This is a simplified calculation. In a full implementation,
        # we'd track winning and losing trades separately with their PnL.
        # For now, we use win rate as a proxy.
        win_rate = self.state.winning_trades / max(1, total_trades)
        
        # Estimate profit factor from win rate
        # If win rate > 50%, profit factor > 1.0 (assuming equal avg win/loss)
        # This is a simplification - actual profit factor should use actual PnL
        if win_rate > 0.5:
            # Rough estimate: profit factor increases with win rate
            estimated_profit_factor = 1.0 + (win_rate - 0.5) * 2.0
        else:
            estimated_profit_factor = win_rate * 2.0
        
        return estimated_profit_factor
    
    def _calculate_reward(self, prev_pnl: float, current_pnl: float, commission_cost: float = 0.0) -> float:
        """
        Calculate reward based on NET PnL (after commission) - PnL-aligned reward function
        
        CRITICAL FIX: Rewards must align with actual PnL. If PnL is negative, rewards should be negative.
        Bonuses are minimal and only apply when PnL is already positive.
        
        Changes:
        1. PnL is the PRIMARY signal (90%+ of reward)
        2. Bonuses only apply when PnL is positive
        3. Strong penalty when overall PnL is negative
        4. R:R check - penalize if actual R:R < required R:R
        """
        # Net PnL change (already includes commission deduction)
        net_pnl_change = (current_pnl - prev_pnl) / self.initial_capital
        total_pnl_normalized = current_pnl / self.initial_capital  # Overall PnL as % of capital
        
        # PRIMARY: PnL change is the main reward signal (90% weight)
        reward = self.reward_config["pnl_weight"] * net_pnl_change
        
        # Calculate actual R:R from recent trades
        actual_rr_ratio = 0.0
        if self.state and self.state.trades_count > 10:  # Need enough trades for reliable R:R
            avg_win = self.state.total_win_pnl / max(1, self.state.winning_trades) if self.state.winning_trades > 0 else 0.0
            avg_loss = self.state.total_loss_pnl / max(1, self.state.losing_trades) if self.state.losing_trades > 0 else 0.0
            # CRITICAL FIX #3: Division by zero guard
            if avg_loss > 0:
                actual_rr_ratio = avg_win / avg_loss
            elif avg_loss == 0 and avg_win > 0:
                # No losses yet - assume good R:R
                actual_rr_ratio = 10.0  # High R:R if no losses
        
        # CRITICAL: If overall PnL is negative, apply strong penalty
        if total_pnl_normalized < -0.01:  # More than 1% down
            # Strong penalty for negative PnL - this ensures rewards align with actual performance
            pnl_penalty = abs(total_pnl_normalized) * 0.5  # 50% of negative PnL as penalty
            reward -= pnl_penalty
        
        # CRITICAL: If actual R:R < required R:R, apply STRONG penalty (increased from 10% to 30%)
        # Current R:R is 0.71:1, but we need 2.0:1+ to be profitable with commission costs
        required_rr = self.min_risk_reward_ratio
        
        # Aggregate R:R penalty (lagging indicator - based on all trades)
        aggregate_rr_penalty = 0.0
        if actual_rr_ratio > 0 and actual_rr_ratio < required_rr:
            # STRONG penalty for poor aggregate R:R - this encourages the agent to achieve better R:R
            # Penalty scales from 0% (at required_rr) to 50% (at 0.5:1 or worse) - INCREASED from 30%
            min_rr = 0.5  # Minimum R:R we'll penalize (below this is catastrophic)
            if actual_rr_ratio < min_rr:
                aggregate_rr_penalty = 0.50  # Maximum 50% penalty for very poor R:R (increased from 30%)
            else:
                # Scale penalty linearly: 50% at min_rr, 0% at required_rr
                aggregate_rr_penalty = 0.50 * (required_rr - actual_rr_ratio) / (required_rr - min_rr)
            reward -= aggregate_rr_penalty
        
        # Per-trade R:R penalty (IMMEDIATE feedback - penalize trades that exit before target R:R)
        # This provides immediate feedback on each trade, not just aggregate stats
        per_trade_rr_penalty = 0.0
        per_trade_rr_bonus = 0.0
        
        # Initialize recent_trades_rr if not exists
        if not hasattr(self, 'recent_trades_rr'):
            self.recent_trades_rr = []
        
        # Check if we just closed a trade (have recent R:R data)
        if len(self.recent_trades_rr) > 0:
            last_trade_rr = abs(self.recent_trades_rr[-1])  # Use absolute value for comparison
            
            # Penalize if last trade exited before achieving target R:R (only for winning trades)
            # For losing trades (negative R:R), aggregate penalty already covers it
            if last_trade_rr > 0 and last_trade_rr < required_rr:
                # Trade was profitable but exited too early - STRONG penalty
                # Penalty scales: 30% for R:R=0, 0% for R:R=required_rr
                per_trade_rr_penalty = 0.30 * (required_rr - last_trade_rr) / required_rr  # Increased to 30%
                reward -= per_trade_rr_penalty
                
            # Reward trades that achieve good R:R (immediate positive feedback)
            if last_trade_rr >= required_rr:
                # Bonus: up to 20% for achieving target R:R or better
                per_trade_rr_bonus = min(0.20, (last_trade_rr - required_rr) / required_rr * 0.20)  # Increased to 20%
                reward += per_trade_rr_bonus
        
        # Aggregate R:R bonus (for overall good performance)
        if actual_rr_ratio >= required_rr:
            rr_bonus = min(0.10, (actual_rr_ratio - required_rr) / required_rr * 0.10)  # Up to 10% bonus
            reward += rr_bonus
        
        # Log reward components for debugging (every 100 steps to reduce noise)
        if hasattr(self, '_reward_log_counter'):
            self._reward_log_counter += 1
        else:
            self._reward_log_counter = 0
        
        # Log reward components for debugging (every 500 steps to reduce noise)
        if self._reward_log_counter % 500 == 0:
            debug_info = []
            if aggregate_rr_penalty > 0:
                debug_info.append(f"agg_rr={actual_rr_ratio:.2f}, agg_penalty={aggregate_rr_penalty:.3f}")
            if per_trade_rr_penalty > 0:
                debug_info.append(f"per_trade_penalty={per_trade_rr_penalty:.3f}")
            if per_trade_rr_bonus > 0:
                debug_info.append(f"per_trade_bonus={per_trade_rr_bonus:.3f}")
            if debug_info:
                print(f"[REWARD DEBUG] Step {self.current_step}: {', '.join(debug_info)}, final_reward={reward:.6f}", flush=True)
        
        # Risk penalty (drawdown) - apply earlier and more aggressively
        current_equity = self.initial_capital + current_pnl
        if current_equity > self.max_equity:
            self.max_equity = current_equity
        
        drawdown = (self.max_equity - current_equity) / self.max_equity if self.max_equity > 0 else 0.0
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        # FIX: Apply risk penalties earlier and more aggressively to prevent high drawdowns
        risk_penalty = self.reward_config.get("risk_penalty", 0.05)
        drawdown_penalty = self.reward_config.get("drawdown_penalty", 0.07)
        
        # Progressive risk penalty starting at 3% drawdown (instead of 10%)
        if drawdown > 0.03:  # Start penalizing at 3% drawdown
            # Scale penalty: 3-5% = light, 5-8% = medium, 8%+ = heavy
            if drawdown <= 0.05:
                penalty_multiplier = 0.5  # Light penalty for 3-5% drawdown
            elif drawdown <= 0.08:
                penalty_multiplier = 1.0  # Medium penalty for 5-8% drawdown
            else:
                penalty_multiplier = 2.0  # Heavy penalty for 8%+ drawdown
            
            risk_penalty_coef = risk_penalty * penalty_multiplier
            reward -= risk_penalty_coef * drawdown  # Penalize based on full drawdown amount
        
        # Drawdown penalty for severe drawdowns (>8%)
        if drawdown > 0.08:
            reward -= drawdown_penalty * (drawdown - 0.08)  # Additional penalty for excess above 8%
        
        # Action diversity bonus/penalty (MINIMAL - only to prevent saturation)
        action_diversity_bonus = 0.0
        if hasattr(self, 'action_value') and hasattr(self, 'action_history'):
            self.action_history.append(self.action_value)
            if len(self.action_history) > self.action_history_window:
                self.action_history.pop(0)
            
            # Only apply diversity bonus if PnL is positive (don't mask losses)
            if total_pnl_normalized > 0 and len(self.action_history) >= 10:
                action_variance = np.var(self.action_history[-self.action_history_window:])
                diversity_bonus_scale = self.reward_config.get("action_diversity_bonus", 0.0) * 0.1  # 10% of original
                action_diversity_bonus = diversity_bonus_scale * action_variance
                
                # Constant action penalty (always apply to prevent saturation)
                if len(self.action_history) >= 20:
                    recent_actions = self.action_history[-20:]
                    if len(set([round(a, 2) for a in recent_actions])) <= 2:
                        constant_penalty_scale = self.reward_config.get("constant_action_penalty", 0.0)
                        action_diversity_bonus -= constant_penalty_scale * 0.5  # 50% of original
        
        reward += action_diversity_bonus
        
        # Exploration bonus (ONLY if PnL is positive and very few trades)
        if self.state and self.reward_config.get("exploration_bonus_enabled", True):
            position_size = abs(self.state.position)
            if position_size > self.action_threshold:
                # Only apply if PnL is positive AND very few trades
                if total_pnl_normalized > 0 and self.episode_trades < 3:
                    exploration_scale = self.reward_config.get("exploration_bonus_scale", 0.00001) * 0.5  # 50% reduction
                    exploration_bonus = exploration_scale * position_size
                    reward += exploration_bonus
                
                # Minimal holding cost
                holding_cost = self.transaction_cost * 0.0005
                reward -= holding_cost
            else:
                # Inaction penalty - ALWAYS apply when not trading (encourages trading)
                # Apply penalty regardless of PnL to encourage trades when none are happening
                inaction_penalty = self._get_adaptive_inaction_penalty()
                # Apply stronger penalty if we have no trades at all (PnL = 0)
                if self.state and self.state.trades_count == 0:
                    inaction_penalty *= 2.0  # Double penalty if no trades yet
                reward -= inaction_penalty
        
        # NO loss mitigation - losses should be penalized fully
        # (Removed loss mitigation to ensure rewards align with PnL)
        
        # Penalize overtrading
        overtrading_penalty = self._calculate_overtrading_penalty()
        reward -= overtrading_penalty
        
        # Profit factor check - if unprofitable, reduce reward further
        if self.state and self.state.trades_count > 10:
            profit_factor = self._calculate_profit_factor()
            required_profit_factor = self.reward_config.get("profit_factor_required", 1.0)
            if profit_factor < required_profit_factor:
                # Moderate reduction if unprofitable (allows learning while discouraging unprofitable behavior)
                reward *= 0.5  # 50% reduction (was 30% = 70% reduction - too harsh)
        
        # Moderate scaling (3x instead of 5x) to keep rewards aligned with PnL
        reward *= 3.0
        
        return reward
    
    def _calculate_simplified_quality_score(self, action_confidence: float, current_price: float) -> float:
        """
        Calculate a simplified quality score for training (without swarm/reasoning).
        
        This mirrors DecisionGate's quality scoring but uses only available data:
        - Action confidence (magnitude)
        - Recent win rate
        - Market volatility (simplified)
        
        Args:
            action_confidence: Action magnitude (0-1), proxy for confidence
            current_price: Current market price
            
        Returns:
            Quality score (0-1)
        """
        score = 0.0
        
        # Confidence component (30% weight)
        score += 0.3 * action_confidence
        
        # Recent win rate component (30% weight)
        if len(self.recent_trades_pnl) > 0:
            recent_wins = sum(1 for pnl in self.recent_trades_pnl if pnl > 0)
            recent_win_rate = recent_wins / len(self.recent_trades_pnl)
            score += 0.3 * recent_win_rate
        else:
            # No recent trades - assume neutral
            score += 0.3 * 0.5
        
        # Market conditions component (20% weight)
        # Simplified: Use price volatility as proxy
        if self.current_step > 20:
            primary_data = self.data[min(self.timeframes)]
            # Ensure indices are within bounds
            safe_current_step = min(self.current_step, len(primary_data) - 1)
            start_idx = max(0, safe_current_step - 20)
            end_idx = min(safe_current_step + 1, len(primary_data))
            recent_prices = primary_data.iloc[start_idx:end_idx]["close"]
            if len(recent_prices) > 1:
                price_volatility = recent_prices.std() / recent_prices.mean() if recent_prices.mean() > 0 else 0.0
                # Higher volatility = better conditions (more opportunity)
                volatility_score = min(1.0, price_volatility * 100)  # Scale volatility
                score += 0.2 * volatility_score
            else:
                score += 0.2 * 0.5
        else:
            score += 0.2 * 0.5
        
        # Action threshold component (20% weight)
        # Higher action magnitude relative to threshold = better quality
        if self.action_threshold > 0 and hasattr(self, 'action_value'):
            threshold_ratio = abs(self.action_value) / self.action_threshold
            threshold_ratio = min(1.0, threshold_ratio / 2.0)  # Normalize
            score += 0.2 * threshold_ratio
        else:
            score += 0.2 * 0.5
        
        return min(1.0, max(0.0, score))
    
    def _calculate_expected_value_simplified(self) -> Optional[float]:
        """
        Calculate simplified expected value based on recent trade performance.
        
        Returns:
            Expected value (positive = profitable, negative = unprofitable), or None if insufficient data
        """
        if len(self.recent_trades_pnl) < 10:
            return None  # Not enough data
        
        # Calculate win rate and average win/loss
        winning_pnls = [pnl for pnl in self.recent_trades_pnl if pnl > 0]
        losing_pnls = [abs(pnl) for pnl in self.recent_trades_pnl if pnl < 0]
        
        if len(winning_pnls) == 0 and len(losing_pnls) == 0:
            return None
        
        win_rate = len(winning_pnls) / len(self.recent_trades_pnl) if len(self.recent_trades_pnl) > 0 else 0.0
        avg_win = np.mean(winning_pnls) if len(winning_pnls) > 0 else 0.0
        avg_loss = np.mean(losing_pnls) if len(losing_pnls) > 0 else 0.0
        
        if avg_loss == 0:
            return None  # Cannot calculate if no losses
        
        # Expected value = (win_rate * avg_win) - ((1 - win_rate) * avg_loss) - commission
        commission_cost = self.commission_rate * self.initial_capital
        expected_value = (win_rate * avg_win) - ((1 - win_rate) * avg_loss) - commission_cost
        
        return expected_value
    
    def _get_adaptive_inaction_penalty(self) -> float:
        """Get adaptive inaction penalty (can be adjusted during training)"""
        import json
        from pathlib import Path
        
        # Check for adaptive training config
        config_path = Path("logs/adaptive_training/current_reward_config.json")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return config.get("inaction_penalty", 0.0001)
            except:
                pass
        
        # Default penalty
        return 0.0001
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        super().reset(seed=seed)
        
        # CRITICAL FIX: Restore original max_steps if it was adjusted for short data
        if hasattr(self, '_original_max_steps'):
            self.max_steps = self._original_max_steps
        
        # Reset episode tracking
        self._reset_episode_tracking()
        
        # Clear forecast cache for new episode
        if self.forecast_predictor is not None and hasattr(self.forecast_predictor, 'clear_cache'):
            self.forecast_predictor.clear_cache()
        
        # CRITICAL FIX: Randomize episode start point to avoid all episodes starting at the same location
        # This prevents episodes from all hitting the end of data at the same time
        # Based on GitHub repo fix: https://github.com/goagiq/reinforcement_learning
        primary_data = self.data[min(self.timeframes)]
        data_len = len(primary_data)
        
        # Calculate maximum valid start step (must leave room for max_steps and lookback)
        # If data is shorter than max_steps, we can start anywhere (episode will end when data ends)
        max_valid_start = max(0, data_len - self.max_steps - self.lookback_bars - 1)
        min_start = self.lookback_bars
        
        # If data is shorter than max_steps, adjust max_steps to fit available data
        # This prevents episodes from always starting at the same place
        if data_len < self.max_steps + self.lookback_bars:
            # Data is shorter than requested episode length - adjust max_steps for this episode
            # We can use most of the data, leaving some buffer for lookback
            available_steps = max(0, data_len - self.lookback_bars - 1)
            # Store original max_steps and temporarily adjust
            self._original_max_steps = getattr(self, '_original_max_steps', self.max_steps)
            self.max_steps = min(self.max_steps, available_steps)
            max_valid_start = max(0, data_len - self.max_steps - self.lookback_bars - 1)
            print(f"[WARN] Data ({data_len} bars) shorter than requested max_steps ({self._original_max_steps}). Using {self.max_steps} steps.", flush=True)
        
        if max_valid_start > min_start:
            # Random start point within valid range
            import random
            start_step = random.randint(min_start, max_valid_start)
            # Verify we have enough data remaining (need at least lookback_bars + some buffer)
            # After start_step, we need at least lookback_bars remaining for state features
            remaining_after_start = data_len - start_step
            min_required_remaining = self.lookback_bars + 10  # Add buffer for safety
            if remaining_after_start < min_required_remaining:
                # Start position doesn't leave enough room - adjust to safe position
                safe_start = max(min_start, data_len - min_required_remaining)
                self.current_step = safe_start
                print(warn(f"[WARN] Episode reset: Adjusted start from {start_step} to {safe_start} (not enough data remaining: {remaining_after_start} < {min_required_remaining})"), flush=True)
            else:
                self.current_step = start_step
                print(f"[DEBUG] Episode reset: Starting at step {start_step} (data_len={data_len}, max_steps={self.max_steps}, remaining={remaining_after_start})", flush=True)
        else:
            # Data is too short - start at minimum required position
            # But ensure we have at least lookback_bars + 1 remaining
            if data_len - min_start <= self.lookback_bars:
                # Even starting at min_start doesn't leave enough room
                print(error(f"[ERROR] Episode reset: Data too short! data_len={data_len}, min_start={min_start}, lookback={self.lookback_bars}, remaining={data_len - min_start}"), flush=True)
                # Start at beginning anyway - episode will terminate immediately but at least won't crash
                self.current_step = min_start
            else:
                self.current_step = min_start
                print(f"[WARN] Episode reset: Data very short ({data_len} bars). Starting at step {min_start}, will end when data ends.", flush=True)
        # Reset max_equity and drawdown tracking for new episode
        self.max_equity = self.initial_capital
        self.max_drawdown = 0.0
        # Reset equity curve
        self.equity_curve = [self.initial_capital]
        
        # Optional callback for trade logging (non-intrusive)
        # Initialize if not already set (allows external setup)
        if not hasattr(self, 'trade_callback'):
            self.trade_callback = None
        if not hasattr(self, '_last_entry_price'):
            self._last_entry_price = None
        if not hasattr(self, '_last_entry_step'):
            self._last_entry_step = None
        if not hasattr(self, '_last_entry_position'):
            self._last_entry_position = None
        if not hasattr(self, '_last_entry_action_confidence'):
            self._last_entry_action_confidence = None
        
        # Track current episode for logging
        self._current_episode = getattr(options, 'episode', 0) if options else 0
        
        if not hasattr(self, '_reset_count'):
            self._reset_count = 0
        self._reset_count += 1
        
        self.state = TradeState(
            highest_price=None,  # Reset trailing stop tracking
            lowest_price=None,
            trailing_stop_price=None,
            position=0.0,
            entry_price=None,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            total_pnl=0.0,
            trades_count=0,
            winning_trades=0,
            losing_trades=0,
            consecutive_losses=0,
            trading_paused=False,
            total_win_pnl=0.0,
            total_loss_pnl=0.0
        )
        
        self.equity_curve = [self.initial_capital]
        self.max_equity = self.initial_capital
        self.max_drawdown = 0.0
        
        state = self._get_state_features(self.current_step)
        info = {"step": self.current_step}
        
        return state, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step"""
        action_value = float(action[0])
        action_value = np.clip(action_value, -1.0, 1.0)
        self.action_value = action_value  # Store for quality score calculation

        # CRITICAL FIX: Capture old position BEFORE any position changes for commission calculation
        old_position_before_update = self.state.position if self.state else 0.0

        prev_pnl = self.state.total_pnl if self.state else 0.0

        # Get current price with boundary check
        primary_data = self.data[min(self.timeframes)]
        # Ensure current_step is within data bounds
        safe_step = min(self.current_step, len(primary_data) - 1)
        if safe_step < 0:
            safe_step = 0
        current_price = primary_data.iloc[safe_step]["close"]
        
        # CRITICAL FIX #2: Apply volatility-normalized position sizing
        # Normalize position size based on volatility to ensure consistent risk per trade
        normalized_action_value = self._normalize_position_by_volatility(
            action_value, 
            current_price, 
            safe_step
        )
        
        # Update position (use normalized value)
        position_change = normalized_action_value - self.state.position
        new_position = normalized_action_value
        
        # Track trade info for journaling (set when trades execute, used after commission calculation)
        trade_info_for_journal = None
        
        # CRITICAL FIX: Initialize exit_price to avoid UnboundLocalError
        # exit_price will be assigned in specific code paths (stop loss, position reversal, position close)
        exit_price = None
        
        # Get consecutive loss limit (needed for stop loss logic)
        max_consecutive_losses = self.reward_config.get("max_consecutive_losses", 3)
        
        # CRITICAL FIX #3: Division by zero guard for entry_price
        if self.state.entry_price is not None and self.state.entry_price <= 0:
            # Invalid entry price - reset to None
            print(f"[WARN] Invalid entry_price detected: {self.state.entry_price}, resetting")
            self.state.entry_price = None
            # Reset trailing stop tracking
            self.state.highest_price = None
            self.state.lowest_price = None
            self.state.trailing_stop_price = None
        
        # Calculate PnL
        if self.state.entry_price is not None:
            # CRITICAL FIX #3: Division by zero guard
            if self.state.entry_price <= 0:
                # Invalid entry price - skip PnL calculation
                unrealized_pnl = 0.0
            else:
                # Unrealized PnL for current position
                price_change = (current_price - self.state.entry_price) / self.state.entry_price
                unrealized_pnl = self.state.position * price_change * self.initial_capital
            
            # NEW: Check take profit target (soft target - reward bonus, not forced exit)
            self._take_profit_bonus = 0.0  # Initialize bonus
            if self.take_profit_enabled and self.state.position != 0 and self.state.entry_price is not None and self.state.entry_price > 0:
                # Calculate and update take profit target
                take_profit_target = self._calculate_take_profit_target(
                    current_price=current_price,
                    safe_step=safe_step,
                    entry_price=self.state.entry_price,
                    position=self.state.position
                )
                
                # Update target in state
                if take_profit_target is not None:
                    self.state.take_profit_target_price = take_profit_target
                    
                    # Check if target is hit (for reward bonus)
                    if not self.state.take_profit_hit:
                        target_hit = False
                        if self.state.position > 0:  # Long position
                            target_hit = current_price >= take_profit_target
                        else:  # Short position
                            target_hit = current_price <= take_profit_target
                        
                        if target_hit:
                            self.state.take_profit_hit = True
                            # Calculate bonus reward for hitting target
                            # Bonus = (price beyond target) / entry_price * bonus_multiplier
                            if self.state.position > 0:
                                excess = (current_price - take_profit_target) / self.state.entry_price
                            else:
                                excess = (take_profit_target - current_price) / self.state.entry_price
                            
                            # Bonus scales with how far beyond target (encourages holding to target)
                            self._take_profit_bonus = excess * self.take_profit_bonus_multiplier * 0.1  # 10% of excess as bonus
            
            # NEW: Check trailing stop first (if enabled and active)
            trailing_stop_hit = False
            if self.trailing_stop_enabled and self.state.position != 0 and self.state.entry_price is not None and self.state.entry_price > 0:
                # Calculate and update trailing stop
                trailing_stop_price = self._calculate_trailing_stop(
                    current_price=current_price,
                    safe_step=safe_step,
                    entry_price=self.state.entry_price,
                    position=self.state.position
                )
                
                # Update trailing stop price in state
                if trailing_stop_price is not None:
                    self.state.trailing_stop_price = trailing_stop_price
                    
                    # Check if trailing stop is hit
                    if self.state.position > 0:  # Long position
                        trailing_stop_hit = current_price <= trailing_stop_price
                    else:  # Short position
                        trailing_stop_hit = current_price >= trailing_stop_price
            
            # CRITICAL FIX: Enforce stop loss to cap losses
            # Check loss from ENTRY price (not from previous step!)
            # Trailing stop takes priority if active, otherwise use fixed stop loss
            stop_loss_enabled = True  # RE-ENABLED - Stop loss is important risk management (45% win rate confirms trades can be profitable)
            fixed_stop_loss_hit = False
            
            if not trailing_stop_hit and stop_loss_enabled and self.state.position != 0 and self.state.entry_price is not None and self.state.entry_price > 0:
                # CRITICAL FIX #3: Division by zero guard - entry_price must be positive
                # Calculate loss from ENTRY price (not from previous step's price_change)
                if self.state.position > 0:  # Long position
                    loss_pct = (self.state.entry_price - current_price) / self.state.entry_price
                else:  # Short position
                    loss_pct = (current_price - self.state.entry_price) / self.state.entry_price
                
                # If loss exceeds stop loss, force close position
                if loss_pct >= self.stop_loss_pct:
                    fixed_stop_loss_hit = True
            
            # Close position if either trailing stop or fixed stop loss is hit
            if trailing_stop_hit or fixed_stop_loss_hit:
                    # Stop loss hit - force close position
                    # CRITICAL FIX: Calculate exit price based on entry price and stop loss, not current_price
                    # This ensures exit price matches the stop loss level, not wherever current_price happened to be
                    if fixed_stop_loss_hit and self.state.entry_price is not None and self.state.entry_price > 0:
                        # For fixed stop loss: exit at exactly the stop loss price
                        if self.state.position > 0:  # Long position
                            # Long stop loss: entry - (entry * stop_loss_pct)
                            stop_loss_price = self.state.entry_price * (1.0 - self.stop_loss_pct)
                        else:  # Short position
                            # Short stop loss: entry + (entry * stop_loss_pct)
                            stop_loss_price = self.state.entry_price * (1.0 + self.stop_loss_pct)
                        # Use stop loss price as base, then apply spread
                        is_buy_exit = self.state.position < 0  # Short position = buy to cover
                        exit_price = self._apply_bid_ask_spread(stop_loss_price, is_buy_exit)
                    else:
                        # For trailing stop: use current_price (trailing stop tracks current price)
                        is_buy_exit = self.state.position < 0  # Short position = buy to cover
                        exit_price = self._apply_bid_ask_spread(current_price, is_buy_exit)
                    
                    # CRITICAL FIX #3: Division by zero guard (already checked entry_price > 0 above)
                    old_pnl = (exit_price - self.state.entry_price) / self.state.entry_price * self.state.position
                    trade_pnl_amount = old_pnl * self.initial_capital
                    trade_pnl_for_journal = trade_pnl_amount  # Store for journal callback
                    
                    # Log stop loss hit for diagnostics
                    import logging
                    logger = logging.getLogger(__name__)
                    stop_type = "TRAILING STOP" if trailing_stop_hit else "FIXED STOP LOSS"
                    logger.debug(
                        f"[{stop_type}] Hit at step {self.current_step}: "
                        f"{'trailing_stop_price' if trailing_stop_hit else f'loss_pct={loss_pct:.2%}, stop_threshold={self.stop_loss_pct:.2%}'}, "
                        f"position={self.state.position:.3f}, pnl=${trade_pnl_amount:.2f}, "
                        f"entry_price=${self.state.entry_price:.2f}, exit_price=${exit_price:.2f} (base=${current_price:.2f})"
                    )
                    self.state.realized_pnl += trade_pnl_amount
                    
                    # Track win/loss
                    if old_pnl > 0:
                        self.state.winning_trades += 1
                        self.state.total_win_pnl += trade_pnl_amount
                        self.state.consecutive_losses = 0
                        self.state.trading_paused = False
                    else:
                        self.state.losing_trades += 1
                        self.state.total_loss_pnl += abs(trade_pnl_amount)
                        self.state.consecutive_losses += 1
                        if self.state.consecutive_losses >= max_consecutive_losses:
                            self.state.trading_paused = True
                    
                    self.state.trades_count += 1
                    self.episode_trades += 1
                    
                    # Log trade closing (stop loss)
                    outcome = "WIN" if old_pnl > 0 else "LOSS"
                    print(f"[TRADE CLOSE] Step {self.current_step}: {stop_type} | {outcome} | Entry=${self.state.entry_price:.2f} Exit=${exit_price:.2f} | PnL=${trade_pnl_amount:.2f} | Trades={self.state.trades_count}", flush=True)
                    
                    # Track PnL for expected value calculation
                    self.recent_trades_pnl.append(trade_pnl_amount)
                    if len(self.recent_trades_pnl) > self.recent_trades_window:
                        self.recent_trades_pnl.pop(0)
                    
                    # Track per-trade R:R (immediate feedback for stop loss exits)
                    if self._last_entry_price is not None:
                        entry_price = self._last_entry_price
                        position_direction = 1.0 if self.state.position > 0 else -1.0
                        stop_loss_price = entry_price * (1 - position_direction * self.stop_loss_pct)
                        risk = abs(entry_price - stop_loss_price)
                        reward_distance = abs(current_price - entry_price)
                        if risk > 0:
                            trade_rr = reward_distance / risk
                            # Stop loss hit means losing trade - negative R:R
                            if (position_direction > 0 and current_price < entry_price) or (position_direction < 0 and current_price > entry_price):
                                trade_rr = -trade_rr
                        else:
                            trade_rr = 0.0
                        if not hasattr(self, 'recent_trades_rr'):
                            self.recent_trades_rr = []
                        self.recent_trades_rr.append(trade_rr)
                        if len(self.recent_trades_rr) > self.recent_trades_window:
                            self.recent_trades_rr.pop(0)
                    
                    # Store trade info for journal callback (after commission is calculated)
                    if self.trade_callback and self._last_entry_price is not None:
                        # CRITICAL FIX: Ensure exit_price is assigned (it should be from line 1214, but guard against edge cases)
                        if exit_price is None:
                            # Fallback: calculate exit_price if somehow not assigned
                            is_buy_exit = self.state.position < 0 if self.state.position != 0 else False
                            exit_price = self._apply_bid_ask_spread(current_price, is_buy_exit)
                        trade_info_for_journal = {
                            "episode": getattr(self, '_current_episode', 0),
                            "step": self.current_step,
                            "entry_price": self._last_entry_price,
                            "exit_price": exit_price,  # Use exit_price with spread applied
                            "position_size": self._last_entry_position,
                            "pnl": trade_pnl_amount,
                            "entry_step": self._last_entry_step,
                            "action_confidence": self._last_entry_action_confidence if self._last_entry_action_confidence is not None else abs(self._last_entry_position)  # Use stored confidence or fallback
                        }
                    
                    # Close position
                    new_position = 0.0
                    position_change = -self.state.position
                    self.state.entry_price = None
                    # Reset trailing stop tracking when position closed
                    self.state.highest_price = None
                    self.state.lowest_price = None
                    self.state.trailing_stop_price = None
                    unrealized_pnl = 0.0
        else:
            unrealized_pnl = 0.0
        
        # If trading is paused due to consecutive losses, prevent new trades
        # NOTE: Also add auto-resume after N steps to prevent getting stuck paused
        if self.state.trading_paused and self.state.consecutive_losses >= max_consecutive_losses:
            # Auto-resume after 100 steps to prevent getting stuck (for training)
            # This allows episodes to continue even if trading is paused
            steps_since_pause = getattr(self, '_steps_since_pause', 0)
            if steps_since_pause >= 100:
                # Auto-resume after 100 steps of being paused
                self.state.trading_paused = False
                self.state.consecutive_losses = 0
                self._steps_since_pause = 0
            else:
                # Still paused - reject this trade
                self._steps_since_pause = steps_since_pause + 1
                position_change = 0.0
                new_position = self.state.position  # Keep current position
        else:
            # Not paused - reset counter
            self._steps_since_pause = 0
        
        # CRITICAL FIX: Check risk/reward ratio before allowing trade
        # Calculate estimated risk/reward ratio based on recent performance
        # Only enforce after enough trades to get reliable R:R estimate
        if abs(position_change) > self.action_threshold and self.state.trades_count > 20:  # Need at least 20 trades for reliable R:R
            # Calculate average win and loss from recent trades
            avg_win = self.state.total_win_pnl / max(1, self.state.winning_trades) if self.state.winning_trades > 0 else 0.0
            avg_loss = self.state.total_loss_pnl / max(1, self.state.losing_trades) if self.state.losing_trades > 0 else 0.0
            
            # If we have enough data, check risk/reward ratio
            if avg_loss > 0 and avg_win > 0:
                risk_reward_ratio = avg_win / avg_loss
                
                # Get floor value from adaptive config or use default
                # This floor is separate from adaptive learning floor - it's the absolute minimum to allow trades
                # Adaptive learning adjusts min_risk_reward_ratio (target), floor prevents catastrophic trades
                adaptive_config_path = Path("logs/adaptive_training/current_reward_config.json")
                min_acceptable_rr_floor = 1.0  # Default floor - reject trades below break-even (increased from 0.7 to prevent further losses)
                
                if adaptive_config_path.exists():
                    try:
                        with open(adaptive_config_path, 'r') as f:
                            adaptive_config = json.load(f)
                            # Read floor from adaptive config if available, otherwise use default
                            min_acceptable_rr_floor = adaptive_config.get("min_rr_floor", 0.7)
                    except Exception:
                        pass  # Use default if reading fails
                
                # Enforcement: Only reject trades if actual R:R is below the floor
                # This is more lenient than the adaptive target (min_risk_reward_ratio) to allow learning
                # The reward function will penalize poor R:R, encouraging improvement toward the target
                if risk_reward_ratio < min_acceptable_rr_floor:
                    # Risk/reward ratio catastrophically poor - reject trade to prevent further losses
                    # Note: The reward function will still penalize poor R:R, encouraging improvement
                    position_change = 0.0
                    new_position = self.state.position
        
        # Apply simplified quality filters (mirrors DecisionGate for training)
        # These filters reject low-quality trades during training
        quality_filters_enabled = self.reward_config.get("quality_filters", {}).get("enabled", True)
        if quality_filters_enabled and abs(position_change) > self.action_threshold:  # Only check quality if action is significant
            # Calculate simplified quality metrics
            action_confidence = abs(self.action_value)  # Use action magnitude as proxy for confidence
            quality_score = self._calculate_simplified_quality_score(action_confidence, current_price)
            expected_value = self._calculate_expected_value_simplified()
            
            # Apply quality filters
            if action_confidence < self.min_action_confidence:
                # Reject: Action confidence too low
                position_change = 0.0
                new_position = self.state.position
            elif quality_score < self.min_quality_score:
                # Reject: Quality score too low
                position_change = 0.0
                new_position = self.state.position
            elif self.require_positive_expected_value and expected_value is not None and expected_value <= 0:
                # Reject: Expected value is negative or zero
                # BUT: Allow trade if we have no trade history (expected_value is None)
                # This prevents blocking the first trades when there's no historical data
                if expected_value is None:
                    # No trade history yet - allow trade to proceed
                    pass
                else:
                    # Expected value is negative or zero - reject
                    position_change = 0.0
                    new_position = self.state.position
        
        # Store position change for commission calculation
        self.last_position_change = position_change
        
        # CRITICAL FIX: Store old position BEFORE updating for commission calculation
        old_position_before_update = self.state.position if self.state else 0.0
        
        # Realize PnL if position closed or reversed
        # INCREASED threshold from 0.001 to 0.05 (5%) to reduce overtrading and focus on quality trades
        if abs(position_change) > self.action_threshold:  # Significant position change (configurable threshold)
            # Trading is allowed - proceed with trade
            if self.state.position != 0 and new_position * self.state.position < 0:
                # Position reversed - realize old position
                if self.state.entry_price is not None:
                    # CRITICAL FIX: Apply bid-ask spread to exit price for position reversal
                    # Long: exit at BID (sell), Short: exit at ASK (buy to cover)
                    is_buy_exit = self.state.position < 0  # Short position = buy to cover
                    exit_price = self._apply_bid_ask_spread(current_price, is_buy_exit)
                    # Use exit_price with spread applied for PnL calculation
                    old_pnl = (exit_price - self.state.entry_price) / self.state.entry_price * self.state.position
                    trade_pnl_amount = old_pnl * self.initial_capital
                    trade_pnl_for_journal = trade_pnl_amount  # Store for journal callback
                    self.state.realized_pnl += trade_pnl_amount
                    if old_pnl > 0:
                        self.state.winning_trades += 1
                        self.state.total_win_pnl += trade_pnl_amount  # Track winning trade PnL
                        self.state.consecutive_losses = 0  # Reset on win
                        self.state.trading_paused = False  # Resume trading on win
                    else:
                        self.state.losing_trades += 1
                        self.state.total_loss_pnl += abs(trade_pnl_amount)  # Track losing trade PnL (absolute)
                        self.state.consecutive_losses += 1  # Increment consecutive losses
                        # Check if we should pause trading
                        if self.state.consecutive_losses >= max_consecutive_losses:
                            self.state.trading_paused = True
                    self.state.trades_count += 1
                    self.episode_trades += 1
                    
                    # Log trade closing (position reversal)
                    outcome = "WIN" if old_pnl > 0 else "LOSS"
                    old_direction = "LONG" if self.state.position > 0 else "SHORT"
                    new_direction = "LONG" if new_position > 0 else "SHORT"
                    print(f"[TRADE CLOSE] Step {self.current_step}: REVERSAL ({old_direction}→{new_direction}) | {outcome} | Entry=${self.state.entry_price:.2f} Exit=${exit_price:.2f} | PnL=${trade_pnl_amount:.2f} | Trades={self.state.trades_count}", flush=True)
                    
                    # Track PnL for expected value calculation
                    self.recent_trades_pnl.append(trade_pnl_amount)
                    if len(self.recent_trades_pnl) > self.recent_trades_window:
                        self.recent_trades_pnl.pop(0)  # Keep only recent N trades
                    
                    # Track per-trade R:R (immediate feedback)
                    if self._last_entry_price is not None:
                        entry_price = self._last_entry_price
                        position_direction = 1.0 if self.state.position > 0 else -1.0
                        stop_loss_price = entry_price * (1 - position_direction * self.stop_loss_pct)
                        risk = abs(entry_price - stop_loss_price)
                        reward_distance = abs(current_price - entry_price)
                        if risk > 0:
                            trade_rr = reward_distance / risk
                            if (position_direction > 0 and current_price < entry_price) or (position_direction < 0 and current_price > entry_price):
                                trade_rr = -trade_rr
                        else:
                            trade_rr = 0.0
                        self.recent_trades_rr.append(trade_rr)
                        if len(self.recent_trades_rr) > self.recent_trades_window:
                            self.recent_trades_rr.pop(0)
                    
                    # Store trade info for journal callback (called later after commission is calculated)
                    # CRITICAL FIX: Don't call callback here - use unified callback at end to prevent duplicates
                    if self.trade_callback and self._last_entry_price is not None:
                        # CRITICAL FIX: Ensure exit_price is assigned (it should be from line 1396, but guard against edge cases)
                        if exit_price is None:
                            # Fallback: calculate exit_price if somehow not assigned
                            is_buy_exit = self.state.position < 0 if self.state.position != 0 else False
                            exit_price = self._apply_bid_ask_spread(current_price, is_buy_exit)
                        trade_info_for_journal = {
                            "episode": getattr(self, '_current_episode', 0),
                            "step": self.current_step,
                            "entry_price": self._last_entry_price,
                            "exit_price": exit_price,  # Use exit_price with spread applied
                            "position_size": self._last_entry_position,
                            "pnl": trade_pnl_amount,
                            "entry_step": self._last_entry_step,
                            "action_confidence": self._last_entry_action_confidence if self._last_entry_action_confidence is not None else abs(self._last_entry_position)
                        }
                    
                    # Clear old entry price before setting new one for reversal
                    old_entry_price = self.state.entry_price
                    self.state.entry_price = None
                    # Reset trailing stop tracking for position reversal
                    self.state.highest_price = None
                    self.state.lowest_price = None
                    self.state.trailing_stop_price = None
            elif self.state.position != 0 and abs(new_position) < self.action_threshold:
                # Position closed
                if self.state.entry_price is not None and self.state.entry_price > 0:
                    # CRITICAL FIX #1: Apply bid-ask spread to exit price
                    # Long position: exit at BID (sell), Short position: exit at ASK (buy to cover)
                    is_buy_exit = self.state.position < 0  # Short position = buy to cover
                    exit_price = self._apply_bid_ask_spread(current_price, is_buy_exit)
                    # CRITICAL FIX #3: Division by zero guard (entry_price > 0 checked above)
                    old_pnl = (exit_price - self.state.entry_price) / self.state.entry_price * self.state.position
                    trade_pnl_amount = old_pnl * self.initial_capital
                    trade_pnl_for_journal = trade_pnl_amount  # Store for journal callback
                    self.state.realized_pnl += trade_pnl_amount
                    if old_pnl > 0:
                        self.state.winning_trades += 1
                        self.state.total_win_pnl += trade_pnl_amount  # Track winning trade PnL
                        self.state.consecutive_losses = 0  # Reset on win
                        self.state.trading_paused = False  # Resume trading on win
                    else:
                        self.state.losing_trades += 1
                        self.state.total_loss_pnl += abs(trade_pnl_amount)  # Track losing trade PnL (absolute)
                        self.state.consecutive_losses += 1  # Increment consecutive losses
                        # Check if we should pause trading
                        if self.state.consecutive_losses >= max_consecutive_losses:
                            self.state.trading_paused = True
                    self.state.trades_count += 1
                    self.episode_trades += 1
                    
                    # Log trade closing (position closed)
                    outcome = "WIN" if old_pnl > 0 else "LOSS"
                    direction = "LONG" if self.state.position > 0 else "SHORT"
                    print(f"[TRADE CLOSE] Step {self.current_step}: {direction} CLOSED | {outcome} | Entry=${self.state.entry_price:.2f} Exit=${exit_price:.2f} | PnL=${trade_pnl_amount:.2f} | Trades={self.state.trades_count}", flush=True)
                    
                    # Track PnL for expected value calculation
                    self.recent_trades_pnl.append(trade_pnl_amount)
                    if len(self.recent_trades_pnl) > self.recent_trades_window:
                        self.recent_trades_pnl.pop(0)  # Keep only recent N trades
                    
                    # Track per-trade R:R (immediate feedback)
                    if self._last_entry_price is not None:
                        entry_price = self._last_entry_price
                        position_direction = 1.0 if self.state.position > 0 else -1.0
                        stop_loss_price = entry_price * (1 - position_direction * self.stop_loss_pct)
                        risk = abs(entry_price - stop_loss_price)
                        reward_distance = abs(current_price - entry_price)
                        if risk > 0:
                            trade_rr = reward_distance / risk
                            if (position_direction > 0 and current_price < entry_price) or (position_direction < 0 and current_price > entry_price):
                                trade_rr = -trade_rr
                        else:
                            trade_rr = 0.0
                        self.recent_trades_rr.append(trade_rr)
                        if len(self.recent_trades_rr) > self.recent_trades_window:
                            self.recent_trades_rr.pop(0)
                    
                    # Store trade info for journal callback (after commission is calculated)
                    if self.trade_callback and self._last_entry_price is not None:
                        # CRITICAL FIX: Ensure exit_price is assigned (it should be from line 1461, but guard against edge cases)
                        if exit_price is None:
                            # Fallback: calculate exit_price if somehow not assigned
                            is_buy_exit = self.state.position < 0 if self.state.position != 0 else False
                            exit_price = self._apply_bid_ask_spread(current_price, is_buy_exit)
                        trade_info_for_journal = {
                            "episode": getattr(self, '_current_episode', 0),
                            "step": self.current_step,
                            "entry_price": self._last_entry_price,
                            "exit_price": exit_price,  # Use exit_price with spread applied
                            "position_size": self._last_entry_position,
                            "pnl": trade_pnl_amount,
                            "entry_step": self._last_entry_step,
                            "action_confidence": self._last_entry_action_confidence if self._last_entry_action_confidence is not None else abs(self._last_entry_position)  # Use stored confidence or fallback
                        }
                self.state.entry_price = None
                # Reset trailing stop tracking when position closed
                self.state.highest_price = None
                self.state.lowest_price = None
                self.state.trailing_stop_price = None
                # Reset take profit tracking when position closed
                self.state.take_profit_target_price = None
                self.state.take_profit_hit = False
            elif self.state.entry_price is None and abs(new_position) > self.action_threshold:
                # New position opened - apply slippage
                market_metrics = self._get_market_metrics(safe_step)
                
                if self.slippage_enabled and self.slippage_model:
                    is_buy = new_position > 0
                    
                    # First apply market impact (price moves due to order size)
                    if self.market_impact_enabled and self.market_impact_model:
                        price_after_impact = self.market_impact_model.apply_market_impact(
                            intended_price=current_price,
                            order_size=abs(new_position),
                            is_buy=is_buy,
                            avg_volume=market_metrics.get("avg_volume", 1.0),
                            volatility=market_metrics.get("volatility")
                        )
                        market_impact = abs(price_after_impact - current_price) / current_price
                    else:
                        price_after_impact = current_price
                        market_impact = 0.0
                    
                    # Then apply slippage (execution quality)
                    execution_price = self.slippage_model.apply_slippage(
                        intended_price=price_after_impact,
                        order_size=abs(new_position),
                        is_buy=is_buy,
                        volatility=market_metrics.get("volatility"),
                        volume=market_metrics.get("volume"),
                        avg_volume=market_metrics.get("avg_volume"),
                        timestamp=market_metrics.get("timestamp")
                    )
                    
                    # Track execution quality
                    if self.execution_tracker:
                        self.execution_tracker.track_execution(
                            expected_price=current_price,
                            actual_price=execution_price,
                            order_size=abs(new_position),
                            fill_time=market_metrics.get("timestamp") or datetime.now(),
                            market_impact=market_impact,
                            volatility=market_metrics.get("volatility"),
                            volume=market_metrics.get("volume")
                        )
                    
                    self.state.entry_price = execution_price
                    # Reset trailing stop tracking for new position
                    self.state.highest_price = None
                    self.state.lowest_price = None
                    self.state.trailing_stop_price = None
                else:
                    # CRITICAL FIX #1: Apply bid-ask spread to entry price
                    # Buy orders: pay ASK (higher), Sell orders: receive BID (lower)
                    is_buy_entry = new_position > 0
                    self.state.entry_price = self._apply_bid_ask_spread(current_price, is_buy_entry)
                    # Reset trailing stop tracking for new position
                    self.state.highest_price = None
                    self.state.lowest_price = None
                    self.state.trailing_stop_price = None
                
                # Track entry for trade logging
                self._last_entry_price = self.state.entry_price
                self._last_entry_step = self.current_step
                self._last_entry_position = new_position
                self._last_entry_action_confidence = abs(self.action_value)  # Store actual action confidence
                
                self.episode_trades += 1
                
                # Log trade opening
                direction = "LONG" if new_position > 0 else "SHORT"
                print(f"[TRADE OPEN] Step {self.current_step}: {direction} @ ${self.state.entry_price:.2f} (size={new_position:.3f}, action={self.action_value:.3f})", flush=True)
        
        # Calculate commission cost for this trade
        # CRITICAL FIX: Pass old and new positions to prevent double-charging (entry + exit)
        commission_cost = self._calculate_commission_cost(position_change, old_position_before_update, new_position)
        self.total_commission_cost += commission_cost
        
        # Subtract commission from realized PnL (net profit)
        if abs(position_change) > self.action_threshold:
            self.state.realized_pnl -= commission_cost
            
            # Non-intrusive trade logging callback (called after commission is calculated)
            # This captures all trade types: position closed, reversed, or stop loss
            if self.trade_callback and trade_info_for_journal is not None:
                try:
                    self.trade_callback(
                        episode=trade_info_for_journal["episode"],
                        step=trade_info_for_journal["step"],
                        entry_price=trade_info_for_journal["entry_price"],
                        exit_price=trade_info_for_journal["exit_price"],
                        position_size=trade_info_for_journal["position_size"],
                        pnl=trade_info_for_journal["pnl"],
                        commission=commission_cost,
                        entry_step=trade_info_for_journal["entry_step"],
                        action_confidence=trade_info_for_journal.get("action_confidence", abs(trade_info_for_journal["position_size"]))  # Pass actual confidence
                    )
                    # Clear entry tracking after callback
                    self._last_entry_price = None
                    self._last_entry_step = None
                    self._last_entry_position = None
                    self._last_entry_action_confidence = None
                except Exception:
                    pass  # Don't let logging break training
        
        # Update state
        self.state.position = new_position
        self.state.unrealized_pnl = unrealized_pnl
        self.state.total_pnl = self.state.realized_pnl + self.state.unrealized_pnl
        
        # Calculate reward (using net profit after commission)
        reward = self._calculate_reward(prev_pnl, self.state.total_pnl, commission_cost)
        
        # NEW: Add take profit bonus if target was hit (soft target - reward bonus only)
        # This is calculated in the step() method before reward calculation
        if hasattr(self, '_take_profit_bonus') and self._take_profit_bonus > 0:
            reward += self._take_profit_bonus
            self._take_profit_bonus = 0.0  # Reset after applying
        
        # Update equity curve
        current_equity = self.initial_capital + self.state.total_pnl
        self.equity_curve.append(current_equity)
        
        # Non-intrusive equity logging callback (every 10 steps to reduce overhead)
        if self.trade_callback and self.current_step % 10 == 0:
            try:
                equity_callback = getattr(self.trade_callback, 'log_equity', None)
                if equity_callback:
                    equity_callback(
                        episode=getattr(self, '_current_episode', 0),
                        step=self.current_step,
                        equity=current_equity,
                        cumulative_pnl=self.state.total_pnl
                    )
            except Exception:
                pass  # Don't let logging break training
        
        # Next step
        self.current_step += 1
        
        # Check if done
        # Note: current_step is 0-indexed, so if max_steps=10000, valid steps are 0-9999
        # After step 9999, current_step will be 10000, which should trigger termination
        terminated = self.current_step >= self.max_steps
        truncated = False  # Can add early stopping logic here
        
        # DEBUG: Log if episode terminates on first step
        if terminated and self.current_step <= 2:
            print(error(f"[ERROR] Episode terminated by max_steps check: current_step={self.current_step}, max_steps={self.max_steps}"), flush=True)
        
        # Get next state
        if not terminated:
            # Ensure current_step is within data bounds before getting state features
            primary_data = self.data[min(self.timeframes)]
            safe_step = min(self.current_step, len(primary_data) - 1)
            if safe_step < 0:
                safe_step = 0
            
            # CRITICAL FIX: Check if we've exceeded data bounds
            # Only terminate if we truly can't proceed (need lookback_bars ahead)
            # Account for where episode started (via max_steps limit)
            # FIX: Use safe_step (not current_step) and check if we can get lookback_bars ahead
            # remaining_data = how many bars are left AFTER safe_step (including safe_step itself)
            # We need at least lookback_bars + 1 remaining (1 for current bar, lookback_bars for features)
            remaining_data = len(primary_data) - safe_step
            min_required = self.lookback_bars + 1  # Need current bar + lookback_bars for next state
            if remaining_data < min_required:
                # Not enough data remaining for lookback - terminate episode
                # Only log if we're actually at the end (not just starting)
                if safe_step > self.lookback_bars + 10:  # Only log if we've made some progress
                    if remaining_data > 0:
                        print(f"[INFO] Episode ending: Reached end of data (remaining={remaining_data} < required={min_required})", flush=True)
                    else:
                        print(f"[INFO] Episode ending: Reached end of data (safe_step={safe_step}, data_len={len(primary_data)})", flush=True)
                else:
                    # Episode started too close to end - this shouldn't happen with proper reset logic
                    print(warn(f"[WARN] Episode terminating early: safe_step={safe_step}, remaining={remaining_data}, required={min_required}, lookback={self.lookback_bars}, data_len={len(primary_data)}, current_step={self.current_step}"), flush=True)
                    # DEBUG: Check if this is happening on first step
                    if self.current_step <= self.lookback_bars + 2:
                        print(error(f"[ERROR] Episode terminated on step {self.current_step}! This indicates a reset() bug. Check reset() logic."), flush=True)
                        # Additional debug info
                        print(error(f"[ERROR] Debug: data_len={len(primary_data)}, lookback={self.lookback_bars}, max_steps={self.max_steps}"), flush=True)
                terminated = True
                next_state = np.zeros(self.state_dim, dtype=np.float32)
            else:
                try:
                    next_state = self._get_state_features(safe_step)
                except (IndexError, KeyError) as e:
                    # CRITICAL FIX: Catch exceptions in state feature extraction and terminate gracefully
                    print(error(f"[ERROR] Exception in _get_state_features at step {self.current_step} (safe_step={safe_step}): {e}"), flush=True)
                    import traceback
                    traceback.print_exc()
                    terminated = True
                    next_state = np.zeros(self.state_dim, dtype=np.float32)
        else:
            next_state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Calculate average win/loss for monitoring
        avg_win = self.state.total_win_pnl / max(1, self.state.winning_trades) if self.state.winning_trades > 0 else 0.0
        avg_loss = self.state.total_loss_pnl / max(1, self.state.losing_trades) if self.state.losing_trades > 0 else 0.0
        risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        
        # Execution quality metrics (Priority 1 optimization)
        execution_quality = {}
        if self.execution_tracker:
            exec_stats = self.execution_tracker.get_statistics()
            execution_quality = {
                "avg_slippage": exec_stats.get("avg_slippage", 0.0),
                "median_slippage": exec_stats.get("median_slippage", 0.0),
                "p95_slippage": exec_stats.get("p95_slippage", 0.0),
                "total_executions": exec_stats.get("total_executions", 0)
            }
        
        # Info
        info = {
            "step": self.current_step,
            "position": self.state.position,
            "pnl": self.state.total_pnl,
            "trades": self.state.trades_count,  # Cumulative trades (for backward compatibility)
            "episode_trades": self.episode_trades,  # Episode-specific trade count (resets each episode)
            "win_rate": self.state.winning_trades / max(1, self.state.trades_count),
            "equity": current_equity,
            "max_drawdown": self.max_drawdown,
            "commission_cost": commission_cost,
            "total_commission_cost": self.total_commission_cost,
            "net_pnl": self.state.total_pnl,  # Already includes commission deduction
            "avg_win": avg_win,  # Average winning trade PnL
            "avg_loss": avg_loss,  # Average losing trade PnL
            "risk_reward_ratio": risk_reward_ratio,  # Risk/reward ratio (avg_win / avg_loss)
            "execution_quality": execution_quality  # Execution quality metrics
        }
        
        return next_state, reward, terminated, truncated, info
    
    def render(self):
        """Render environment (print current state)"""
        if self.state:
            print(f"Step: {self.current_step}")
            print(f"Position: {self.state.position:.2f}")
            print(f"PnL: ${self.state.total_pnl:.2f}")
            print(f"Trades: {self.state.trades_count}")
            print(f"Equity: ${self.initial_capital + self.state.total_pnl:.2f}")


# Example usage
if __name__ == "__main__":
    # This is a template - you'll need actual data
    print("Trading Environment created.")
    print("To use:")
    print("1. Load data using DataExtractor")
    print("2. Create environment with multi-timeframe data")
    print("3. Use with Stable-Baselines3 or custom RL algorithm")

