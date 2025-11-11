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
        
        # Calculate state dimension
        # Features per timeframe: OHLCV (5) + volume_ratio (1) + returns (1) + indicators (estimated 8)
        features_per_tf = 5 + 1 + 1 + 8  # ~15 features per timeframe
        self.state_dim = features_per_tf * len(self.timeframes) * self.lookback_bars
        
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
        
        # Reset episode tracking
        self._reset_episode_tracking()
    
    def _reset_episode_tracking(self):
        """Reset episode-specific tracking variables"""
        self.episode_trades = 0
        self.total_commission_cost = 0.0
        self.last_position_change = 0.0
        self._steps_since_pause = 0  # Reset pause counter
        # Note: recent_trades_pnl persists across episodes for EV calculation
        
        # Stop loss configuration (fixed - not adaptive)
        self.stop_loss_pct = self.reward_config.get("stop_loss_pct", 0.02)  # Default 2% stop loss
        
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
    
    def _calculate_commission_cost(self, position_change: float) -> float:
        """Calculate commission cost for a trade"""
        if abs(position_change) < self.action_threshold:
            return 0.0
        
        # Commission = position_change * capital * commission_rate
        commission_cost = abs(position_change) * self.initial_capital * self.commission_rate
        return commission_cost
    
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
        Calculate reward based on NET PnL (after commission) and risk - optimized for profitability
        
        Changes from previous version:
        1. Uses net PnL (after commission) instead of gross PnL
        2. Balanced exploration bonus (reduced, only if few trades)
        3. Reduced loss mitigation (5% instead of 30%)
        4. Penalizes overtrading
        5. Checks profit factor requirement
        """
        # Net PnL change (already includes commission deduction)
        net_pnl_change = (current_pnl - prev_pnl) / self.initial_capital
        
        # Risk penalty (drawdown) - but only penalize significant drawdowns
        current_equity = self.initial_capital + current_pnl
        if current_equity > self.max_equity:
            self.max_equity = current_equity
        
        drawdown = (self.max_equity - current_equity) / self.max_equity if self.max_equity > 0 else 0.0
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        # Risk penalties (reduced)
        risk_penalty_coef = self.reward_config.get("risk_penalty", 0.5) * 0.05
        drawdown_penalty_coef = self.reward_config.get("drawdown_penalty", 0.3) * 0.05
        
        # Base reward components - focus on NET PnL (after commission)
        reward = (
            self.reward_config["pnl_weight"] * net_pnl_change
            - risk_penalty_coef * drawdown
            - drawdown_penalty_coef * max(0, self.max_drawdown - 0.20)
        )
        
        # BALANCED exploration bonus (reduced, only if few trades)
        if self.state and self.reward_config.get("exploration_bonus_enabled", True):
            position_size = abs(self.state.position)
            if position_size > self.action_threshold:
                # Only apply exploration bonus if we haven't had many trades recently
                # This balances between encouraging trading and preventing overtrading
                if self.episode_trades < 5:  # Only if very few trades
                    exploration_scale = self.reward_config.get("exploration_bonus_scale", 0.00001)
                    exploration_bonus = exploration_scale * position_size  # 10x reduction from 0.0001
                    reward += exploration_bonus
                
                # Minimal holding cost
                holding_cost = self.transaction_cost * 0.0005
                reward -= holding_cost
            else:
                # Adaptive inaction penalty (reduced)
                inaction_penalty = self._get_adaptive_inaction_penalty() * 0.5  # 50% reduction
                reward -= inaction_penalty
        
        # REDUCED loss mitigation (5% instead of 30%)
        if net_pnl_change < 0:
            loss_mitigation_coef = self.reward_config.get("loss_mitigation", 0.05)  # 5% mitigation
            loss_mitigation = abs(net_pnl_change) * loss_mitigation_coef
            reward += loss_mitigation  # Add back small portion of loss (reduces penalty)
        
        # Penalize overtrading
        overtrading_penalty = self._calculate_overtrading_penalty()
        reward -= overtrading_penalty
        
        # Profit factor requirement
        profit_factor = self._calculate_profit_factor()
        required_profit_factor = self.reward_config.get("profit_factor_required", 1.0)
        if profit_factor < required_profit_factor:
            # Reduce reward if profit factor is below requirement
            reward *= 0.5  # 50% reduction if unprofitable
        
        # Reduced scaling (5x instead of 10x) for more granular learning
        reward *= 5.0
        
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
        
        # Reset episode tracking
        self._reset_episode_tracking()
        
        self.current_step = self.lookback_bars
        # Reset max_equity and drawdown tracking for new episode
        self.max_equity = self.initial_capital
        self.max_drawdown = 0.0
        # Reset equity curve
        self.equity_curve = [self.initial_capital]
        
        if not hasattr(self, '_reset_count'):
            self._reset_count = 0
        self._reset_count += 1
        
        self.state = TradeState(
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

        prev_pnl = self.state.total_pnl if self.state else 0.0

        # Get current price with boundary check
        primary_data = self.data[min(self.timeframes)]
        # Ensure current_step is within data bounds
        safe_step = min(self.current_step, len(primary_data) - 1)
        if safe_step < 0:
            safe_step = 0
        current_price = primary_data.iloc[safe_step]["close"]
        
        # Update position
        position_change = action_value - self.state.position
        new_position = action_value
        
        # Get consecutive loss limit (needed for stop loss logic)
        max_consecutive_losses = self.reward_config.get("max_consecutive_losses", 3)
        
        # Calculate PnL
        if self.state.entry_price is not None:
            # Unrealized PnL for current position
            price_change = (current_price - self.state.entry_price) / self.state.entry_price
            unrealized_pnl = self.state.position * price_change * self.initial_capital
            
            # CRITICAL FIX: Enforce stop loss to cap losses
            # Calculate loss percentage (negative price_change for losing position)
            if (self.state.position * price_change) < 0:  # Position is losing
                loss_pct = abs(price_change)
                
                # If loss exceeds stop loss, force close position
                if loss_pct >= self.stop_loss_pct:
                    # Stop loss hit - force close position
                    old_pnl = (current_price - self.state.entry_price) / self.state.entry_price * self.state.position
                    trade_pnl_amount = old_pnl * self.initial_capital
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
                    
                    # Track PnL for expected value calculation
                    self.recent_trades_pnl.append(trade_pnl_amount)
                    if len(self.recent_trades_pnl) > self.recent_trades_window:
                        self.recent_trades_pnl.pop(0)
                    
                    # Close position
                    new_position = 0.0
                    position_change = -self.state.position
                    self.state.entry_price = None
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
        if abs(position_change) > self.action_threshold and self.state.trades_count > 0:
            # Calculate average win and loss from recent trades
            avg_win = self.state.total_win_pnl / max(1, self.state.winning_trades) if self.state.winning_trades > 0 else 0.0
            avg_loss = self.state.total_loss_pnl / max(1, self.state.losing_trades) if self.state.losing_trades > 0 else 0.0
            
            # If we have enough data, check risk/reward ratio
            if avg_loss > 0 and avg_win > 0:
                risk_reward_ratio = avg_win / avg_loss
                
                # Reject trades with poor risk/reward ratio
                if risk_reward_ratio < self.min_risk_reward_ratio:
                    # Risk/reward ratio too poor - reject trade
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
                position_change = 0.0
                new_position = self.state.position
        
        # Store position change for commission calculation
        self.last_position_change = position_change
        
        # Realize PnL if position closed or reversed
        # INCREASED threshold from 0.001 to 0.05 (5%) to reduce overtrading and focus on quality trades
        if abs(position_change) > self.action_threshold:  # Significant position change (configurable threshold)
            # Trading is allowed - proceed with trade
            if self.state.position != 0 and new_position * self.state.position < 0:
                # Position reversed - realize old position
                if self.state.entry_price is not None:
                    old_pnl = (current_price - self.state.entry_price) / self.state.entry_price * self.state.position
                    trade_pnl_amount = old_pnl * self.initial_capital
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
                    # Track PnL for expected value calculation
                    self.recent_trades_pnl.append(trade_pnl_amount)
                    if len(self.recent_trades_pnl) > self.recent_trades_window:
                        self.recent_trades_pnl.pop(0)  # Keep only recent N trades
                self.state.entry_price = current_price if new_position != 0 else None
            elif self.state.position != 0 and abs(new_position) < self.action_threshold:
                # Position closed
                if self.state.entry_price is not None:
                    old_pnl = (current_price - self.state.entry_price) / self.state.entry_price * self.state.position
                    trade_pnl_amount = old_pnl * self.initial_capital
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
                    # Track PnL for expected value calculation
                    self.recent_trades_pnl.append(trade_pnl_amount)
                    if len(self.recent_trades_pnl) > self.recent_trades_window:
                        self.recent_trades_pnl.pop(0)  # Keep only recent N trades
                self.state.entry_price = None
            elif self.state.entry_price is None and abs(new_position) > self.action_threshold:
                # New position opened
                self.state.entry_price = current_price
                self.episode_trades += 1
        
        # Calculate commission cost for this trade
        commission_cost = self._calculate_commission_cost(position_change)
        self.total_commission_cost += commission_cost
        
        # Subtract commission from realized PnL (net profit)
        if abs(position_change) > self.action_threshold:
            self.state.realized_pnl -= commission_cost
        
        # Update state
        self.state.position = new_position
        self.state.unrealized_pnl = unrealized_pnl
        self.state.total_pnl = self.state.realized_pnl + self.state.unrealized_pnl
        
        # Calculate reward (using net profit after commission)
        reward = self._calculate_reward(prev_pnl, self.state.total_pnl, commission_cost)
        
        # Update equity curve
        current_equity = self.initial_capital + self.state.total_pnl
        self.equity_curve.append(current_equity)
        
        # Next step
        self.current_step += 1
        
        # Check if done
        # Note: current_step is 0-indexed, so if max_steps=10000, valid steps are 0-9999
        # After step 9999, current_step will be 10000, which should trigger termination
        terminated = self.current_step >= self.max_steps
        truncated = False  # Can add early stopping logic here
        
        # Get next state
        if not terminated:
            # Ensure current_step is within data bounds before getting state features
            primary_data = self.data[min(self.timeframes)]
            safe_step = min(self.current_step, len(primary_data) - 1)
            if safe_step < 0:
                safe_step = 0
            
            # CRITICAL FIX: Check if we've exceeded data bounds (shouldn't happen, but safety check)
            if safe_step >= len(primary_data) - self.lookback_bars:
                # We're too close to the end of data - terminate episode early
                print(f"[WARNING] Episode terminating early: current_step={self.current_step}, safe_step={safe_step}, data_len={len(primary_data)}, lookback_bars={self.lookback_bars}", flush=True)
                terminated = True
                next_state = np.zeros(self.state_dim, dtype=np.float32)
            else:
                try:
                    next_state = self._get_state_features(safe_step)
                except (IndexError, KeyError) as e:
                    # CRITICAL FIX: Catch exceptions in state feature extraction and terminate gracefully
                    print(f"[ERROR] Exception in _get_state_features at step {self.current_step}: {e}", flush=True)
                    terminated = True
                    next_state = np.zeros(self.state_dim, dtype=np.float32)
        else:
            next_state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Calculate average win/loss for monitoring
        avg_win = self.state.total_win_pnl / max(1, self.state.winning_trades) if self.state.winning_trades > 0 else 0.0
        avg_loss = self.state.total_loss_pnl / max(1, self.state.losing_trades) if self.state.losing_trades > 0 else 0.0
        risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        
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
            "risk_reward_ratio": risk_reward_ratio  # Risk/reward ratio (avg_win / avg_loss)
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

