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
        transaction_cost: float = 0.0001,
        lookback_bars: int = 20,
        reward_config: Optional[Dict] = None,
        max_episode_steps: Optional[int] = None  # Optional limit on episode length
    ):
        super().__init__()
        
        self.data = data
        self.timeframes = sorted(timeframes)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.lookback_bars = lookback_bars
        
        # Reward configuration
        self.reward_config = reward_config or {
            "pnl_weight": 1.0,
            "risk_penalty": 0.5,
            "drawdown_penalty": 0.3
        }
        
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
            avg_volume = full_data["volume"].iloc[current_idx-20:current_idx].mean()
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
    
    def _calculate_reward(self, prev_pnl: float, current_pnl: float) -> float:
        """Calculate reward based on PnL and risk"""
        # PnL change (normalized by initial capital)
        pnl_change = (current_pnl - prev_pnl) / self.initial_capital
        
        # Risk penalty (drawdown) - but only penalize significant drawdowns
        current_equity = self.initial_capital + current_pnl
        if current_equity > self.max_equity:
            self.max_equity = current_equity
        
        drawdown = (self.max_equity - current_equity) / self.max_equity if self.max_equity > 0 else 0.0
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        # Reward components - focus more on PnL, less on penalties
        # Reduce penalty weights to allow positive rewards when PnL is positive
        reward = (
            self.reward_config["pnl_weight"] * pnl_change
            - self.reward_config["risk_penalty"] * 0.1 * drawdown  # Reduced drawdown penalty
            - self.reward_config["drawdown_penalty"] * 0.1 * max(0, self.max_drawdown - 0.15)  # Only penalize if DD > 15%
        )
        
        # Transaction cost - only apply a minimal holding cost
        # Don't penalize every step heavily - this was causing consistent negative rewards
        if self.state and abs(self.state.position) > 0.01:
            # Very small holding cost (0.1% of transaction cost per step)
            holding_cost = self.transaction_cost * 0.001
            reward -= holding_cost
        
        # Small bonus for positive PnL change to encourage profitable moves
        if pnl_change > 0:
            reward += abs(pnl_change) * 0.1  # Small bonus multiplier for profits
        
        # Scale reward moderately (not 100x!) to help with learning
        # But keep it reasonable - scale by 10 instead of 100
        reward *= 10.0
        
        return reward
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_bars
        # Reset max_equity and drawdown tracking for new episode
        self.max_equity = self.initial_capital
        self.max_drawdown = 0.0
        # Reset equity curve
        self.equity_curve = [self.initial_capital]
        
        # Debug: Log max_steps on reset (only first few resets to avoid spam)
        if not hasattr(self, '_reset_count'):
            self._reset_count = 0
        self._reset_count += 1
        if self._reset_count <= 3:
            import sys
            print(f"[DEBUG] Reset #{self._reset_count}: current_step={self.current_step}, max_steps={self.max_steps}, data_max_steps={getattr(self, 'data_max_steps', 'N/A')}, lookback_bars={self.lookback_bars}", flush=True)
            sys.stdout.flush()
        
        self.state = TradeState(
            position=0.0,
            entry_price=None,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            total_pnl=0.0,
            trades_count=0,
            winning_trades=0,
            losing_trades=0
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
        
        prev_pnl = self.state.total_pnl if self.state else 0.0
        
        # Get current price
        current_price = self.data[min(self.timeframes)].iloc[self.current_step]["close"]
        
        # Update position
        position_change = action_value - self.state.position
        new_position = action_value
        
        # Calculate PnL
        if self.state.entry_price is not None:
            # Unrealized PnL for current position
            price_change = (current_price - self.state.entry_price) / self.state.entry_price
            unrealized_pnl = self.state.position * price_change * self.initial_capital
        else:
            unrealized_pnl = 0.0
        
        # Realize PnL if position closed or reversed
        if abs(position_change) > 0.01:  # Significant position change
            if self.state.position != 0 and new_position * self.state.position < 0:
                # Position reversed - realize old position
                if self.state.entry_price is not None:
                    old_pnl = (current_price - self.state.entry_price) / self.state.entry_price * self.state.position
                    self.state.realized_pnl += old_pnl * self.initial_capital
                    if old_pnl > 0:
                        self.state.winning_trades += 1
                    else:
                        self.state.losing_trades += 1
                    self.state.trades_count += 1
                self.state.entry_price = current_price if new_position != 0 else None
            elif self.state.position != 0 and abs(new_position) < 0.01:
                # Position closed
                if self.state.entry_price is not None:
                    old_pnl = (current_price - self.state.entry_price) / self.state.entry_price * self.state.position
                    self.state.realized_pnl += old_pnl * self.initial_capital
                    if old_pnl > 0:
                        self.state.winning_trades += 1
                    else:
                        self.state.losing_trades += 1
                    self.state.trades_count += 1
                self.state.entry_price = None
            elif self.state.entry_price is None and abs(new_position) > 0.01:
                # New position opened
                self.state.entry_price = current_price
        
        # Update state
        self.state.position = new_position
        self.state.unrealized_pnl = unrealized_pnl
        self.state.total_pnl = self.state.realized_pnl + self.state.unrealized_pnl
        
        # Calculate reward
        reward = self._calculate_reward(prev_pnl, self.state.total_pnl)
        
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
        
        # Debug logging for long episodes
        if self.current_step >= self.max_steps - 5 or self.current_step >= 9995:
            import sys
            print(f"[DEBUG] TradingEnvironment: current_step={self.current_step}, max_steps={self.max_steps}, terminated={terminated}, lookback_bars={self.lookback_bars}", flush=True)
            sys.stdout.flush()
        
        # Get next state
        if not terminated:
            next_state = self._get_state_features(self.current_step)
        else:
            next_state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Info
        info = {
            "step": self.current_step,
            "position": self.state.position,
            "pnl": self.state.total_pnl,
            "trades": self.state.trades_count,
            "win_rate": self.state.winning_trades / max(1, self.state.trades_count),
            "equity": current_equity,
            "max_drawdown": self.max_drawdown
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

