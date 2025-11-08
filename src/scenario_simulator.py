"""
Scenario Simulation Framework

Provides robustness testing, stress testing, and parameter sensitivity analysis
across different market regimes and conditions.

Veteran Futures Trader Approach:
- Test strategy across multiple market scenarios
- Stress test under extreme conditions
- Analyze parameter sensitivity
- Identify optimal parameter ranges
- Validate strategy robustness
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import warnings
import os
import yaml
from pathlib import Path
warnings.filterwarnings('ignore')


class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    GAP_EVENT = "gap_event"
    LOW_LIQUIDITY = "low_liquidity"
    CRASH = "crash"
    FLASH_CRASH = "flash_crash"
    NORMAL = "normal"


@dataclass
class ScenarioResult:
    """Result from a single scenario simulation"""
    scenario_name: str
    market_regime: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    total_pnl: float
    volatility: float
    calmar_ratio: float
    parameters: Dict[str, Any]
    equity_curve: List[float]
    timestamp: datetime


@dataclass
class StressTestResult:
    """Result from stress testing"""
    scenario_name: str
    max_drawdown: float
    recovery_time: int  # Periods to recover
    worst_case_loss: float
    survived: bool
    equity_at_min: float
    details: Dict[str, Any]


@dataclass
class ParameterSensitivityResult:
    """Result from parameter sensitivity analysis"""
    parameter_name: str
    parameter_values: List[float]
    performance_metrics: Dict[str, List[float]]
    optimal_value: float
    sensitivity_score: float  # 0-1, higher = more sensitive
    recommendations: List[str]


class ScenarioSimulator:
    """
    Scenario Simulator for robustness testing.
    
    Simulates trading strategies across different market conditions
    to assess robustness and identify weaknesses.
    """
    
    def __init__(
        self,
        base_price_data: pd.DataFrame,
        initial_capital: float = 100000.0
    ):
        """
        Initialize scenario simulator.
        
        Args:
            base_price_data: Base historical price data
            initial_capital: Starting capital
        """
        self.base_price_data = base_price_data.copy()
        self.initial_capital = initial_capital
    
    def apply_market_regime(
        self,
        price_data: pd.DataFrame,
        regime: MarketRegime,
        intensity: float = 1.0
    ) -> pd.DataFrame:
        """
        Apply market regime transformation to price data.
        
        Args:
            price_data: Original price data
            regime: Market regime to simulate
            intensity: Intensity of the regime (0.5 = mild, 1.0 = normal, 2.0 = extreme)
        
        Returns:
            Transformed price data
        """
        data = price_data.copy()
        
        if regime == MarketRegime.TRENDING_UP:
            # Add upward trend
            trend_strength = 0.001 * intensity  # 0.1% per period
            trend = np.arange(len(data)) * trend_strength
            data['close'] = data['close'] * (1 + trend)
            data['high'] = data['high'] * (1 + trend * 1.1)
            data['low'] = data['low'] * (1 + trend * 0.9)
            data['open'] = data['open'] * (1 + trend)
            
        elif regime == MarketRegime.TRENDING_DOWN:
            # Add downward trend
            trend_strength = -0.001 * intensity
            trend = np.arange(len(data)) * trend_strength
            data['close'] = data['close'] * (1 + trend)
            data['high'] = data['high'] * (1 + trend * 0.9)
            data['low'] = data['low'] * (1 + trend * 1.1)
            data['open'] = data['open'] * (1 + trend)
            
        elif regime == MarketRegime.RANGING:
            # Add mean-reverting behavior (reduce trend, increase noise)
            returns = data['close'].pct_change().fillna(0)
            # Mean-revert: reduce trend component
            mean_reversion_strength = 0.3 * intensity
            reverted_returns = returns * (1 - mean_reversion_strength)
            data['close'] = data['close'].iloc[0] * (1 + reverted_returns).cumprod()
            # Update other columns proportionally
            price_ratio = data['close'] / price_data['close']
            data['high'] = data['high'] * price_ratio
            data['low'] = data['low'] * price_ratio
            data['open'] = data['open'] * price_ratio
            
        elif regime == MarketRegime.HIGH_VOLATILITY:
            # Increase volatility
            returns = data['close'].pct_change().fillna(0)
            volatility_multiplier = 1.0 + (intensity - 1.0) * 0.5  # Up to 2x volatility
            noisy_returns = returns * volatility_multiplier + np.random.randn(len(returns)) * 0.002 * intensity
            data['close'] = data['close'].iloc[0] * (1 + noisy_returns).cumprod()
            # Widen high/low ranges
            range_multiplier = 1.0 + (intensity - 1.0) * 0.3
            data['high'] = data['close'] * (1 + abs(np.random.randn(len(data)) * 0.01 * range_multiplier))
            data['low'] = data['close'] * (1 - abs(np.random.randn(len(data)) * 0.01 * range_multiplier))
            
        elif regime == MarketRegime.LOW_VOLATILITY:
            # Decrease volatility
            returns = data['close'].pct_change().fillna(0)
            volatility_multiplier = max(0.3, 1.0 - (intensity - 1.0) * 0.3)  # Down to 30% volatility
            smoothed_returns = returns * volatility_multiplier
            data['close'] = data['close'].iloc[0] * (1 + smoothed_returns).cumprod()
            # Tighten high/low ranges
            data['high'] = data['close'] * 1.002
            data['low'] = data['close'] * 0.998
            
        elif regime == MarketRegime.GAP_EVENT:
            # Add significant gaps
            gap_periods = [len(data) // 3, len(data) * 2 // 3]  # Gaps at 1/3 and 2/3
            gap_size = 0.02 * intensity  # 2% gap per intensity unit
            for gap_idx in gap_periods:
                if gap_idx < len(data):
                    # Upward gap
                    if gap_idx % 2 == 0:
                        data.loc[gap_idx:, 'close'] = data.loc[gap_idx:, 'close'] * (1 + gap_size)
                        data.loc[gap_idx, 'open'] = data.loc[gap_idx, 'close'] * (1 + gap_size * 0.5)
                    # Downward gap
                    else:
                        data.loc[gap_idx:, 'close'] = data.loc[gap_idx:, 'close'] * (1 - gap_size)
                        data.loc[gap_idx, 'open'] = data.loc[gap_idx, 'close'] * (1 - gap_size * 0.5)
            
        elif regime == MarketRegime.LOW_LIQUIDITY:
            # Reduce volume and increase spreads
            volume_multiplier = max(0.1, 1.0 - (intensity - 1.0) * 0.4)
            data['volume'] = data['volume'] * volume_multiplier
            # Widen spreads (high-low range)
            spread_multiplier = 1.0 + (intensity - 1.0) * 0.5
            spread = (data['high'] - data['low']) * spread_multiplier
            mid_price = (data['high'] + data['low']) / 2
            data['high'] = mid_price + spread / 2
            data['low'] = mid_price - spread / 2
            
        elif regime == MarketRegime.CRASH:
            # Simulate market crash (sudden drop, then recovery attempt)
            crash_period = len(data) // 2
            crash_size = 0.15 * intensity  # 15% crash per intensity unit
            # Crash
            data.loc[crash_period:, 'close'] = data.loc[crash_period:, 'close'] * (1 - crash_size)
            # Gradual recovery
            recovery_periods = min(20, len(data) - crash_period)
            recovery = np.linspace(0, crash_size * 0.5, recovery_periods)  # Recover 50% of crash
            data.loc[crash_period:crash_period+recovery_periods-1, 'close'] *= (1 + recovery)
            
        elif regime == MarketRegime.FLASH_CRASH:
            # Simulate flash crash (sudden drop, quick recovery)
            crash_period = len(data) // 2
            crash_size = 0.10 * intensity  # 10% flash crash
            recovery_size = crash_size * 0.8  # Recover 80%
            # Crash
            data.loc[crash_period, 'close'] = data.loc[crash_period, 'close'] * (1 - crash_size)
            # Quick recovery (next few periods)
            recovery_periods = min(5, len(data) - crash_period - 1)
            recovery_step = recovery_size / recovery_periods
            for i in range(1, recovery_periods + 1):
                if crash_period + i < len(data):
                    data.loc[crash_period + i, 'close'] = data.loc[crash_period + i, 'close'] * (1 + recovery_step)
        
        # Normal regime - no transformation
        # elif regime == MarketRegime.NORMAL:
        #     pass
        
        # Ensure data integrity
        data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
        data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
        data['volume'] = data['volume'].abs()  # Ensure non-negative
        
        return data
    
    def simulate_scenario(
        self,
        scenario_name: str,
        regime: MarketRegime,
        price_data: Optional[pd.DataFrame] = None,
        intensity: float = 1.0,
        backtest_func: Optional[callable] = None,
        backtest_params: Optional[Dict] = None
    ) -> ScenarioResult:
        """
        Simulate a trading scenario.
        
        Args:
            scenario_name: Name of the scenario
            regime: Market regime to simulate
            price_data: Price data (uses base if None)
            intensity: Intensity of the regime
            backtest_func: Backtest function to use
            backtest_params: Parameters for backtest function
        
        Returns:
            ScenarioResult with performance metrics
        """
        # Use provided data or base data
        if price_data is None:
            price_data = self.base_price_data.copy()
        
        # Apply market regime
        transformed_data = self.apply_market_regime(price_data, regime, intensity)
        
        # If backtest function provided, run it
        if backtest_func:
            # Run backtest with transformed data
            results = backtest_func(transformed_data, **(backtest_params or {}))
        else:
            # Default: simple simulation
            results = self._simple_backtest(transformed_data)
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        
        return ScenarioResult(
            scenario_name=scenario_name,
            market_regime=regime.value,
            total_return=metrics['total_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor'],
            total_trades=metrics['total_trades'],
            winning_trades=metrics['winning_trades'],
            losing_trades=metrics['losing_trades'],
            avg_win=metrics['avg_win'],
            avg_loss=metrics['avg_loss'],
            largest_win=metrics['largest_win'],
            largest_loss=metrics['largest_loss'],
            total_pnl=metrics['total_pnl'],
            volatility=metrics['volatility'],
            calmar_ratio=metrics['calmar_ratio'],
            parameters=backtest_params or {},
            equity_curve=results.get('equity_curve', []),
            timestamp=datetime.now()
        )
    
    def _simple_backtest(self, price_data: pd.DataFrame) -> Dict:
        """Simple backtest simulation (placeholder - should use actual backtest)"""
        # This is a placeholder - in production, would call actual backtest system
        returns = price_data['close'].pct_change().fillna(0)
        equity_curve = self.initial_capital * (1 + returns).cumprod()
        
        # Simulate some trades
        trades = []
        for i in range(10, len(price_data), 20):
            entry_price = price_data['close'].iloc[i]
            exit_price = price_data['close'].iloc[min(i+10, len(price_data)-1)]
            pnl = (exit_price - entry_price) / entry_price
            trades.append(pnl)
        
        return {
            'equity_curve': equity_curve.tolist(),
            'trades': trades,
            'final_equity': equity_curve.iloc[-1]
        }
    
    @staticmethod
    def _dataframe_to_multi_timeframe(
        price_data: pd.DataFrame,
        timeframes: List[int] = [1, 5, 15]
    ) -> Dict[int, pd.DataFrame]:
        """
        Convert a single DataFrame to multi-timeframe format.
        
        Args:
            price_data: Single DataFrame with OHLCV data
            timeframes: List of timeframes in minutes
        
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        # Ensure we have a timestamp column
        if 'timestamp' not in price_data.columns:
            if 'datetime' in price_data.columns:
                price_data = price_data.copy()
                price_data['timestamp'] = pd.to_datetime(price_data['datetime'])
            else:
                # Create synthetic timestamps
                price_data = price_data.copy()
                price_data['timestamp'] = pd.date_range(
                    end=datetime.now(),
                    periods=len(price_data),
                    freq='1min'
                )
        else:
            price_data = price_data.copy()
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        
        # Set timestamp as index for resampling
        price_data_indexed = price_data.set_index('timestamp')
        
        # Create multi-timeframe data
        multi_tf_data = {}
        for tf in sorted(timeframes):
            if tf == 1:
                # Use original data for 1-minute
                multi_tf_data[tf] = price_data.copy()
            else:
                # Resample to higher timeframes
                tf_data = price_data_indexed.resample(f'{tf}min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna().reset_index()
                multi_tf_data[tf] = tf_data
        
        return multi_tf_data
    
    @staticmethod
    def create_rl_agent_backtest_func(
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        n_episodes: int = 1
    ) -> Callable:
        """
        Create an RL agent backtest function for use in scenario simulation.
        
        Args:
            model_path: Path to trained model (if None, uses default location)
            config_path: Path to config file (if None, uses default)
            n_episodes: Number of episodes to run per scenario
        
        Returns:
            Backtest function that can be passed to simulate_scenario
        """
        def rl_agent_backtest(
            price_data: pd.DataFrame,
            timeframes: List[int] = [1, 5, 15],
            initial_capital: float = 100000.0,
            transaction_cost: float = 0.0001
        ) -> Dict:
            """
            Run RL agent backtest on price data.
            
            Args:
                price_data: Price data DataFrame
                timeframes: Timeframes to use
                initial_capital: Initial capital
                transaction_cost: Transaction cost ratio
            
            Returns:
                Dictionary with backtest results
            """
            try:
                # Import here to avoid circular dependencies
                import torch
                from src.trading_env import TradingEnvironment
                from src.rl_agent import PPOAgent
                
                # Convert to multi-timeframe format
                multi_tf_data = ScenarioSimulator._dataframe_to_multi_timeframe(
                    price_data, timeframes
                )
                
                # Load config
                if config_path and Path(config_path).exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                else:
                    # Use default config
                    default_config_path = Path("configs/train_config.yaml")
                    if default_config_path.exists():
                        with open(default_config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    else:
                        # Create minimal config
                        config = {
                            "environment": {
                                "timeframes": timeframes,
                                "action_range": [-1.0, 1.0],
                                "reward": {
                                    "pnl_weight": 1.0,
                                    "risk_penalty": 0.5,
                                    "drawdown_penalty": 0.3
                                }
                            },
                            "risk_management": {
                                "initial_capital": initial_capital,
                                "commission": transaction_cost * initial_capital
                            }
                        }
                
                # Determine model path
                if model_path and Path(model_path).exists():
                    actual_model_path = model_path
                else:
                    # Try to find best model
                    models_dir = Path("models")
                    best_model = models_dir / "best_model.pt"
                    if best_model.exists():
                        actual_model_path = str(best_model)
                    else:
                        # Look for any .pt file in models directory
                        model_files = list(models_dir.glob("*.pt"))
                        if model_files:
                            actual_model_path = str(model_files[0])
                        else:
                            # No model available - return empty results
                            return {
                                'equity_curve': [initial_capital],
                                'trades': [],
                                'final_equity': initial_capital,
                                'pnl': 0.0,
                                'trades': 0,
                                'win_rate': 0.0,
                                'max_drawdown': 0.0
                            }
                
                # Create environment
                env = TradingEnvironment(
                    data=multi_tf_data,
                    timeframes=timeframes,
                    initial_capital=initial_capital,
                    transaction_cost=transaction_cost,
                    reward_config=config.get("environment", {}).get("reward", {})
                )
                
                # Load agent - try to detect architecture from checkpoint
                action_range = config.get("environment", {}).get("action_range", [-1.0, 1.0])
                
                # Try to load checkpoint to detect architecture
                try:
                    checkpoint = torch.load(actual_model_path, map_location='cpu', weights_only=False)
                    saved_hidden_dims = checkpoint.get('hidden_dims', None)
                    saved_state_dim = checkpoint.get('state_dim', None)
                    
                    if saved_hidden_dims:
                        print(f"📋 Detected saved model architecture: hidden_dims={saved_hidden_dims}")
                        # Use saved architecture
                        agent = PPOAgent(
                            state_dim=env.state_dim,
                            action_range=tuple(action_range),
                            device="cpu",
                            hidden_dims=saved_hidden_dims  # Use saved architecture
                        )
                    else:
                        # No architecture saved, use default
                        agent = PPOAgent(
                            state_dim=env.state_dim,
                            action_range=tuple(action_range),
                            device="cpu"  # Use CPU for backtesting
                        )
                except Exception as e:
                    # If we can't read checkpoint, use default architecture
                    print(f"⚠️  Could not read checkpoint architecture: {e}. Using default.")
                    agent = PPOAgent(
                        state_dim=env.state_dim,
                        action_range=tuple(action_range),
                        device="cpu"  # Use CPU for backtesting
                    )
                
                agent.load(actual_model_path)
                agent.actor.eval()
                agent.critic.eval()
                
                # Run backtest episodes
                all_trades = []
                all_pnls = []
                all_win_rates = []
                all_drawdowns = []
                all_equity_curves = []
                
                for episode in range(n_episodes):
                    state, info = env.reset()
                    done = False
                    episode_trades = []
                    
                    while not done:
                        # Select action (deterministic for backtesting)
                        action, _, _ = agent.select_action(state, deterministic=True)
                        
                        # Step environment
                        state, reward, terminated, truncated, step_info = env.step(action)
                        done = terminated or truncated
                        
                        # Track trades if any
                        if step_info.get("trades", 0) > len(episode_trades):
                            episode_trades.append(step_info)
                    
                    # Episode complete - collect metrics
                    final_info = step_info
                    all_pnls.append(final_info.get("pnl", 0))
                    all_trades.append(final_info.get("trades", 0))
                    all_win_rates.append(final_info.get("win_rate", 0))
                    all_drawdowns.append(final_info.get("max_drawdown", 0))
                    all_equity_curves.append(env.equity_curve.copy())
                
                # Aggregate results across episodes
                if len(all_equity_curves) > 0:
                    # Use average equity curve
                    equity_array = np.array(all_equity_curves)
                    avg_equity_curve = equity_array.mean(axis=0).tolist()
                else:
                    avg_equity_curve = [initial_capital]
                
                # Calculate trade list (aggregate across episodes)
                total_trades_count = sum(all_trades) if all_trades else 0
                
                return {
                    'equity_curve': avg_equity_curve,
                    'trades': list(range(total_trades_count)),  # Placeholder trade list
                    'final_equity': avg_equity_curve[-1] if avg_equity_curve else initial_capital,
                    'pnl': np.mean(all_pnls) if all_pnls else 0.0,
                    'trades': total_trades_count,
                    'win_rate': np.mean(all_win_rates) if all_win_rates else 0.0,
                    'max_drawdown': np.min(all_drawdowns) if all_drawdowns else 0.0
                }
                
            except Exception as e:
                # If RL agent backtest fails, fall back to simple backtest
                import warnings
                warnings.warn(f"RL agent backtest failed: {e}. Falling back to simple backtest.")
                
                # Fallback to simple backtest
                returns = price_data['close'].pct_change().fillna(0)
                equity_curve = initial_capital * (1 + returns).cumprod()
                
                trades = []
                for i in range(10, len(price_data), 20):
                    entry_price = price_data['close'].iloc[i]
                    exit_price = price_data['close'].iloc[min(i+10, len(price_data)-1)]
                    pnl = (exit_price - entry_price) / entry_price
                    trades.append(pnl)
                
                return {
                    'equity_curve': equity_curve.tolist(),
                    'trades': trades,
                    'final_equity': equity_curve.iloc[-1],
                    'pnl': equity_curve.iloc[-1] - initial_capital,
                    'trades': len(trades),
                    'win_rate': len([t for t in trades if t > 0]) / len(trades) if trades else 0.0,
                    'max_drawdown': 0.0  # Would need to calculate properly
                }
        
        return rl_agent_backtest
    
    def _calculate_metrics(self, results: Dict) -> Dict:
        """Calculate performance metrics from backtest results"""
        equity_curve = np.array(results.get('equity_curve', [self.initial_capital]))
        trades = results.get('trades', [])
        
        # Check if results already contain metrics from RL agent backtest
        if 'win_rate' in results and 'max_drawdown' in results:
            # Use metrics directly from RL agent backtest if available
            total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] if len(equity_curve) > 0 else 0.0
            win_rate = results.get('win_rate', 0.0)
            max_drawdown = results.get('max_drawdown', 0.0)
            total_trades_count = results.get('trades', 0)
            
            # Calculate additional metrics from equity curve
            running_max = np.maximum.accumulate(equity_curve)
            drawdowns = (equity_curve - running_max) / running_max
            calculated_max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
            
            # Use the more negative drawdown
            max_drawdown = min(max_drawdown, calculated_max_drawdown) if max_drawdown != 0 else calculated_max_drawdown
            
            # Sharpe ratio
            returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else [0]
            sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 0 and np.std(returns) > 0 else 0.0
            
            # Estimate trade PnL distribution from equity curve changes
            if len(equity_curve) > 1:
                equity_changes = np.diff(equity_curve)
                winning_trades_count = len([x for x in equity_changes if x > 0])
                losing_trades_count = len([x for x in equity_changes if x < 0])
                
                if winning_trades_count > 0:
                    avg_win = float(np.mean([x for x in equity_changes if x > 0]))
                    largest_win = float(np.max([x for x in equity_changes if x > 0]))
                else:
                    avg_win = 0.0
                    largest_win = 0.0
                
                if losing_trades_count > 0:
                    avg_loss = float(np.mean([x for x in equity_changes if x < 0]))
                    largest_loss = float(np.min([x for x in equity_changes if x < 0]))
                else:
                    avg_loss = 0.0
                    largest_loss = 0.0
                
                # Profit factor
                gross_profit = sum([x for x in equity_changes if x > 0]) if winning_trades_count > 0 else 0.0
                gross_loss = abs(sum([x for x in equity_changes if x < 0])) if losing_trades_count > 0 else 1.0
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
            else:
                avg_win = 0.0
                avg_loss = 0.0
                largest_win = 0.0
                largest_loss = 0.0
                profit_factor = 0.0
                winning_trades_count = 0
                losing_trades_count = 0
        else:
            # Calculate metrics from trade list (simple backtest format)
            # Basic metrics
            total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] if len(equity_curve) > 0 else 0.0
            
            # Drawdown
            running_max = np.maximum.accumulate(equity_curve)
            drawdowns = (equity_curve - running_max) / running_max
            max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
            
            # Sharpe ratio (simplified)
            returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else [0]
            sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 0 and np.std(returns) > 0 else 0.0
            
            # Trade metrics
            if len(trades) > 0 and isinstance(trades[0], (int, float)):
                # Trades is a list of PnL values
                winning_trades = [t for t in trades if t > 0]
                losing_trades = [t for t in trades if t < 0]
                win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0.0
                avg_win = float(np.mean(winning_trades)) if len(winning_trades) > 0 else 0.0
                avg_loss = float(np.mean(losing_trades)) if len(losing_trades) > 0 else 0.0
                largest_win = float(np.max(trades)) if len(trades) > 0 else 0.0
                largest_loss = float(np.min(trades)) if len(trades) > 0 else 0.0
                
                # Profit factor
                gross_profit = sum(winning_trades) if len(winning_trades) > 0 else 0.0
                gross_loss = abs(sum(losing_trades)) if len(losing_trades) > 0 else 1.0
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
                
                total_trades_count = len(trades)
                winning_trades_count = len(winning_trades)
                losing_trades_count = len(losing_trades)
            else:
                # No valid trades
                win_rate = 0.0
                avg_win = 0.0
                avg_loss = 0.0
                largest_win = 0.0
                largest_loss = 0.0
                profit_factor = 0.0
                total_trades_count = 0
                winning_trades_count = 0
                losing_trades_count = 0
        
        # Volatility
        returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else [0]
        volatility = float(np.std(returns) * np.sqrt(252)) if len(returns) > 0 else 0.0
        
        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'total_trades': int(total_trades_count),
            'winning_trades': int(winning_trades_count),
            'losing_trades': int(losing_trades_count),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'largest_win': float(largest_win),
            'largest_loss': float(largest_loss),
            'total_pnl': float(equity_curve[-1] - equity_curve[0]) if len(equity_curve) > 0 else 0.0,
            'volatility': float(volatility),
            'calmar_ratio': float(calmar_ratio)
        }
    
    def run_stress_test(
        self,
        scenarios: List[Tuple[str, MarketRegime, float]],
        backtest_func: Optional[callable] = None,
        backtest_params: Optional[Dict] = None
    ) -> List[StressTestResult]:
        """
        Run stress tests across multiple scenarios.
        
        Args:
            scenarios: List of (name, regime, intensity) tuples
            backtest_func: Backtest function
            backtest_params: Backtest parameters
        
        Returns:
            List of StressTestResult
        """
        stress_results = []
        
        for scenario_name, regime, intensity in scenarios:
            scenario_result = self.simulate_scenario(
                scenario_name=scenario_name,
                regime=regime,
                intensity=intensity,
                backtest_func=backtest_func,
                backtest_params=backtest_params
            )
            
            # Determine if strategy survived
            survived = scenario_result.max_drawdown > -0.50  # Survived if drawdown < 50%
            
            # Calculate recovery time (simplified)
            equity_curve = scenario_result.equity_curve
            # Ensure equity_curve is a list
            if not isinstance(equity_curve, list):
                equity_curve = list(equity_curve) if hasattr(equity_curve, '__iter__') else [self.initial_capital]
            
            if len(equity_curve) > 0:
                initial_equity = equity_curve[0]
                min_equity = min(equity_curve)
                min_idx = equity_curve.index(min_equity)
                
                # Find recovery (return to initial equity)
                recovery_time = len(equity_curve) - min_idx
                for i in range(min_idx, len(equity_curve)):
                    if equity_curve[i] >= initial_equity:
                        recovery_time = i - min_idx
                        break
            else:
                recovery_time = 0
                equity_curve = [self.initial_capital]
            
            stress_result = StressTestResult(
                scenario_name=scenario_name,
                max_drawdown=scenario_result.max_drawdown,
                recovery_time=recovery_time,
                worst_case_loss=scenario_result.largest_loss,
                survived=survived,
                equity_at_min=min(equity_curve) if len(equity_curve) > 0 else self.initial_capital,
                details={
                    'total_return': scenario_result.total_return,
                    'sharpe_ratio': scenario_result.sharpe_ratio,
                    'win_rate': scenario_result.win_rate
                }
            )
            
            stress_results.append(stress_result)
        
        return stress_results
    
    def parameter_sensitivity_analysis(
        self,
        parameter_name: str,
        parameter_values: List[float],
        backtest_func: callable,
        base_params: Dict,
        regime: MarketRegime = MarketRegime.NORMAL
    ) -> ParameterSensitivityResult:
        """
        Analyze sensitivity to a parameter.
        
        Args:
            parameter_name: Name of parameter to analyze
            parameter_values: List of values to test
            backtest_func: Backtest function
            base_params: Base parameters (parameter_name will be varied)
            regime: Market regime to use
        
        Returns:
            ParameterSensitivityResult
        """
        results = []
        
        for param_value in parameter_values:
            # Create params with this value
            test_params = base_params.copy()
            test_params[parameter_name] = param_value
            
            # Run scenario
            scenario_result = self.simulate_scenario(
                scenario_name=f"{parameter_name}_{param_value}",
                regime=regime,
                intensity=1.0,
                backtest_func=backtest_func,
                backtest_params=test_params
            )
            
            results.append(scenario_result)
        
        # Extract performance metrics
        metrics = {
            'total_return': [r.total_return for r in results],
            'sharpe_ratio': [r.sharpe_ratio for r in results],
            'max_drawdown': [r.max_drawdown for r in results],
            'win_rate': [r.win_rate for r in results],
            'profit_factor': [r.profit_factor for r in results]
        }
        
        # Find optimal value (highest Sharpe ratio)
        sharpe_ratios = metrics['sharpe_ratio']
        if len(sharpe_ratios) > 0:
            optimal_idx = np.argmax(sharpe_ratios)
            optimal_value = parameter_values[optimal_idx]
        else:
            optimal_value = parameter_values[0] if len(parameter_values) > 0 else 0.0
        
        # Calculate sensitivity score (coefficient of variation)
        if len(sharpe_ratios) > 0 and np.mean(sharpe_ratios) != 0:
            sensitivity_score = min(1.0, abs(np.std(sharpe_ratios) / np.mean(sharpe_ratios)))
        else:
            sensitivity_score = 0.0
        
        # Generate recommendations
        recommendations = []
        if sensitivity_score > 0.5:
            recommendations.append(f"High sensitivity to {parameter_name} - optimize carefully")
        if optimal_value != parameter_values[len(parameter_values)//2]:
            recommendations.append(f"Optimal {parameter_name} is {optimal_value:.4f}")
        
        return ParameterSensitivityResult(
            parameter_name=parameter_name,
            parameter_values=parameter_values,
            performance_metrics=metrics,
            optimal_value=optimal_value,
            sensitivity_score=sensitivity_score,
            recommendations=recommendations
        )


def run_robustness_test(
    price_data: pd.DataFrame,
    scenarios: List[Tuple[str, MarketRegime, float]],
    backtest_func: Optional[callable] = None,
    backtest_params: Optional[Dict] = None
) -> List[ScenarioResult]:
    """
    Convenience function for robustness testing.
    
    Args:
        price_data: Historical price data
        scenarios: List of (name, regime, intensity) tuples
        backtest_func: Backtest function
        backtest_params: Backtest parameters
    
    Returns:
        List of ScenarioResult
    """
    simulator = ScenarioSimulator(price_data)
    results = []
    
    for scenario_name, regime, intensity in scenarios:
        result = simulator.simulate_scenario(
            scenario_name=scenario_name,
            regime=regime,
            intensity=intensity,
            backtest_func=backtest_func,
            backtest_params=backtest_params
        )
        results.append(result)
    
    return results

