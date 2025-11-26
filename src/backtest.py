"""
Backtesting Framework for RL Trading Strategy

Tests the trained agent on historical data and calculates performance metrics.

Usage:
    python src/backtest.py --model models/best_model.pt --data data/processed/test_data.csv
"""

import argparse
import yaml
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from src.data_extraction import DataExtractor
from src.trading_env import TradingEnvironment
from src.rl_agent import PPOAgent
from src.trading_hours import TradingHoursManager
from src.walk_forward import WalkForwardAnalyzer


class Backtester:
    """Handles backtesting of the trained agent"""
    
    def __init__(self, config: dict, model_path: str):
        self.config = config
        self.model_path = model_path
        
        # Load data
        print("Loading test data...")
        self._load_data()
        
        # Create environment
        self.env = TradingEnvironment(
            data=self.multi_tf_data,
            timeframes=config["environment"]["timeframes"],
            initial_capital=config["risk_management"]["initial_capital"],
            transaction_cost=config["risk_management"]["commission"] / config["risk_management"]["initial_capital"],
            reward_config=config["environment"]["reward"]
        )
        
        # Load agent
        print(f"Loading agent from: {model_path}")
        self.agent = PPOAgent(
            state_dim=self.env.state_dim,
            action_range=tuple(config["environment"]["action_range"]),
            device="cpu"  # Backtesting doesn't need GPU
        )
        self.agent.load(model_path)
        
        # Results storage
        self.results = []
        self.equity_curves = []
        
    def _load_data(self):
        """Load test data"""
        extractor = DataExtractor()
        instrument = self.config["environment"]["instrument"]
        timeframes = self.config["environment"]["timeframes"]
        trading_hours_cfg = self.config["environment"].get("trading_hours", {})
        trading_hours_manager = None
        if trading_hours_cfg.get("enabled"):
            trading_hours_manager = TradingHoursManager.from_dict(trading_hours_cfg)
        
        self.multi_tf_data = extractor.load_multi_timeframe_data(
            instrument,
            timeframes,
            trading_hours=trading_hours_manager
        )
    
    def run_backtest(self, n_episodes: int = 10) -> dict:
        """
        Run backtest over multiple episodes.
        
        Args:
            n_episodes: Number of episodes to run
        
        Returns:
            Dictionary with performance metrics
        """
        print(f"\nRunning backtest over {n_episodes} episodes...")
        print("-" * 60)
        
        all_rewards = []
        all_pnls = []
        all_trades = []
        all_win_rates = []
        all_max_drawdowns = []
        
        for episode in range(n_episodes):
            state, info = self.env.reset()
            done = False
            episode_reward = 0
            episode_trades = []
            episode_step = 0
            
            while not done:
                # Select action (deterministic for backtesting)
                action, _, _ = self.agent.select_action(state, deterministic=True)
                raw_action = (
                    float(action[0])
                    if isinstance(action, (list, tuple, np.ndarray))
                    else float(action)
                )
                
                # Step environment
                state, reward, terminated, truncated, step_info = self.env.step(action)
                if isinstance(step_info, dict):
                    step_info["raw_action"] = raw_action
                done = terminated or truncated
                episode_reward += reward
                episode_step += 1
            
                # Track trades
                if step_info.get("trades", 0) > len(episode_trades):
                    episode_trades.append(step_info)
            
            # Episode complete
            final_info = step_info
            all_rewards.append(episode_reward)
            all_pnls.append(final_info.get("pnl", 0))
            all_trades.append(final_info.get("trades", 0))
            all_win_rates.append(final_info.get("win_rate", 0))
            all_max_drawdowns.append(final_info.get("max_drawdown", 0))
            
            # Store equity curve
            self.equity_curves.append(self.env.equity_curve.copy())
            
            print(f"Episode {episode+1}/{n_episodes}: "
                  f"Reward: {episode_reward:.2f}, "
                  f"PnL: ${final_info.get('pnl', 0):.2f}, "
                  f"Trades: {final_info.get('trades', 0)}, "
                  f"Win Rate: {final_info.get('win_rate', 0)*100:.1f}%")
        
        # Calculate aggregate metrics
        results = {
            "n_episodes": n_episodes,
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "mean_pnl": np.mean(all_pnls),
            "std_pnl": np.std(all_pnls),
            "total_pnl": np.sum(all_pnls),
            "mean_trades": np.mean(all_trades),
            "total_trades": int(np.sum(all_trades)),
            "mean_win_rate": np.mean(all_win_rates),
            "mean_max_drawdown": np.mean(all_max_drawdowns),
            "sharpe_ratio": self._calculate_sharpe(all_pnls),
            "sortino_ratio": self._calculate_sortino(all_pnls),
            "profit_factor": self._calculate_profit_factor(all_pnls),
        }
        
        self.results.append(results)
        
        return results
    
    def run_walk_forward(
        self,
        train_window: int = 252,  # 1 year
        test_window: int = 63,     # 3 months
        step_size: int = 21,       # 1 month
        window_type: str = "rolling"
    ) -> dict:
        """
        Run walk-forward analysis to prevent overfitting.
        
        Args:
            train_window: Number of periods for training
            test_window: Number of periods for testing
            step_size: Step size between windows
            window_type: "rolling" or "expanding"
        
        Returns:
            Dictionary with walk-forward results
        """
        from src.walk_forward import WalkForwardAnalyzer
        
        # Get primary timeframe data for walk-forward
        primary_tf = min(self.config["environment"]["timeframes"])
        primary_data = self.multi_tf_data[primary_tf]
        
        # Create walk-forward analyzer
        analyzer = WalkForwardAnalyzer(
            train_window=train_window,
            test_window=test_window,
            step_size=step_size,
            window_type=window_type
        )
        
        # Define training function
        def train_model(train_data: pd.DataFrame) -> str:
            """Train model on training data"""
            # Create temporary config with training data
            # For now, we'll use a simplified approach
            # In production, you'd retrain the model here
            import tempfile
            import shutil
            
            # For walk-forward, we'll use the existing model
            # In a full implementation, you'd retrain here
            temp_model_path = f"models/walkforward_temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            shutil.copy(self.model_path, temp_model_path)
            return temp_model_path
        
        # Define backtest function
        def backtest_model(model_path: str, test_data: pd.DataFrame) -> dict:
            """Backtest model on test data"""
            # Create temporary multi-timeframe data with test data
            test_multi_tf = {tf: test_data for tf in self.config["environment"]["timeframes"]}
            
            # Create environment with test data
            test_env = TradingEnvironment(
                data=test_multi_tf,
                timeframes=self.config["environment"]["timeframes"],
                initial_capital=self.config["risk_management"]["initial_capital"],
                transaction_cost=self.config["risk_management"]["commission"] / self.config["risk_management"]["initial_capital"],
                reward_config=self.config["environment"]["reward"]
            )
            
            # Load model
            test_agent = PPOAgent(
                state_dim=test_env.state_dim,
                action_range=tuple(self.config["environment"]["action_range"]),
                device="cpu"
            )
            test_agent.load(model_path)
            
            # Run backtest
            state, info = test_env.reset()
            done = False
            pnls = []
            
            while not done:
                action, _, _ = test_agent.select_action(state, deterministic=True)
                state, reward, terminated, truncated, step_info = test_env.step(action)
                done = terminated or truncated
                if step_info.get("pnl") is not None:
                    pnls.append(step_info["pnl"])
            
            # Calculate metrics
            if len(pnls) == 0:
                return {"total_return": 0.0, "sharpe_ratio": 0.0, "win_rate": 0.0}
            
            total_return = sum(pnls) / self.config["risk_management"]["initial_capital"]
            sharpe = self._calculate_sharpe(pnls)
            win_rate = step_info.get("win_rate", 0.0)
            
            return {
                "total_return": total_return,
                "sharpe_ratio": sharpe,
                "win_rate": win_rate,
                "total_pnl": sum(pnls)
            }
        
        # Run walk-forward analysis
        results = analyzer.run_walk_forward(
            data=primary_data,
            train_func=train_model,
            backtest_func=backtest_model
        )
        
        return results
    
    def _calculate_sharpe(self, pnls: list) -> float:
        """
        Calculate Sharpe ratio from PnL values.
        
        CRITICAL FIX #5: Converts PnL to percentage returns before calculation.
        """
        if len(pnls) < 2:
            return 0.0
        
        # Get initial capital from config
        initial_capital = self.config["risk_management"]["initial_capital"]
        if initial_capital <= 0:
            return 0.0
        
        # Convert PnL to percentage returns (standard Sharpe formula)
        returns = np.array(pnls) / initial_capital
        
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        risk_free_rate = 0.0  # Default risk-free rate
        
        if std_return == 0:
            return 0.0
        
        # Sharpe ratio = (mean_return - risk_free_rate) / std_return * sqrt(periods_per_year)
        # Using 252 trading days for annualization (standard)
        return (mean_return - risk_free_rate) / std_return * np.sqrt(252)
    
    def _calculate_sortino(self, returns: list) -> float:
        """Calculate Sortino ratio (only penalizes downside volatility)"""
        if len(returns) < 2:
            return 0.0
        downside = [r for r in returns if r < 0]
        if len(downside) == 0 or np.std(downside) == 0:
            return 0.0
        return np.mean(returns) / np.std(downside) * np.sqrt(252)
    
    def _calculate_profit_factor(self, pnls: list) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = sum([p for p in pnls if p > 0])
        gross_loss = abs(sum([p for p in pnls if p < 0]))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def print_results(self, results: dict):
        """Print backtest results"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Episodes: {results['n_episodes']}")
        print(f"\nPerformance:")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Total PnL: ${results['total_pnl']:.2f}")
        print(f"  Mean PnL per Episode: ${results['mean_pnl']:.2f} ± ${results['std_pnl']:.2f}")
        
        print(f"\nTrading Statistics:")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Mean Trades per Episode: {results['mean_trades']:.1f}")
        print(f"  Mean Win Rate: {results['mean_win_rate']*100:.1f}%")
        
        print(f"\nRisk Metrics:")
        print(f"  Mean Max Drawdown: {results['mean_max_drawdown']*100:.2f}%")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {results['sortino_ratio']:.2f}")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print("="*60)
    
    def plot_results(self, save_path: str = "backtest_results.png"):
        """Plot equity curves and performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity curves
        ax1 = axes[0, 0]
        for i, equity_curve in enumerate(self.equity_curves[:5]):  # Show first 5
            ax1.plot(equity_curve, alpha=0.7, label=f"Episode {i+1}")
        ax1.set_title("Equity Curves (First 5 Episodes)")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Equity ($)")
        ax1.legend()
        ax1.grid(True)
        
        # PnL distribution
        ax2 = axes[0, 1]
        if self.results:
            pnls = [self.results[-1].get('mean_pnl', 0)]
            ax2.hist([r['mean_pnl'] for r in self.results], bins=20, edgecolor='black')
            ax2.set_title("PnL Distribution")
            ax2.set_xlabel("PnL ($)")
            ax2.set_ylabel("Frequency")
            ax2.axvline(x=0, color='r', linestyle='--', label='Break Even')
            ax2.legend()
            ax2.grid(True)
        
        # Win rate
        ax3 = axes[1, 0]
        if self.results:
            win_rates = [r['mean_win_rate']*100 for r in self.results]
            ax3.plot(win_rates, marker='o')
            ax3.set_title("Win Rate Over Episodes")
            ax3.set_xlabel("Backtest Run")
            ax3.set_ylabel("Win Rate (%)")
            ax3.grid(True)
        
        # Performance metrics
        ax4 = axes[1, 1]
        if self.results:
            metrics = ['mean_pnl', 'sharpe_ratio', 'profit_factor']
            values = [abs(self.results[-1][m]) for m in metrics]
            labels = ['Mean PnL (abs)', 'Sharpe Ratio', 'Profit Factor']
            ax4.bar(labels, values)
            ax4.set_title("Key Performance Metrics")
            ax4.set_ylabel("Value")
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\n📊 Results plot saved to: {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Backtest RL Trading Agent")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config_full.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backtest_results.png",
        help="Output path for plots"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create backtester
    backtester = Backtester(config, args.model)
    
    # Run backtest
    results = backtester.run_backtest(n_episodes=args.episodes)
    
    # Print results
    backtester.print_results(results)
    
    # Plot results
    backtester.plot_results(save_path=args.output)
    
    print("\n✅ Backtest complete!")


if __name__ == "__main__":
    main()

