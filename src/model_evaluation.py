"""
Model Evaluation and Comparison Framework

Evaluates and compares model versions to select the best performing one.
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from src.rl_agent import PPOAgent
from src.trading_env import TradingEnvironment
from src.data_extraction import DataExtractor
from src.trading_hours import TradingHoursManager


@dataclass
class ModelMetrics:
    """Performance metrics for a model"""
    model_path: str
    timestamp: str
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    average_win: float
    average_loss: float


class ModelEvaluator:
    """
    Evaluates and compares model performance.
    """
    
    def __init__(
        self,
        config: Dict,
        test_data: Optional[Dict] = None
    ):
        """
        Initialize model evaluator.
        
        Args:
            config: Configuration
            test_data: Test dataset (if None, will load from config)
        """
        self.config = config
        self.test_data = test_data
        
        # Evaluation results storage
        self.evaluation_results: List[ModelMetrics] = []
        
        # Load test data if not provided
        if self.test_data is None:
            self._load_test_data()
    
    def _load_test_data(self):
        """Load test data for evaluation"""
        extractor = DataExtractor()
        instrument = self.config["environment"]["instrument"]
        timeframes = self.config["environment"]["timeframes"]
        trading_hours_cfg = self.config["environment"].get("trading_hours", {})
        trading_hours_manager = None
        if trading_hours_cfg.get("enabled"):
            trading_hours_manager = TradingHoursManager.from_dict(trading_hours_cfg)
        
        self.test_data = extractor.load_multi_timeframe_data(
            instrument,
            timeframes,
            trading_hours=trading_hours_manager
        )
    
    def evaluate_model(
        self,
        model_path: str,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> ModelMetrics:
        """
        Evaluate a model and return metrics.
        
        Args:
            model_path: Path to model checkpoint
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic actions
        
        Returns:
            ModelMetrics with performance scores
        """
        print(f"\n📊 Evaluating model: {model_path}")
        
        # Load agent
        agent = PPOAgent(
            state_dim=self.config["environment"]["state_features"],
            action_range=tuple(self.config["environment"]["action_range"]),
            device="cpu"
        )
        agent.load(model_path)
        agent.actor.eval()
        agent.critic.eval()
        
        # Create environment with ALL parameters from config (matching training)
        action_threshold = self.config["environment"].get("action_threshold", 0.05)
        max_episode_steps = self.config["environment"].get("max_episode_steps", 10000)
        
        env = TradingEnvironment(
            data=self.test_data,
            timeframes=self.config["environment"]["timeframes"],
            initial_capital=self.config["risk_management"]["initial_capital"],
            transaction_cost=self.config["risk_management"]["commission"] / \
                           self.config["risk_management"]["initial_capital"],
            reward_config=self.config["environment"]["reward"],
            action_threshold=action_threshold,
            max_episode_steps=max_episode_steps
        )
        
        # Run evaluation episodes
        all_rewards = []
        all_pnls = []
        all_trades = []
        all_win_rates = []
        all_drawdowns = []
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _, _ = agent.select_action(state, deterministic=deterministic)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            # Collect metrics
            all_rewards.append(episode_reward)
            all_pnls.append(info.get("pnl", 0))
            all_trades.append(info.get("trades", 0))
            all_win_rates.append(info.get("win_rate", 0))
            all_drawdowns.append(info.get("max_drawdown", 0))
        
        # Calculate aggregate metrics
        metrics = self._calculate_metrics(
            model_path,
            all_rewards,
            all_pnls,
            all_trades,
            all_win_rates,
            all_drawdowns
        )
        
        self.evaluation_results.append(metrics)
        
        return metrics
    
    def _calculate_metrics(
        self,
        model_path: str,
        rewards: List[float],
        pnls: List[float],
        trades: List[int],
        win_rates: List[float],
        drawdowns: List[float]
    ) -> ModelMetrics:
        """Calculate performance metrics"""
        # Basic stats
        total_return = sum(pnls) / self.config["risk_management"]["initial_capital"]
        mean_return = np.mean(pnls)
        std_return = np.std(pnls)
        
        # Sharpe ratio
        sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0.0
        
        # Sortino ratio (only downside volatility)
        downside_returns = [r for r in pnls if r < 0]
        downside_std = np.std(downside_returns) if downside_returns else 0.0
        sortino = mean_return / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
        
        # Max drawdown
        max_drawdown = max(drawdowns) if drawdowns else 0.0
        
        # Win rate
        win_rate = np.mean(win_rates) if win_rates else 0.0
        
        # Profit factor
        gross_profit = sum([p for p in pnls if p > 0])
        gross_loss = abs(sum([p for p in pnls if p < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Trade stats
        total_trades = sum(trades)
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0.0
        
        return ModelMetrics(
            model_path=model_path,
            timestamp=datetime.now().isoformat(),
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            average_win=avg_win,
            average_loss=avg_loss
        )
    
    def compare_models(
        self,
        model_paths: List[str],
        n_episodes: int = 10
    ) -> Dict[str, ModelMetrics]:
        """
        Compare multiple models.
        
        Args:
            model_paths: List of model paths to compare
            n_episodes: Episodes per model
        
        Returns:
            Dictionary mapping model paths to metrics
        """
        print("\n" + "="*60)
        print("Model Comparison")
        print("="*60)
        
        results = {}
        
        for model_path in model_paths:
            metrics = self.evaluate_model(model_path, n_episodes)
            results[model_path] = metrics
        
        # Print comparison
        self._print_comparison(results)
        
        # Save results
        self._save_comparison(results)
        
        return results
    
    def _print_comparison(self, results: Dict[str, ModelMetrics]):
        """Print comparison table"""
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        for path, metrics in results.items():
            print(f"\n{Path(path).name}:")
            print(f"  Total Return: {metrics.total_return*100:.2f}%")
            print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
            print(f"  Max Drawdown: {metrics.max_drawdown*100:.2f}%")
            print(f"  Win Rate: {metrics.win_rate*100:.1f}%")
            print(f"  Profit Factor: {metrics.profit_factor:.2f}")
            print(f"  Total Trades: {metrics.total_trades}")
        
        # Find best model
        best = max(results.values(), key=lambda m: m.sharpe_ratio)
        print(f"\n🏆 Best Model (by Sharpe): {Path(best.model_path).name}")
        print(f"   Sharpe: {best.sharpe_ratio:.2f}, Return: {best.total_return*100:.2f}%")
    
    def _save_comparison(self, results: Dict[str, ModelMetrics]):
        """Save comparison results"""
        comparison_file = Path("logs") / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        comparison_file.parent.mkdir(exist_ok=True)
        
        data = {
            "comparison_timestamp": datetime.now().isoformat(),
            "models": {path: asdict(metrics) for path, metrics in results.items()}
        }
        
        with open(comparison_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Comparison saved to: {comparison_file}")
    
    def select_best_model(
        self,
        model_paths: List[str],
        metric: str = "sharpe_ratio"
    ) -> str:
        """
        Select best model based on specified metric.
        
        Args:
            model_paths: List of model paths
            metric: Metric to optimize (sharpe_ratio, total_return, etc.)
        
        Returns:
            Path to best model
        """
        # Evaluate all models
        results = {}
        for path in model_paths:
            metrics = self.evaluate_model(path, n_episodes=5)
            results[path] = metrics
        
        # Select best
        best_path = max(results.keys(), key=lambda p: getattr(results[p], metric))
        best_metrics = results[best_path]
        
        print(f"\n🏆 Best model selected: {Path(best_path).name}")
        print(f"   Metric ({metric}): {getattr(best_metrics, metric):.4f}")
        
        return best_path


# Example usage
if __name__ == "__main__":
    # This would be called with actual config
    print("Model Evaluation Module")
    print("Use this to compare and select best models")

