"""
Walk-Forward Analysis for Backtesting

Performs walk-forward analysis to prevent overfitting:
- Train on period N, test on period N+1
- Rolling window or expanding window
- Out-of-sample performance tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class WalkForwardResult:
    """Result from a single walk-forward period"""
    train_period_start: int
    train_period_end: int
    test_period_start: int
    test_period_end: int
    test_metrics: Dict
    model_path: Optional[str] = None


class WalkForwardAnalyzer:
    """
    Performs walk-forward analysis to validate model performance.
    
    Walk-forward analysis:
    - Trains model on period N
    - Tests on period N+1 (out-of-sample)
    - Rolling or expanding window
    - Tracks stability across periods
    """
    
    def __init__(
        self,
        train_window: int = 252,  # 1 year (trading days)
        test_window: int = 63,     # 3 months
        step_size: int = 21,       # 1 month step
        window_type: str = "rolling"  # "rolling" or "expanding"
    ):
        """
        Initialize walk-forward analyzer.
        
        Args:
            train_window: Number of periods for training
            test_window: Number of periods for testing
            step_size: Step size between windows
            window_type: "rolling" (fixed size) or "expanding" (grows over time)
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.window_type = window_type
        
        self.results: List[WalkForwardResult] = []
    
    def run_walk_forward(
        self,
        data: pd.DataFrame,
        train_func: callable,
        backtest_func: callable,
        train_params: Optional[Dict] = None,
        backtest_params: Optional[Dict] = None
    ) -> Dict:
        """
        Run walk-forward analysis.
        
        Args:
            data: Full dataset (will be split into train/test)
            train_func: Function to train model: train_func(train_data, **train_params) -> model_path
            backtest_func: Function to backtest: backtest_func(model_path, test_data, **backtest_params) -> metrics
            train_params: Parameters for training function
            backtest_params: Parameters for backtest function
        
        Returns:
            Dictionary with:
            - walk_forward_results: List of WalkForwardResult
            - stability_metrics: Consistency across periods
            - overfitting_score: Measure of overfitting risk
        """
        train_params = train_params or {}
        backtest_params = backtest_params or {}
        
        results = []
        total_periods = len(data)
        
        # Calculate number of windows
        if self.window_type == "rolling":
            # Rolling window: fixed size, slides forward
            num_windows = (total_periods - self.train_window - self.test_window) // self.step_size + 1
            train_start = 0
        else:
            # Expanding window: grows over time
            num_windows = (total_periods - self.train_window - self.test_window) // self.step_size + 1
            train_start = 0  # Always start from beginning
        
        print(f"\n{'='*70}")
        print(f"WALK-FORWARD ANALYSIS")
        print(f"{'='*70}")
        print(f"Total data points: {total_periods}")
        print(f"Train window: {self.train_window} periods")
        print(f"Test window: {self.test_window} periods")
        print(f"Step size: {self.step_size} periods")
        print(f"Window type: {self.window_type}")
        print(f"Number of windows: {num_windows}")
        print(f"{'='*70}\n")
        
        for i in range(num_windows):
            # Calculate window boundaries
            if self.window_type == "rolling":
                train_start_idx = train_start + i * self.step_size
            else:
                train_start_idx = 0  # Always start from beginning
            
            train_end_idx = train_start_idx + self.train_window
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + self.test_window
            
            # Check if we have enough data
            if test_end_idx > total_periods:
                print(f"Window {i+1}: Not enough data (need {test_end_idx}, have {total_periods})")
                break
            
            print(f"Window {i+1}/{num_windows}:")
            print(f"  Train: {train_start_idx} to {train_end_idx} ({self.train_window} periods)")
            print(f"  Test:  {test_start_idx} to {test_end_idx} ({self.test_window} periods)")
            
            # Split data
            train_data = data.iloc[train_start_idx:train_end_idx].copy()
            test_data = data.iloc[test_start_idx:test_end_idx].copy()
            
            # Train model
            print(f"  Training model...")
            try:
                model_path = train_func(train_data, **train_params)
                print(f"  [OK] Model trained: {model_path}")
            except Exception as e:
                print(f"  [FAIL] Training failed: {e}")
                continue
            
            # Test model (out-of-sample)
            print(f"  Backtesting on out-of-sample data...")
            try:
                test_metrics = backtest_func(model_path, test_data, **backtest_params)
                print(f"  [OK] Backtest complete")
                print(f"     Return: {test_metrics.get('total_return', 0):.2%}")
                print(f"     Sharpe: {test_metrics.get('sharpe_ratio', 0):.2f}")
                print(f"     Win Rate: {test_metrics.get('win_rate', 0):.2%}")
            except Exception as e:
                print(f"  [FAIL] Backtest failed: {e}")
                continue
            
            # Store result
            result = WalkForwardResult(
                train_period_start=train_start_idx,
                train_period_end=train_end_idx,
                test_period_start=test_start_idx,
                test_period_end=test_end_idx,
                test_metrics=test_metrics,
                model_path=model_path
            )
            results.append(result)
            self.results.append(result)
        
        # Analyze stability
        stability = self._analyze_stability(results)
        
        # Calculate overfitting score
        overfitting_score = self._calculate_overfitting_score(results)
        
        print(f"\n{'='*70}")
        print(f"WALK-FORWARD ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"Total windows: {len(results)}")
        print(f"Average return: {stability.get('avg_return', 0):.2%}")
        print(f"Return std: {stability.get('return_std', 0):.2%}")
        print(f"Overfitting score: {overfitting_score:.2f} (lower is better)")
        print(f"{'='*70}\n")
        
        return {
            "walk_forward_results": results,
            "stability_metrics": stability,
            "overfitting_score": overfitting_score,
            "num_windows": len(results)
        }
    
    def _analyze_stability(self, results: List[WalkForwardResult]) -> Dict:
        """
        Analyze stability of results across periods.
        
        Args:
            results: List of walk-forward results
        
        Returns:
            Dictionary with stability metrics
        """
        if len(results) == 0:
            return {}
        
        # Extract metrics
        returns = [r.test_metrics.get("total_return", 0) for r in results]
        sharpe_ratios = [r.test_metrics.get("sharpe_ratio", 0) for r in results]
        win_rates = [r.test_metrics.get("win_rate", 0) for r in results]
        
        return {
            "avg_return": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)),
            "avg_sharpe": float(np.mean(sharpe_ratios)),
            "sharpe_std": float(np.std(sharpe_ratios)),
            "avg_win_rate": float(np.mean(win_rates)),
            "win_rate_std": float(np.std(win_rates)),
            "consistency": float(1.0 - (np.std(returns) / (abs(np.mean(returns)) + 1e-6)))  # Higher = more consistent
        }
    
    def _calculate_overfitting_score(self, results: List[WalkForwardResult]) -> float:
        """
        Calculate overfitting score.
        
        Lower score = less overfitting (better)
        Higher score = more overfitting (worse)
        
        Args:
            results: List of walk-forward results
        
        Returns:
            Overfitting score (0-1, lower is better)
        """
        if len(results) < 2:
            return 0.0
        
        # Extract returns
        returns = [r.test_metrics.get("total_return", 0) for r in results]
        
        # Calculate coefficient of variation (std / mean)
        # Higher variation = more overfitting (less stable)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if abs(mean_return) < 1e-6:
            return 1.0  # Very bad if mean is near zero
        
        cv = std_return / abs(mean_return)
        
        # Normalize to 0-1 (higher = more overfitting)
        # CV > 1.0 is very bad, CV < 0.2 is good
        overfitting_score = min(1.0, cv / 1.0)
        
        return float(overfitting_score)
    
    def get_summary(self) -> Dict:
        """
        Get summary of walk-forward analysis.
        
        Returns:
            Summary dictionary
        """
        if len(self.results) == 0:
            return {"message": "No walk-forward results available"}
        
        stability = self._analyze_stability(self.results)
        overfitting_score = self._calculate_overfitting_score(self.results)
        
        return {
            "num_periods": len(self.results),
            "stability": stability,
            "overfitting_score": overfitting_score,
            "recommendation": self._get_recommendation(stability, overfitting_score)
        }
    
    def _get_recommendation(self, stability: Dict, overfitting_score: float) -> str:
        """
        Get recommendation based on stability and overfitting score.
        
        Args:
            stability: Stability metrics
            overfitting_score: Overfitting score
        
        Returns:
            Recommendation string
        """
        if overfitting_score < 0.2:
            return "[PASS] Low overfitting risk - model is stable across periods"
        elif overfitting_score < 0.5:
            return "[WARN] Moderate overfitting risk - consider reducing model complexity"
        else:
            return "[FAIL] High overfitting risk - model may not generalize well"

