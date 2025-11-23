"""
Adaptive Training System

Intelligently monitors training performance and automatically adjusts parameters
without stopping training. Evaluates models during training and adapts:
- Exploration (entropy_coef)
- Reward function weights
- Action thresholds
- Learning rates
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import yaml

from src.model_evaluation import ModelEvaluator, ModelMetrics
from src.quality_scorer import QualityScorer


@dataclass
class PerformanceSnapshot:
    """Snapshot of model performance at a point in time"""
    timestep: int
    episode: int
    total_trades: int
    win_rate: float
    total_return: float
    sharpe_ratio: float
    mean_reward: float
    entropy_coef: float
    inaction_penalty: float
    timestamp: str


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive adjustments"""
    # Evaluation settings
    eval_frequency: int = 10000  # Evaluate every N timesteps
    eval_episodes: int = 3  # Episodes per evaluation
    
    # Performance thresholds
    min_trades_per_episode: float = 0.5  # Minimum trades per episode
    min_win_rate: float = 0.35  # Minimum acceptable win rate
    target_sharpe: float = 0.5  # Target Sharpe ratio
    
    # Adjustment parameters
    entropy_adjustment_rate: float = 0.01  # How much to adjust entropy_coef
    min_entropy_coef: float = 0.01
    max_entropy_coef: float = 0.1
    
    inaction_penalty_base: float = 0.0001  # Base inaction penalty
    inaction_penalty_max: float = 0.001  # Maximum inaction penalty
    inaction_adjustment_rate: float = 0.00005  # How much to increase penalty
    
    # Learning rate adjustment
    lr_adjustment_enabled: bool = True
    lr_reduction_factor: float = 0.95  # Reduce LR if performance plateaus
    
    # Auto-save settings
    auto_save_on_improvement: bool = True
    improvement_threshold: float = 0.05  # 5% improvement triggers save
    
    # History tracking
    max_history_size: int = 50  # Keep last N snapshots
    
    # Risk/reward ratio adjustment (NEW)
    rr_adjustment_enabled: bool = True
    min_rr_threshold: float = 1.3  # Minimum R:R threshold
    max_rr_threshold: float = 2.5  # Maximum R:R threshold
    rr_adjustment_rate: float = 0.1  # How much to adjust per step
    
    # Quality filter adjustment (NEW)
    quality_filter_adjustment_enabled: bool = True
    min_action_confidence_range: Tuple[float, float] = (0.1, 0.2)  # (min, max)
    min_quality_score_range: Tuple[float, float] = (0.3, 0.5)  # (min, max)
    quality_adjustment_rate: float = 0.01  # How much to adjust per step


class AdaptiveTrainer:
    """
    Adaptive training system that monitors and adjusts parameters automatically.
    """
    
    def __init__(self, config_path: str, adaptive_config: Optional[AdaptiveConfig] = None):
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.adaptive_config = adaptive_config or AdaptiveConfig()
        
        # Performance history
        self.performance_history: List[PerformanceSnapshot] = []
        self.best_performance: Optional[PerformanceSnapshot] = None
        
        # Current adaptive parameters (will be modified during training)
        self.current_entropy_coef = self.config["model"]["entropy_coef"]
        self.current_inaction_penalty = self.adaptive_config.inaction_penalty_base
        self.current_learning_rate = self.config["model"]["learning_rate"]
        
        # Current adaptive profitability parameters (NEW)
        self.current_min_risk_reward_ratio = self.config["environment"]["reward"].get("min_risk_reward_ratio", 1.5)
        quality_filters_config = self.config["environment"]["reward"].get("quality_filters", {})
        self.current_min_action_confidence = quality_filters_config.get("min_action_confidence", 0.15)
        self.current_min_quality_score = quality_filters_config.get("min_quality_score", 0.4)
        
        # Evaluation
        self.evaluator = ModelEvaluator(self.config)
        
        # Quality scorer for win rate profitability checks
        quality_scorer_config = self.config.get("decision_gate", {}).get("quality_scorer", {})
        self.quality_scorer = QualityScorer(quality_scorer_config)
        
        # Metrics tracking
        self.last_eval_timestep = 0
        self.consecutive_no_trade_evals = 0
        self.consecutive_low_performance = 0
        
        # NEW: Win rate tracking for aggressive adjustments
        self.consecutive_low_win_rate = 0
        self.consecutive_good_performance = 0
        self.training_paused = False
        self.pause_reason = None
        
        # Win rate profitability tracking
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.winning_pnls = []  # Track PnL of winning trades
        self.losing_pnls = []  # Track PnL of losing trades
        self.rolling_window = 100  # Use last 100 trades for averages
        
        # Real-time monitoring
        self.last_trade_check_timestep = 0
        self.trade_check_frequency = 2000  # Check every 2k timesteps (more frequent for faster response)
        self.consecutive_no_trade_episodes = 0  # Episodes with no trades
        self.last_adjustment_timestep = 0
        self.min_adjustment_interval = 1000  # Minimum time between adjustments (allow more frequent adjustments)
        
        # Logging
        self.log_dir = Path("logs/adaptive_training")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_file = self.log_dir / "performance_snapshots.jsonl"
        self.config_history_file = self.log_dir / "config_adjustments.jsonl"
        
        # Initialize adaptive config file with current values
        self._update_reward_config()
    
    def should_evaluate(self, timestep: int) -> bool:
        """Check if we should run an evaluation"""
        return (timestep - self.last_eval_timestep) >= self.adaptive_config.eval_frequency
    
    def check_trading_activity(
        self,
        timestep: int,
        episode: int,
        current_episode_trades: int,
        current_episode_length: int,
        agent
    ) -> Optional[Dict]:
        """
        Real-time check of trading activity during training.
        This is a lightweight check that doesn't require full evaluation.
        
        Returns adjustments dict if adjustments are needed, None otherwise.
        """
        # Only check periodically to avoid overhead
        if (timestep - self.last_trade_check_timestep) < self.trade_check_frequency:
            return None
        
        self.last_trade_check_timestep = timestep
        
        # Check if we've had enough time since last adjustment
        if (timestep - self.last_adjustment_timestep) < self.min_adjustment_interval:
            return None
        
        adjustments = {}
        # Estimate episode progress (use a reasonable default if we don't know max_steps)
        # Most episodes are ~10k steps, but we'll be conservative and check earlier
        estimated_max_steps = 10000
        episode_progress = current_episode_length / max(1, estimated_max_steps)
        
        # Detect no trades condition - check earlier and more aggressively
        # If we're past 10% of estimated episode and still no trades, that's a problem
        # Also check if we've been training for a while with no trades across episodes
        no_trades_detected = current_episode_trades == 0
        
        # Early detection: if episode is progressing but no trades
        early_no_trades = no_trades_detected and episode_progress > 0.1  # 10% of episode
        
        # Persistent no trades: multiple episodes with no trades
        persistent_no_trades = self.consecutive_no_trade_episodes >= 1
        
        if early_no_trades or persistent_no_trades:
            # Only increment counter when we first detect no trades in an episode
            # (counter is incremented when episode completes, not during episode)
            print(f"\n[ADAPTIVE] [WARN] NO TRADES DETECTED")
            print(f"   Episode: {episode}, Progress: {episode_progress*100:.0f}%, Trades: {current_episode_trades}")
            print(f"   Consecutive no-trade episodes: {self.consecutive_no_trade_episodes}")
            print(f"   Early detection: {early_no_trades}, Persistent: {persistent_no_trades}")
            
            # Intelligent adjustment based on how long we've had no trades
            adjustment_severity = min(self.consecutive_no_trade_episodes, 5)  # Cap at 5x
            
            # Increase exploration more aggressively
            old_entropy = self.current_entropy_coef
            entropy_increase = self.adaptive_config.entropy_adjustment_rate * adjustment_severity * 2  # 2x for real-time
            self.current_entropy_coef = min(
                self.adaptive_config.max_entropy_coef,
                self.current_entropy_coef + entropy_increase
            )
            
            if self.current_entropy_coef != old_entropy:
                agent.entropy_coef = self.current_entropy_coef
                adjustments["entropy_coef"] = {
                    "old": old_entropy,
                    "new": self.current_entropy_coef,
                    "reason": f"No trades detected (episode {episode}, {episode_progress*100:.0f}% complete)"
                }
                print(f"   [ADAPT] Increased entropy_coef: {old_entropy:.4f} -> {self.current_entropy_coef:.4f}")
            
            # Increase inaction penalty more aggressively
            old_penalty = self.current_inaction_penalty
            penalty_increase = self.adaptive_config.inaction_adjustment_rate * adjustment_severity * 3  # 3x for real-time
            self.current_inaction_penalty = min(
                self.adaptive_config.inaction_penalty_max,
                self.current_inaction_penalty + penalty_increase
            )
            
            if self.current_inaction_penalty != old_penalty:
                adjustments["inaction_penalty"] = {
                    "old": old_penalty,
                    "new": self.current_inaction_penalty,
                    "reason": f"Encourage trading - no trades in {self.consecutive_no_trade_episodes} episodes"
                }
                self._update_reward_config()
                print(f"   [ADAPT] Increased inaction_penalty: {old_penalty:.6f} -> {self.current_inaction_penalty:.6f}")
            
            # If still no trades after multiple episodes, also adjust learning rate
            if self.consecutive_no_trade_episodes >= 3:
                # Slightly reduce learning rate to allow more exploration
                if self.adaptive_config.lr_adjustment_enabled:
                    old_lr = self.current_learning_rate
                    self.current_learning_rate *= 0.98  # Small reduction
                    
                    for param_group in agent.actor_optimizer.param_groups:
                        param_group['lr'] = self.current_learning_rate
                    for param_group in agent.critic_optimizer.param_groups:
                        param_group['lr'] = self.current_learning_rate
                    
                    adjustments["learning_rate"] = {
                        "old": old_lr,
                        "new": self.current_learning_rate,
                        "reason": f"No trades for {self.consecutive_no_trade_episodes} episodes - reduce LR for more exploration"
                    }
                    print(f"   [ADAPT] Reduced learning_rate: {old_lr:.6f} -> {self.current_learning_rate:.6f}")
        
        # Reset counter if trades are happening
        elif current_episode_trades > 0:
            if self.consecutive_no_trade_episodes > 0:
                print(f"[ADAPTIVE] [OK] Trades detected! Resetting no-trade counter (was {self.consecutive_no_trade_episodes})")
            self.consecutive_no_trade_episodes = 0
        
        # Save adjustments if any were made
        if adjustments:
            self.last_adjustment_timestep = timestep
            self._save_adjustment(timestep=timestep, adjustments=adjustments)
            print(f"   [OK] Adjustments saved to log")
        
        return adjustments if adjustments else None
    
    def evaluate_and_adapt(
        self,
        model_path: str,
        timestep: int,
        episode: int,
        mean_reward: float,
        agent  # PPOAgent instance
    ) -> Dict:
        """
        Evaluate model performance and adapt parameters if needed.
        
        Returns:
            Dict with evaluation results and any adjustments made
        """
        print(f"\n{'='*70}")
        print(f"ADAPTIVE EVALUATION (Timestep: {timestep:,}, Episode: {episode})")
        print(f"{'='*70}")
        
        # Run evaluation
        try:
            metrics = self.evaluator.evaluate_model(
                model_path=model_path,
                n_episodes=self.adaptive_config.eval_episodes,
                deterministic=False  # Use stochastic for evaluation
            )
        except Exception as e:
            print(f"[ERROR] Evaluation failed: {e}")
            return {"error": str(e)}
        
        # Create snapshot
        snapshot = PerformanceSnapshot(
            timestep=timestep,
            episode=episode,
            total_trades=metrics.total_trades,
            win_rate=metrics.win_rate,
            total_return=metrics.total_return,
            sharpe_ratio=metrics.sharpe_ratio,
            mean_reward=mean_reward,
            entropy_coef=self.current_entropy_coef,
            inaction_penalty=self.current_inaction_penalty,
            timestamp=datetime.now().isoformat()
        )
        
        # Save snapshot
        self._save_snapshot(snapshot)
        self.performance_history.append(snapshot)
        if len(self.performance_history) > self.adaptive_config.max_history_size:
            self.performance_history.pop(0)
        
        # Update best performance
        if self.best_performance is None or self._is_better(snapshot, self.best_performance):
            improvement = self._calculate_improvement(snapshot, self.best_performance)
            self.best_performance = snapshot
            
            if improvement and improvement >= self.adaptive_config.improvement_threshold:
                print(f"[SUCCESS] SIGNIFICANT IMPROVEMENT: {improvement*100:.1f}%")
                if self.adaptive_config.auto_save_on_improvement:
                    # Trigger checkpoint save (will be handled by trainer)
                    return {
                        "metrics": metrics,
                        "snapshot": snapshot,
                        "improvement": improvement,
                        "should_save": True,
                        "adjustments": {}
                    }
        
        # Analyze performance and make adjustments
        adjustments = self._analyze_and_adjust(snapshot, agent)
        
        # Print results
        print(f"\n[PERF] Performance Metrics:")
        print(f"   Total Trades: {metrics.total_trades}")
        print(f"   Win Rate: {metrics.win_rate*100:.1f}%")
        print(f"   Total Return: {metrics.total_return*100:.2f}%")
        print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        
        if adjustments:
            print(f"\n[ADAPT] Adaptive Adjustments:")
            for key, value in adjustments.items():
                print(f"   {key}: {value}")
        
        self.last_eval_timestep = timestep
        
        return {
            "metrics": metrics,
            "snapshot": snapshot,
            "adjustments": adjustments,
            "should_save": False
        }
    
    def check_win_rate_profitability(
        self,
        total_trades: int,
        winning_trades: int,
        winning_pnls: List[float],
        losing_pnls: List[float],
        commission_rate: float = 0.0003
    ) -> Dict:
        """
        Check if current win rate is profitable after commissions.
        
        Args:
            total_trades: Total number of trades
            winning_trades: Number of winning trades
            winning_pnls: List of PnL values for winning trades
            losing_pnls: List of PnL values for losing trades
            commission_rate: Commission rate (default 0.0003 = 0.03%)
        
        Returns:
            Dict with profitability analysis
        """
        if total_trades < 50:  # Not enough data
            return {
                "is_profitable": True,
                "breakeven_win_rate": 0.5,
                "current_win_rate": winning_trades / max(1, total_trades),
                "reason": "Not enough data"
            }
        
        # Calculate averages (use rolling window)
        recent_winning = winning_pnls[-self.rolling_window:] if len(winning_pnls) > self.rolling_window else winning_pnls
        recent_losing = losing_pnls[-self.rolling_window:] if len(losing_pnls) > self.rolling_window else losing_pnls
        
        if not recent_winning or not recent_losing:
            return {
                "is_profitable": True,
                "breakeven_win_rate": 0.5,
                "current_win_rate": winning_trades / max(1, total_trades),
                "reason": "Not enough win/loss data"
            }
        
        avg_win = sum(recent_winning) / len(recent_winning)
        avg_loss = abs(sum(recent_losing) / len(recent_losing))
        
        # Estimate commission cost (simplified - would need actual trade sizes)
        estimated_commission = 100000.0 * commission_rate  # Assume $100k capital
        
        # Calculate breakeven win rate
        breakeven_win_rate = self.quality_scorer.calculate_breakeven_win_rate(
            avg_win=avg_win,
            avg_loss=avg_loss,
            commission_cost=estimated_commission
        )
        
        # Current win rate
        current_win_rate = winning_trades / total_trades
        
        # Check if profitable
        is_profitable = current_win_rate > breakeven_win_rate
        
        return {
            "is_profitable": is_profitable,
            "breakeven_win_rate": breakeven_win_rate,
            "current_win_rate": current_win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expected_value": self.quality_scorer.calculate_expected_value(
                win_rate=current_win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                commission_cost=estimated_commission
            ),
            "reason": "Profitable" if is_profitable else f"Win rate {current_win_rate:.1%} < breakeven {breakeven_win_rate:.1%}"
        }
    
    def _analyze_and_adjust(
        self,
        snapshot: PerformanceSnapshot,
        agent
    ) -> Dict:
        """Analyze performance and make adaptive adjustments"""
        adjustments = {}
        
        # Check for no trades
        trades_per_episode = snapshot.total_trades / max(1, self.adaptive_config.eval_episodes)
        
        if trades_per_episode < self.adaptive_config.min_trades_per_episode:
            self.consecutive_no_trade_evals += 1
            print(f"[WARN] LOW TRADE ACTIVITY: {trades_per_episode:.2f} trades/episode")
            
            # Increase exploration
            old_entropy = self.current_entropy_coef
            self.current_entropy_coef = min(
                self.adaptive_config.max_entropy_coef,
                self.current_entropy_coef + self.adaptive_config.entropy_adjustment_rate * self.consecutive_no_trade_evals
            )
            
            if self.current_entropy_coef != old_entropy:
                agent.entropy_coef = self.current_entropy_coef
                adjustments["entropy_coef"] = {
                    "old": old_entropy,
                    "new": self.current_entropy_coef,
                    "reason": "Low trade activity"
                }
            
            # Increase inaction penalty
            old_penalty = self.current_inaction_penalty
            self.current_inaction_penalty = min(
                self.adaptive_config.inaction_penalty_max,
                self.current_inaction_penalty + self.adaptive_config.inaction_adjustment_rate * self.consecutive_no_trade_evals
            )
            
            if self.current_inaction_penalty != old_penalty:
                adjustments["inaction_penalty"] = {
                    "old": old_penalty,
                    "new": self.current_inaction_penalty,
                    "reason": "Encourage trading"
                }
                # Update reward function (will be applied in next episode)
                self._update_reward_config()
        else:
            self.consecutive_no_trade_evals = 0
        
        # Check win rate profitability (NEW)
        profitability_check = self.check_win_rate_profitability(
            total_trades=snapshot.total_trades,
            winning_trades=int(snapshot.total_trades * snapshot.win_rate),
            winning_pnls=self.winning_pnls,
            losing_pnls=self.losing_pnls,
            commission_rate=0.0003  # 0.03%
        )
        
        # AGGRESSIVE profitability detection and response
        if not profitability_check["is_profitable"]:
            self.consecutive_low_win_rate += 1
            
            print(f"\n[CRITICAL] UNPROFITABLE DETECTED:")
            print(f"   Current win rate: {profitability_check['current_win_rate']:.1%}")
            print(f"   Breakeven win rate: {profitability_check['breakeven_win_rate']:.1%}")
            print(f"   Expected value: ${profitability_check['expected_value']:.2f}")
            print(f"   Consecutive unprofitable evaluations: {self.consecutive_low_win_rate}")
            
            # AGGRESSIVE tightening when unprofitable (5x normal rate)
            if self.consecutive_low_win_rate >= 2:
                old_confidence = self.current_min_action_confidence
                old_quality = self.current_min_quality_score
                old_rr = self.current_min_risk_reward_ratio
                old_entropy = self.current_entropy_coef
                
                # Aggressive increases (5x normal adjustment rate)
                self.current_min_action_confidence = min(0.25,
                    self.current_min_action_confidence + 0.05)
                self.current_min_quality_score = min(0.60,
                    self.current_min_quality_score + 0.10)
                self.current_min_risk_reward_ratio = min(3.0,
                    self.current_min_risk_reward_ratio + 0.5)
                # Reduce exploration to be more selective
                self.current_entropy_coef = max(0.01, self.current_entropy_coef * 0.9)
                agent.entropy_coef = self.current_entropy_coef
                
                adjustments["quality_filters"] = {
                    "min_action_confidence": {
                        "old": old_confidence,
                        "new": self.current_min_action_confidence,
                        "reason": f"AGGRESSIVE: Unprofitable for {self.consecutive_low_win_rate} evaluations"
                    },
                    "min_quality_score": {
                        "old": old_quality,
                        "new": self.current_min_quality_score,
                        "reason": f"AGGRESSIVE: Unprofitable for {self.consecutive_low_win_rate} evaluations"
                    },
                    "min_risk_reward_ratio": {
                        "old": old_rr,
                        "new": self.current_min_risk_reward_ratio,
                        "reason": f"AGGRESSIVE: Unprofitable for {self.consecutive_low_win_rate} evaluations"
                    },
                    "entropy_coef": {
                        "old": old_entropy,
                        "new": self.current_entropy_coef,
                        "reason": f"AGGRESSIVE: Reducing exploration for unprofitable model"
                    }
                }
                print(f"[ADAPT] AGGRESSIVE tightening: confidence {old_confidence:.3f}->{self.current_min_action_confidence:.3f}, "
                      f"quality {old_quality:.3f}->{self.current_min_quality_score:.3f}, "
                      f"R:R {old_rr:.2f}->{self.current_min_risk_reward_ratio:.2f}, "
                      f"entropy {old_entropy:.4f}->{self.current_entropy_coef:.4f}")
            
            # PAUSE training if win rate < 30% for 3+ consecutive evaluations
            if snapshot.win_rate < 0.30 and self.consecutive_low_win_rate >= 3:
                self.training_paused = True
                self.pause_reason = f"Win rate {snapshot.win_rate:.1%} < 30% for {self.consecutive_low_win_rate} consecutive evaluations"
                self._save_pause_state(snapshot.timestep)
                print(f"\n{'='*70}")
                print(f"[CRITICAL] TRAINING PAUSED")
                print(f"{'='*70}")
                print(f"Reason: {self.pause_reason}")
                print(f"Current win rate: {snapshot.win_rate:.1%}")
                print(f"Breakeven win rate: {profitability_check['breakeven_win_rate']:.1%}")
                print(f"Expected value: ${profitability_check['expected_value']:.2f}")
                print(f"\n[ACTION REQUIRED] Review performance and adjust parameters before resuming")
                print(f"Checkpoint saved. Resume with: --checkpoint models/checkpoint_{snapshot.timestep}.pt")
                print(f"{'='*70}\n")
        else:
            # Reset counter when profitable
            if self.consecutive_low_win_rate > 0:
                print(f"[OK] Profitability restored! Resetting consecutive low win rate counter (was {self.consecutive_low_win_rate})")
            self.consecutive_low_win_rate = 0
        
        # REWARD good performance (win rate > 50%)
        if snapshot.win_rate > 0.50:
            self.consecutive_good_performance += 1
            
            if self.consecutive_good_performance >= 2:
                # Reward good performance - relax filters slightly
                old_confidence = self.current_min_action_confidence
                old_quality = self.current_min_quality_score
                
                self.current_min_action_confidence = max(0.10,
                    self.current_min_action_confidence - 0.02)
                self.current_min_quality_score = max(0.30,
                    self.current_min_quality_score - 0.05)
                
                if "quality_filters" not in adjustments:
                    adjustments["quality_filters"] = {}
                
                adjustments["quality_filters"]["min_action_confidence"] = {
                    "old": old_confidence,
                    "new": self.current_min_action_confidence,
                    "reason": f"Rewarding good performance: Win rate {snapshot.win_rate:.1%} > 50% for {self.consecutive_good_performance} evaluations"
                }
                adjustments["quality_filters"]["min_quality_score"] = {
                    "old": old_quality,
                    "new": self.current_min_quality_score,
                    "reason": f"Rewarding good performance: Win rate {snapshot.win_rate:.1%} > 50% for {self.consecutive_good_performance} evaluations"
                }
                print(f"[ADAPT] Rewarding good performance: confidence {old_confidence:.3f}->{self.current_min_action_confidence:.3f}, "
                      f"quality {old_quality:.3f}->{self.current_min_quality_score:.3f} "
                      f"(win rate: {snapshot.win_rate:.1%})")
        else:
            self.consecutive_good_performance = 0
        
        # NEW: Adaptive Risk/Reward Ratio Adjustment
        if self.adaptive_config.rr_adjustment_enabled and profitability_check.get("avg_win", 0) > 0 and profitability_check.get("avg_loss", 0) > 0:
            current_rr_ratio = profitability_check["avg_win"] / profitability_check["avg_loss"]
            
            # If losing money (R:R < 1.5), tighten threshold
            if current_rr_ratio < 1.5:
                old_rr_threshold = self.current_min_risk_reward_ratio
                self.current_min_risk_reward_ratio = min(
                    self.adaptive_config.max_rr_threshold,
                    self.current_min_risk_reward_ratio + self.adaptive_config.rr_adjustment_rate
                )
                if self.current_min_risk_reward_ratio != old_rr_threshold:
                    adjustments["min_risk_reward_ratio"] = {
                        "old": old_rr_threshold,
                        "new": self.current_min_risk_reward_ratio,
                        "reason": f"Poor R:R ratio ({current_rr_ratio:.2f}:1) - tightening threshold"
                    }
                    print(f"[ADAPT] Tightened R:R threshold: {old_rr_threshold:.2f} -> {self.current_min_risk_reward_ratio:.2f} (current R:R: {current_rr_ratio:.2f}:1)")
            
            # If very profitable (R:R >= 2.0), can relax slightly
            elif current_rr_ratio >= 2.0:
                old_rr_threshold = self.current_min_risk_reward_ratio
                self.current_min_risk_reward_ratio = max(
                    self.adaptive_config.min_rr_threshold,
                    self.current_min_risk_reward_ratio - (self.adaptive_config.rr_adjustment_rate * 0.5)  # Relax slower
                )
                if self.current_min_risk_reward_ratio != old_rr_threshold:
                    adjustments["min_risk_reward_ratio"] = {
                        "old": old_rr_threshold,
                        "new": self.current_min_risk_reward_ratio,
                        "reason": f"Good R:R ratio ({current_rr_ratio:.2f}:1) - relaxing threshold slightly"
                    }
                    print(f"[ADAPT] Relaxed R:R threshold: {old_rr_threshold:.2f} -> {self.current_min_risk_reward_ratio:.2f} (current R:R: {current_rr_ratio:.2f}:1)")
        
        # NEW: Adaptive Quality Filter Adjustment
        if self.adaptive_config.quality_filter_adjustment_enabled:
            # CRITICAL FIX: Evaluation episodes may have different behavior than training
            # Use a much higher threshold (10.0) to avoid tightening based on evaluation data
            # Evaluation episodes might have more trades due to different conditions
            # Only tighten if evaluation shows EXTREMELY high trade count
            if trades_per_episode > 10.0:  # Increased from 2.0 to 10.0 to avoid false positives
                old_confidence = self.current_min_action_confidence
                old_quality = self.current_min_quality_score
                
                self.current_min_action_confidence = min(
                    self.adaptive_config.min_action_confidence_range[1],
                    self.current_min_action_confidence + self.adaptive_config.quality_adjustment_rate
                )
                self.current_min_quality_score = min(
                    self.adaptive_config.min_quality_score_range[1],
                    self.current_min_quality_score + (self.adaptive_config.quality_adjustment_rate * 2)
                )
                
                if self.current_min_action_confidence != old_confidence or self.current_min_quality_score != old_quality:
                    adjustments["quality_filters"] = {
                        "min_action_confidence": {
                            "old": old_confidence,
                            "new": self.current_min_action_confidence,
                            "reason": f"Too many trades ({trades_per_episode:.2f}/episode) - tightening filters"
                        },
                        "min_quality_score": {
                            "old": old_quality,
                            "new": self.current_min_quality_score,
                            "reason": f"Too many trades ({trades_per_episode:.2f}/episode) - tightening filters"
                        }
                    }
                    print(f"[ADAPT] Tightened quality filters: confidence {old_confidence:.3f}->{self.current_min_action_confidence:.3f}, quality {old_quality:.3f}->{self.current_min_quality_score:.3f}")
            
            # If no trades, relax filters
            elif trades_per_episode < 0.3:
                old_confidence = self.current_min_action_confidence
                old_quality = self.current_min_quality_score
                
                self.current_min_action_confidence = max(
                    self.adaptive_config.min_action_confidence_range[0],
                    self.current_min_action_confidence - self.adaptive_config.quality_adjustment_rate
                )
                self.current_min_quality_score = max(
                    self.adaptive_config.min_quality_score_range[0],
                    self.current_min_quality_score - (self.adaptive_config.quality_adjustment_rate * 2)
                )
                
                if self.current_min_action_confidence != old_confidence or self.current_min_quality_score != old_quality:
                    adjustments["quality_filters"] = {
                        "min_action_confidence": {
                            "old": old_confidence,
                            "new": self.current_min_action_confidence,
                            "reason": f"Too few trades ({trades_per_episode:.2f}/episode) - relaxing filters"
                        },
                        "min_quality_score": {
                            "old": old_quality,
                            "new": self.current_min_quality_score,
                            "reason": f"Too few trades ({trades_per_episode:.2f}/episode) - relaxing filters"
                        }
                    }
                    print(f"[ADAPT] Relaxed quality filters: confidence {old_confidence:.3f}->{self.current_min_action_confidence:.3f}, quality {old_quality:.3f}->{self.current_min_quality_score:.3f}")
        
        # Check for low performance
        if snapshot.sharpe_ratio < self.adaptive_config.target_sharpe:
            self.consecutive_low_performance += 1
            
            if self.consecutive_low_performance > 3:
                # Reduce learning rate if performance is consistently low
                if self.adaptive_config.lr_adjustment_enabled:
                    old_lr = self.current_learning_rate
                    self.current_learning_rate *= self.adaptive_config.lr_reduction_factor
                    
                    # Update agent learning rates
                    for param_group in agent.actor_optimizer.param_groups:
                        param_group['lr'] = self.current_learning_rate
                    for param_group in agent.critic_optimizer.param_groups:
                        param_group['lr'] = self.current_learning_rate
                    
                    adjustments["learning_rate"] = {
                        "old": old_lr,
                        "new": self.current_learning_rate,
                        "reason": "Performance plateau"
                    }
                    self.consecutive_low_performance = 0
        else:
            self.consecutive_low_performance = 0
        
        # Save adjustment history
        if adjustments:
            self._save_adjustment(timestep=snapshot.timestep, adjustments=adjustments)
        
        return adjustments
    
    def _update_reward_config(self):
        """Update reward config with adaptive parameters"""
        # This will be read by the environment in the next reset
        # We store it in a way the environment can access
        reward_config_path = self.log_dir / "current_reward_config.json"
        with open(reward_config_path, 'w') as f:
            json.dump({
                "inaction_penalty": self.current_inaction_penalty,
                "entropy_coef": self.current_entropy_coef,
                # NEW: Adaptive profitability parameters
                "min_risk_reward_ratio": self.current_min_risk_reward_ratio,
                "quality_filters": {
                    "min_action_confidence": self.current_min_action_confidence,
                    "min_quality_score": self.current_min_quality_score
                }
            }, f, indent=2)
    
    def _is_better(self, new: PerformanceSnapshot, old: PerformanceSnapshot) -> bool:
        """Check if new snapshot is better than old"""
        # Prioritize: trades > sharpe > return
        if new.total_trades == 0 and old.total_trades > 0:
            return False
        if new.total_trades > 0 and old.total_trades == 0:
            return True
        
        # If both have trades, compare Sharpe ratio
        if new.sharpe_ratio > old.sharpe_ratio + 0.1:
            return True
        if old.sharpe_ratio > new.sharpe_ratio + 0.1:
            return False
        
        # If Sharpe is similar, compare return
        return new.total_return > old.total_return
    
    def _calculate_improvement(
        self,
        new: PerformanceSnapshot,
        old: Optional[PerformanceSnapshot]
    ) -> Optional[float]:
        """Calculate improvement percentage"""
        if old is None:
            return None
        
        # Improvement based on multiple factors
        if old.total_trades == 0:
            if new.total_trades > 0:
                return 1.0  # 100% improvement (went from no trades to trades)
            return None
        
        # Weighted improvement
        trade_improvement = (new.total_trades - old.total_trades) / max(1, old.total_trades)
        sharpe_improvement = (new.sharpe_ratio - old.sharpe_ratio) / max(0.1, abs(old.sharpe_ratio))
        return_improvement = (new.total_return - old.total_return) / max(0.01, abs(old.total_return))
        
        # Combined improvement (weighted)
        improvement = (
            0.4 * min(trade_improvement, 1.0) +
            0.3 * min(sharpe_improvement, 1.0) +
            0.3 * min(return_improvement, 1.0)
        )
        
        return max(0, improvement)
    
    def _save_snapshot(self, snapshot: PerformanceSnapshot):
        """Save performance snapshot to file"""
        with open(self.snapshot_file, 'a') as f:
            f.write(json.dumps(asdict(snapshot)) + '\n')
    
    def _save_adjustment(self, timestep: int, adjustments: Dict):
        """Save configuration adjustments"""
        record = {
            "timestep": timestep,
            "timestamp": datetime.now().isoformat(),
            "adjustments": adjustments
        }
        with open(self.config_history_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
    
    def get_current_inaction_penalty(self) -> float:
        """Get current inaction penalty for reward function"""
        return self.current_inaction_penalty
    
    def _save_pause_state(self, timestep: int):
        """Save pause state to file"""
        pause_file = self.log_dir / "training_paused.json"
        with open(pause_file, 'w') as f:
            json.dump({
                "paused": True,
                "reason": self.pause_reason,
                "timestep": timestep,
                "timestamp": datetime.now().isoformat(),
                "checkpoint": f"checkpoint_{timestep}.pt"
            }, f, indent=2)
    
    def is_training_paused(self) -> bool:
        """Check if training is paused"""
        return self.training_paused
    
    def get_pause_reason(self) -> Optional[str]:
        """Get reason for training pause"""
        return self.pause_reason
    
    def get_summary(self) -> Dict:
        """Get summary of adaptive training progress"""
        if not self.performance_history:
            return {"status": "no_evaluations_yet"}
        
        latest = self.performance_history[-1]
        
        return {
            "total_evaluations": len(self.performance_history),
            "latest_timestep": latest.timestep,
            "latest_trades": latest.total_trades,
            "latest_sharpe": latest.sharpe_ratio,
            "best_sharpe": self.best_performance.sharpe_ratio if self.best_performance else 0,
            "current_entropy_coef": self.current_entropy_coef,
            "current_inaction_penalty": self.current_inaction_penalty,
            "trend": self._calculate_trend()
        }
    
    def _calculate_trend(self) -> str:
        """Calculate performance trend"""
        if len(self.performance_history) < 3:
            return "insufficient_data"
        
        recent = self.performance_history[-3:]
        trades_trend = [s.total_trades for s in recent]
        sharpe_trend = [s.sharpe_ratio for s in recent]
        
        if trades_trend[-1] > trades_trend[0] and sharpe_trend[-1] > sharpe_trend[0]:
            return "improving"
        elif trades_trend[-1] < trades_trend[0] and sharpe_trend[-1] < sharpe_trend[0]:
            return "declining"
        else:
            return "stable"

