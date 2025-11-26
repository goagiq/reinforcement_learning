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
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import yaml

from src.model_evaluation import ModelEvaluator, ModelMetrics
from src.quality_scorer import QualityScorer
from src.utils.colors import error, warn


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
    eval_frequency: int = 5000  # Evaluate every N timesteps (default: 5,000 for more frequent checks)
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
    min_rr_threshold: float = 1.5  # Minimum R:R threshold (floor - adaptive learning won't go below this)
    max_rr_threshold: float = 2.5  # Maximum R:R threshold (ceiling - adaptive learning won't go above this)
    rr_adjustment_rate: float = 0.1  # How much to adjust per step
    min_rr_floor: float = 0.7  # Absolute minimum R:R to allow trades (enforcement floor - separate from adaptive floor)
    
    # Quality filter adjustment (NEW)
    quality_filter_adjustment_enabled: bool = True
    min_action_confidence_range: Tuple[float, float] = (0.1, 0.2)  # (min, max)
    min_quality_score_range: Tuple[float, float] = (0.3, 0.5)  # (min, max)
    quality_adjustment_rate: float = 0.01  # How much to adjust per step
    
    # Stop loss adjustment (NEW - adaptive based on volatility and performance)
    stop_loss_adjustment_enabled: bool = True
    min_stop_loss_pct: float = 0.01  # Hard minimum 1.0% (safety floor)
    max_stop_loss_pct: float = 0.03  # Maximum 3.0% (for high volatility)
    stop_loss_adjustment_rate: float = 0.002  # How much to adjust per step (0.2%)
    base_stop_loss_pct: float = 0.015  # Base/starting stop loss (1.5%)


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
        
        # Current adaptive stop loss (NEW - volatility and performance based)
        self.current_stop_loss_pct = self.config["environment"]["reward"].get("stop_loss_pct", self.adaptive_config.base_stop_loss_pct)
        
        # Stop loss tracking for performance-based adjustments
        self.stop_loss_hit_count = 0  # Track how many times stop loss was hit
        self.stop_loss_hit_history = []  # Track recent stop loss hits
        
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
        self.consecutive_negative_episodes = 0  # Track recent episode trend
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
        self._last_adjustment_episode = 0  # Track last adjustment episode (for when timestep is stuck)
        
        # Logging
        self.log_dir = Path("logs/adaptive_training")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_file = self.log_dir / "performance_snapshots.jsonl"
        self.config_history_file = self.log_dir / "config_adjustments.jsonl"
        
        # Initialize adaptive config file with current values
        self._update_reward_config()
    
    def quick_adjust_for_negative_trend(
        self,
        recent_mean_pnl: float,
        recent_win_rate: float,
        agent,
        recent_trades_data: Optional[List[Dict]] = None,  # NEW: Optional recent trades from journal
        recent_total_trades: int = 0  # NEW: Total trades in recent episodes
    ) -> Optional[Dict]:
        """
        Quick adjustment for negative trend (called every episode, not just during evaluation).
        
        This provides faster response to negative trends without waiting for full evaluation.
        ENHANCED: Now more aggressive for losing streaks and uses trade journal data.
        CRITICAL FIX: Don't tighten filters when there are no trades - this creates a feedback loop!
        
        Args:
            recent_mean_pnl: Mean PnL of last 10 episodes
            recent_win_rate: Mean win rate of last 10 episodes
            agent: PPOAgent instance
            recent_trades_data: Optional list of recent trades from journal (for better analysis)
            recent_total_trades: Total number of trades in recent episodes (to detect no-trade condition)
            
        Returns:
            Dict with adjustments made, or None if no adjustments
        """
        adjustments = {}
        
        # CRITICAL FIX: If there are no trades, DON'T tighten filters - this makes the problem worse!
        # Instead, we should relax filters to allow trades to happen
        if recent_total_trades == 0:
            # No trades detected - relax quality filters to encourage trading
            old_confidence = self.current_min_action_confidence
            old_quality = self.current_min_quality_score
            
            # Relax filters (decrease thresholds)
            confidence_relaxation = 0.01  # Reduce by 1%
            quality_relaxation = 0.02  # Reduce by 2%
            
            # Apply relaxation (with floors to prevent going too low)
            min_confidence_floor = 0.05  # Don't go below 5%
            min_quality_floor = 0.1  # Don't go below 10%
            
            self.current_min_action_confidence = max(
                min_confidence_floor,
                self.current_min_action_confidence - confidence_relaxation
            )
            self.current_min_quality_score = max(
                min_quality_floor,
                self.current_min_quality_score - quality_relaxation
            )
            
            # Only make adjustment if values actually changed
            if (self.current_min_action_confidence != old_confidence or 
                self.current_min_quality_score != old_quality):
                adjustments["quality_filters"] = {
                    "min_action_confidence": {
                        "old": old_confidence,
                        "new": self.current_min_action_confidence,
                        "reason": f"RELAXED: No trades detected - encouraging trading (was {old_confidence:.3f})"
                    },
                    "min_quality_score": {
                        "old": old_quality,
                        "new": self.current_min_quality_score,
                        "reason": f"RELAXED: No trades detected - encouraging trading (was {old_quality:.3f})"
                    }
                }
                
                # Log adjustment details
                print(f"\n[ADAPT] RELAXING Quality Filters (No Trades Detected):")
                print(f"   Total Trades: {recent_total_trades}")
                print(f"   Confidence: {old_confidence:.3f} -> {self.current_min_action_confidence:.3f} (-{confidence_relaxation:.3f})")
                print(f"   Quality: {old_quality:.3f} -> {self.current_min_quality_score:.3f} (-{quality_relaxation:.3f})")
                
                # Update reward config to apply changes
                self._update_reward_config()
                return adjustments
        
        # Only adjust if negative trend is significant AND we have trades
        if recent_mean_pnl < 0 and recent_total_trades > 0:
            # Calculate severity of losing streak
            # More negative PnL = more aggressive adjustment
            pnl_severity = abs(recent_mean_pnl) / 100.0  # Normalize (e.g., -$100 = 1.0 severity)
            pnl_severity = min(5.0, pnl_severity)  # Cap at 5.0 (for -$500+ losses)
            
            # Analyze recent trades if available
            avg_loss = None
            avg_win = None
            if recent_trades_data and len(recent_trades_data) >= 5:
                losing_trades = [t for t in recent_trades_data if t.get("pnl", 0) < 0]
                winning_trades = [t for t in recent_trades_data if t.get("pnl", 0) > 0]
                
                if losing_trades:
                    avg_loss = abs(sum(t.get("pnl", 0) for t in losing_trades) / len(losing_trades))
                if winning_trades:
                    avg_win = sum(t.get("pnl", 0) for t in winning_trades) / len(winning_trades)
            
            # Determine adjustment aggressiveness
            # Base adjustment rate
            base_confidence_adj = 0.005
            base_quality_adj = 0.01
            
            # Increase aggressiveness based on:
            # 1. PnL severity (how negative)
            # 2. Win rate (lower = more aggressive)
            # 3. Average loss size (larger = more aggressive)
            
            confidence_multiplier = 1.0 + (pnl_severity * 0.5)  # Up to 3.5x for severe losses
            quality_multiplier = 1.0 + (pnl_severity * 0.5)  # Up to 3.5x for severe losses
            
            # If win rate is low, be more aggressive
            if recent_win_rate < 0.40:
                confidence_multiplier *= 1.5
                quality_multiplier *= 1.5
            
            # If average loss is large, be more aggressive
            if avg_loss and avg_loss > 100:  # Average loss > $100
                loss_severity = min(2.0, avg_loss / 100.0)  # Up to 2x for $200+ avg loss
                confidence_multiplier *= (1.0 + loss_severity * 0.3)
                quality_multiplier *= (1.0 + loss_severity * 0.3)
            
            # Calculate adjustments
            old_confidence = self.current_min_action_confidence
            old_quality = self.current_min_quality_score
            
            confidence_adjustment = base_confidence_adj * confidence_multiplier
            quality_adjustment = base_quality_adj * quality_multiplier
            
            # Apply adjustments (with caps)
            self.current_min_action_confidence = min(0.30, self.current_min_action_confidence + confidence_adjustment)
            self.current_min_quality_score = min(0.70, self.current_min_quality_score + quality_adjustment)
            
            # Only make adjustment if values actually changed
            if (self.current_min_action_confidence != old_confidence or 
                self.current_min_quality_score != old_quality):
                adjustments["quality_filters"] = {
                    "min_action_confidence": {
                        "old": old_confidence,
                        "new": self.current_min_action_confidence,
                        "reason": f"Quick adjustment: negative trend (mean_pnl=${recent_mean_pnl:.2f}, win_rate={recent_win_rate:.1%}, severity={pnl_severity:.2f})"
                    },
                    "min_quality_score": {
                        "old": old_quality,
                        "new": self.current_min_quality_score,
                        "reason": f"Quick adjustment: negative trend (mean_pnl=${recent_mean_pnl:.2f}, win_rate={recent_win_rate:.1%}, severity={pnl_severity:.2f})"
                    }
                }
                
                # Log adjustment details
                print(f"\n[ADAPT] Quick Quality Filter Tightening (Losing Streak):")
                print(f"   Mean PnL: ${recent_mean_pnl:.2f}")
                print(f"   Win Rate: {recent_win_rate:.1%}")
                if avg_loss:
                    print(f"   Avg Loss: ${avg_loss:.2f}")
                if avg_win:
                    print(f"   Avg Win: ${avg_win:.2f}")
                print(f"   Severity: {pnl_severity:.2f}")
                print(f"   Confidence: {old_confidence:.3f} -> {self.current_min_action_confidence:.3f} (+{confidence_adjustment:.3f})")
                print(f"   Quality: {old_quality:.3f} -> {self.current_min_quality_score:.3f} (+{quality_adjustment:.3f})")
                
                # Update reward config to apply changes
                self._update_reward_config()
        
        return adjustments if adjustments else None
    
    def should_evaluate(self, timestep: int) -> bool:
        """Check if we should run an evaluation"""
        # CRITICAL FIX: If timestep is stuck at 0, use episode-based evaluation instead
        # This handles cases where timestep counter isn't working
        if timestep == 0:
            # Use episode-based evaluation: every 10 episodes (approximately equivalent to 5k timesteps)
            episode_frequency = 10
            return (self.consecutive_no_trade_episodes % episode_frequency == 0 and 
                    self.consecutive_no_trade_episodes > 0) or \
                   (hasattr(self, '_last_eval_episode') and 
                    (self.consecutive_no_trade_episodes - getattr(self, '_last_eval_episode', 0)) >= episode_frequency)
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
        # CRITICAL FIX: Bypass timestep checks for persistent no-trade conditions
        # If we have many consecutive no-trade episodes, we need to adjust immediately
        persistent_no_trade_condition = self.consecutive_no_trade_episodes >= 3
        
        # CRITICAL FIX: Use episode-based checks when timesteps are very low (< 1000)
        # This handles cases where timesteps are incrementing but still too low to meet thresholds
        timestep_stuck = timestep == 0
        timestep_too_low = timestep < 1000  # Use episode-based logic when timesteps are very low
        
        # Only check periodically to avoid overhead (unless persistent no-trade condition or timestep issues)
        if not persistent_no_trade_condition and not timestep_stuck and not timestep_too_low:
            if (timestep - self.last_trade_check_timestep) < self.trade_check_frequency:
                return None
        
        self.last_trade_check_timestep = timestep
        
        # Check if we've had enough time since last adjustment (unless persistent no-trade condition or timestep issues)
        if not persistent_no_trade_condition and not timestep_stuck and not timestep_too_low:
            if (timestep - self.last_adjustment_timestep) < self.min_adjustment_interval:
                return None
        
        # CRITICAL FIX: If timestep is stuck or too low, use episode-based interval instead
        # When timesteps are < 1000, allow adjustments every episode (no timestep-based throttling)
        # This is safe because episodes are the only reliable metric we have when timesteps are low
        if timestep_stuck or timestep_too_low:
            # Only prevent if we just made an adjustment in this same episode
            if hasattr(self, '_last_adjustment_episode') and episode == self._last_adjustment_episode:
                return None
            # Otherwise, allow adjustment (episode-based throttling handled by consecutive_no_trade_episodes logic)
        
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
        # CRITICAL FIX: Lower threshold for very short episodes (1 step = 0.0001 progress)
        # If episode is very short (< 100 steps), check immediately
        early_no_trades = no_trades_detected and (episode_progress > 0.1 or current_episode_length < 100)
        
        # Persistent no trades: multiple episodes with no trades
        # CRITICAL FIX: Lower threshold to trigger immediately when we have any consecutive no-trade episodes
        persistent_no_trades = self.consecutive_no_trade_episodes >= 1
        
        if early_no_trades or persistent_no_trades:
            # Only increment counter when we first detect no trades in an episode
            # (counter is incremented when episode completes, not during episode)
            print(warn(f"\n[ADAPTIVE] [WARN] NO TRADES DETECTED"))
            print(f"   Episode: {episode}, Progress: {episode_progress*100:.0f}%, Trades: {current_episode_trades}")
            print(f"   Consecutive no-trade episodes: {self.consecutive_no_trade_episodes}")
            print(f"   Early detection: {early_no_trades}, Persistent: {persistent_no_trades}")
            
            # Intelligent adjustment based on how long we've had no trades
            adjustment_severity = min(self.consecutive_no_trade_episodes, 5)  # Cap at 5x
            
            # CRITICAL FIX: RELAX quality filters when no trades detected (don't tighten them!)
            # This prevents a feedback loop where no trades -> tighten filters -> even fewer trades
            old_confidence = self.current_min_action_confidence
            old_quality = self.current_min_quality_score
            
            # Relax filters more aggressively based on how long we've had no trades
            confidence_relaxation = 0.01 * adjustment_severity  # 1% per episode, up to 5%
            quality_relaxation = 0.02 * adjustment_severity  # 2% per episode, up to 10%
            
            # Apply relaxation (with floors to prevent going too low)
            min_confidence_floor = 0.05  # Don't go below 5%
            min_quality_floor = 0.1  # Don't go below 10%
            
            self.current_min_action_confidence = max(
                min_confidence_floor,
                self.current_min_action_confidence - confidence_relaxation
            )
            self.current_min_quality_score = max(
                min_quality_floor,
                self.current_min_quality_score - quality_relaxation
            )
            
            if (self.current_min_action_confidence != old_confidence or 
                self.current_min_quality_score != old_quality):
                adjustments["quality_filters"] = {
                    "min_action_confidence": {
                        "old": old_confidence,
                        "new": self.current_min_action_confidence,
                        "reason": f"RELAXED: No trades for {self.consecutive_no_trade_episodes} episodes - encouraging trading"
                    },
                    "min_quality_score": {
                        "old": old_quality,
                        "new": self.current_min_quality_score,
                        "reason": f"RELAXED: No trades for {self.consecutive_no_trade_episodes} episodes - encouraging trading"
                    }
                }
                self._update_reward_config()
                print(f"   [ADAPT] RELAXED quality filters: confidence {old_confidence:.3f} -> {self.current_min_action_confidence:.3f}, quality {old_quality:.3f} -> {self.current_min_quality_score:.3f}")
            
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
            # CRITICAL FIX: Always track episode for adjustments (needed when timestep is stuck)
            self._last_adjustment_episode = episode
            self._save_adjustment(timestep=timestep, episode=episode, adjustments=adjustments)
            print(f"   [OK] Adjustments saved to log (timestep={timestep}, episode={episode})")
        
        return adjustments if adjustments else None
    
    def evaluate_and_adapt(
        self,
        model_path: str,
        timestep: int,
        episode: int,
        mean_reward: float,
        agent,  # PPOAgent instance
        policy_loss: Optional[float] = None  # NEW: Policy loss for convergence detection
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
            print(error(f"[ERROR] Evaluation failed: {e}"))
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
        adjustments = self._analyze_and_adjust(snapshot, agent, metrics=metrics, policy_loss=policy_loss)
        
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
        agent,
        metrics: Optional[Any] = None,  # ModelMetrics from evaluation
        policy_loss: Optional[float] = None  # NEW: Policy loss for convergence detection
    ) -> Dict:
        """Analyze performance and make adaptive adjustments"""
        adjustments = {}
        
        # Initialize consecutive negative episodes counter if not exists
        if not hasattr(self, 'consecutive_negative_episodes'):
            self.consecutive_negative_episodes = 0
        
        # Check for no trades
        trades_per_episode = snapshot.total_trades / max(1, self.adaptive_config.eval_episodes)
        
        if trades_per_episode < self.adaptive_config.min_trades_per_episode:
            self.consecutive_no_trade_evals += 1
            print(warn(f"[WARN] LOW TRADE ACTIVITY: {trades_per_episode:.2f} trades/episode"))
            
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
        
        # NEW: Policy Convergence Detection
        # Check if policy has converged (low policy loss) and recent performance is declining
        if policy_loss is not None and policy_loss < 0.001:
            # Policy is converged - check if recent performance is declining
            recent_snapshots = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
            if len(recent_snapshots) >= 5:
                recent_returns = [s.total_return for s in recent_snapshots]
                recent_mean_return = sum(recent_returns) / len(recent_returns)
                
                # If policy converged AND recent trend is negative, increase exploration
                if recent_mean_return < 0:
                    old_entropy = self.current_entropy_coef
                    # Increase entropy more aggressively (3x normal rate) when policy converged
                    self.current_entropy_coef = min(
                        self.adaptive_config.max_entropy_coef,
                        self.current_entropy_coef + (self.adaptive_config.entropy_adjustment_rate * 3)
                    )
                    if self.current_entropy_coef != old_entropy:
                        agent.entropy_coef = self.current_entropy_coef
                        adjustments["entropy_coef"] = {
                            "old": old_entropy,
                            "new": self.current_entropy_coef,
                            "reason": f"Policy converged (loss={policy_loss:.4f}) + negative trend (mean={recent_mean_return:.2%}) - increasing exploration"
                        }
                        print(f"[ADAPT] Policy converged + negative trend: entropy {old_entropy:.4f} -> {self.current_entropy_coef:.4f} (policy_loss={policy_loss:.4f}, recent_mean={recent_mean_return:.2%})")
        
        # NEW: Recent Episode Trend Tracking
        # Check recent episode trend (last 10-20 episodes)
        recent_snapshots = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        if len(recent_snapshots) >= 10:
            recent_returns = [s.total_return for s in recent_snapshots]
            recent_mean_return = sum(recent_returns) / len(recent_returns)
            
            # If recent trend is negative for 10+ episodes
            if recent_mean_return < 0:
                self.consecutive_negative_episodes = getattr(self, 'consecutive_negative_episodes', 0) + 1
                
                if self.consecutive_negative_episodes >= 10:
                    # Negative trend for 10+ episodes - take action
                    print(warn(f"\n[WARN] Negative trend detected: {self.consecutive_negative_episodes} consecutive negative episodes"))
                    print(f"   Recent mean return: {recent_mean_return:.2%}")
                    print(f"   Recent mean PnL: ${recent_mean_return * 100000:.2f} (estimated)")
                    
                    # Tighten quality filters
                    old_confidence = self.current_min_action_confidence
                    old_quality = self.current_min_quality_score
                    self.current_min_action_confidence = min(0.25, self.current_min_action_confidence + 0.02)
                    self.current_min_quality_score = min(0.60, self.current_min_quality_score + 0.05)
                    
                    if "quality_filters" not in adjustments:
                        adjustments["quality_filters"] = {}
                    
                    adjustments["quality_filters"]["min_action_confidence"] = {
                        "old": old_confidence,
                        "new": self.current_min_action_confidence,
                        "reason": f"Negative trend for {self.consecutive_negative_episodes} episodes (mean={recent_mean_return:.2%})"
                    }
                    adjustments["quality_filters"]["min_quality_score"] = {
                        "old": old_quality,
                        "new": self.current_min_quality_score,
                        "reason": f"Negative trend for {self.consecutive_negative_episodes} episodes (mean={recent_mean_return:.2%})"
                    }
                    print(f"[ADAPT] Tightening filters due to negative trend: confidence {old_confidence:.3f}->{self.current_min_action_confidence:.3f}, "
                          f"quality {old_quality:.3f}->{self.current_min_quality_score:.3f}")
            else:
                # Reset counter if trend is positive
                if self.consecutive_negative_episodes > 0:
                    print(f"[OK] Negative trend reversed! Resetting counter (was {self.consecutive_negative_episodes}, recent_mean={recent_mean_return:.2%})")
                self.consecutive_negative_episodes = 0
        
        # Check win rate profitability (NEW)
        profitability_check = self.check_win_rate_profitability(
            total_trades=snapshot.total_trades,
            winning_trades=int(snapshot.total_trades * snapshot.win_rate),
            winning_pnls=self.winning_pnls,
            losing_pnls=self.losing_pnls,
            commission_rate=0.0003  # 0.03%
        )
        
        # IMPROVED: Check profitability first, then win rate
        # Calculate profit factor and R:R ratio
        avg_win = profitability_check.get("avg_win", 0)
        avg_loss = profitability_check.get("avg_loss", 0)
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0.0
        current_rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        
        # Check if profitable (profit factor > 1.0 or expected value > 0)
        is_profitable = profitability_check["is_profitable"]
        win_rate_low = snapshot.win_rate < profitability_check["breakeven_win_rate"]
        
        # AGGRESSIVE profitability detection and response
        # Only tighten aggressively if unprofitable
        if not is_profitable:
            self.consecutive_low_win_rate += 1
            
            print(f"\n[CRITICAL] UNPROFITABLE DETECTED:")
            print(f"   Current win rate: {profitability_check['current_win_rate']:.1%}")
            print(f"   Breakeven win rate: {profitability_check['breakeven_win_rate']:.1%}")
            print(f"   Expected value: ${profitability_check['expected_value']:.2f}")
            print(f"   Profit factor: {profit_factor:.2f}")
            print(f"   R:R ratio: {current_rr_ratio:.2f}:1")
            print(f"   Consecutive unprofitable evaluations: {self.consecutive_low_win_rate}")
            
            # CRITICAL FIX: 0% win rate requires EXPLORATION, not tightening
            # If win rate is 0% or extremely low (but we have trades), model is stuck
            # Need to INCREASE exploration to break out of local minimum
            if snapshot.win_rate == 0.0 and snapshot.total_trades >= 10:
                print(f"\n[CRITICAL] 0% WIN RATE DETECTED! Model is stuck - increasing exploration...")
                old_confidence = self.current_min_action_confidence
                old_quality = self.current_min_quality_score
                old_rr = self.current_min_risk_reward_ratio
                old_entropy = self.current_entropy_coef
                old_penalty = self.current_inaction_penalty
                
                # INCREASE entropy to encourage exploration (opposite of tightening)
                self.current_entropy_coef = min(
                    self.adaptive_config.max_entropy_coef,
                    self.current_entropy_coef * 1.5  # Increase by 50% to break out of local minimum
                )
                agent.entropy_coef = self.current_entropy_coef
                
                # INCREASE inaction penalty to encourage more trading activity
                self.current_inaction_penalty = min(
                    self.adaptive_config.inaction_penalty_max,
                    self.current_inaction_penalty * 2.0  # Double it to encourage trading
                )
                
                # SLIGHTLY relax filters (don't tighten more - we need diversity)
                self.current_min_action_confidence = max(0.15,
                    self.current_min_action_confidence * 0.95)  # Relax by 5%
                self.current_min_quality_score = max(0.35,
                    self.current_min_quality_score * 0.95)  # Relax by 5%
                
                # Keep R:R reasonable but don't increase it more
                # (already high enough, increasing would make problem worse)
                
                adjustments["entropy_coef"] = {
                    "old": old_entropy,
                    "new": self.current_entropy_coef,
                    "reason": f"CRITICAL: 0% win rate - increasing exploration to break out of local minimum"
                }
                adjustments["inaction_penalty"] = {
                    "old": old_penalty,
                    "new": self.current_inaction_penalty,
                    "reason": f"CRITICAL: 0% win rate - encouraging more trading activity"
                }
                adjustments["quality_filters"] = {
                    "min_action_confidence": {
                        "old": old_confidence,
                        "new": self.current_min_action_confidence,
                        "reason": f"CRITICAL: 0% win rate - slightly relaxing to allow more diverse trades"
                    },
                    "min_quality_score": {
                        "old": old_quality,
                        "new": self.current_min_quality_score,
                        "reason": f"CRITICAL: 0% win rate - slightly relaxing to allow more diverse trades"
                    }
                }
                
                self._update_reward_config()
                print(f"[ADAPT] CRITICAL 0% WIN RATE RESPONSE:")
                print(f"   Entropy: {old_entropy:.4f} -> {self.current_entropy_coef:.4f} (+{(self.current_entropy_coef/old_entropy - 1)*100:.0f}%)")
                print(f"   Inaction Penalty: {old_penalty:.6f} -> {self.current_inaction_penalty:.6f} (+{(self.current_inaction_penalty/old_penalty - 1)*100:.0f}%)")
                print(f"   Confidence: {old_confidence:.3f} -> {self.current_min_action_confidence:.3f} (relaxed)")
                print(f"   Quality: {old_quality:.3f} -> {self.current_min_quality_score:.3f} (relaxed)")
            
            # AGGRESSIVE tightening when unprofitable (but NOT 0% win rate)
            elif self.consecutive_low_win_rate >= 2 and snapshot.win_rate > 0.0:
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
                    max(1.5, self.current_min_risk_reward_ratio + 0.5))  # Ensure minimum 1.5
                # Reduce exploration to be more selective (ONLY if not 0% win rate)
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
        elif win_rate_low and current_rr_ratio < 1.5:
            # Profitable but win rate low AND R:R not compensating - tighten slightly
            print(warn(f"\n[WARN] Profitable but low win rate ({snapshot.win_rate:.1%}) with poor R:R ({current_rr_ratio:.2f}:1)"))
            print(f"   Profit factor: {profit_factor:.2f} (profitable)")
            print(f"   Tightening filters slightly to improve win rate")
            
            old_confidence = self.current_min_action_confidence
            old_quality = self.current_min_quality_score
            self.current_min_action_confidence = min(0.25, self.current_min_action_confidence + 0.01)
            self.current_min_quality_score = min(0.60, self.current_min_quality_score + 0.02)
            
            if "quality_filters" not in adjustments:
                adjustments["quality_filters"] = {}
            
            adjustments["quality_filters"]["min_action_confidence"] = {
                "old": old_confidence,
                "new": self.current_min_action_confidence,
                "reason": f"Profitable but low win rate ({snapshot.win_rate:.1%}) with poor R:R ({current_rr_ratio:.2f}:1)"
            }
            adjustments["quality_filters"]["min_quality_score"] = {
                "old": old_quality,
                "new": self.current_min_quality_score,
                "reason": f"Profitable but low win rate ({snapshot.win_rate:.1%}) with poor R:R ({current_rr_ratio:.2f}:1)"
            }
            print(f"[ADAPT] Slight tightening: confidence {old_confidence:.3f}->{self.current_min_action_confidence:.3f}, "
                  f"quality {old_quality:.3f}->{self.current_min_quality_score:.3f}")
        else:
            # Reset counter when profitable
            if self.consecutive_low_win_rate > 0:
                print(f"[OK] Profitability restored! Resetting consecutive low win rate counter (was {self.consecutive_low_win_rate})")
                if current_rr_ratio >= 1.5:
                    print(f"   R:R ratio ({current_rr_ratio:.2f}:1) is compensating for low win rate - maintaining filters")
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
            # FIX: Ensure minimum is 1.5 (user requirement) - don't go below 1.5
            if current_rr_ratio < 1.5:
                old_rr_threshold = self.current_min_risk_reward_ratio
                self.current_min_risk_reward_ratio = min(
                    self.adaptive_config.max_rr_threshold,
                    max(1.5, self.current_min_risk_reward_ratio + self.adaptive_config.rr_adjustment_rate)  # Ensure min 1.5
                )
                if self.current_min_risk_reward_ratio != old_rr_threshold:
                    adjustments["min_risk_reward_ratio"] = {
                        "old": old_rr_threshold,
                        "new": self.current_min_risk_reward_ratio,
                        "reason": f"Poor R:R ratio ({current_rr_ratio:.2f}:1) - tightening threshold"
                    }
                    print(f"[ADAPT] Tightened R:R threshold: {old_rr_threshold:.2f} -> {self.current_min_risk_reward_ratio:.2f} (current R:R: {current_rr_ratio:.2f}:1)")
            
            # If very profitable (R:R >= 2.0), can relax slightly
            # FIX: Ensure minimum is 1.5 (user requirement)
            elif current_rr_ratio >= 2.0:
                old_rr_threshold = self.current_min_risk_reward_ratio
                new_rr = self.current_min_risk_reward_ratio - (self.adaptive_config.rr_adjustment_rate * 0.5)  # Relax slower
                self.current_min_risk_reward_ratio = max(
                    1.5,  # Hard minimum of 1.5 (user requirement)
                    self.adaptive_config.min_rr_threshold,
                    new_rr
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
        
        # NEW: Adaptive Stop Loss Adjustment (volatility and performance based)
        if self.adaptive_config.stop_loss_adjustment_enabled and metrics:
            try:
                # Get volatility data if available from evaluator's test data
                volatility_multiplier = 1.0
                volatility_percentile = 50.0  # Default to median
                
                try:
                    from src.volatility_predictor import VolatilityPredictor
                    
                    # Try to get price data from evaluator's test data
                    if hasattr(self.evaluator, 'test_data') and self.evaluator.test_data:
                        instrument = self.config["environment"]["instrument"]
                        timeframes = self.config["environment"]["timeframes"]
                        primary_tf = min(timeframes)
                        
                        if primary_tf in self.evaluator.test_data and len(self.evaluator.test_data[primary_tf]) > 20:
                            price_data = self.evaluator.test_data[primary_tf]
                            volatility_predictor = VolatilityPredictor(lookback_periods=252)
                            volatility_forecast = volatility_predictor.predict_volatility(price_data, method="adaptive")
                            volatility_multiplier = volatility_predictor.get_adaptive_stop_loss_multiplier(
                                self.current_stop_loss_pct, volatility_forecast
                            )
                            volatility_percentile = volatility_forecast.volatility_percentile
                except Exception as e:
                    # If volatility prediction fails, use default
                    print(warn(f"[WARN] Volatility prediction failed for stop loss adjustment: {e}"))
                
                # Performance-based adjustments
                # If avg_loss is close to stop_loss_pct * capital, we're hitting stops too frequently
                initial_capital = self.config.get("risk_management", {}).get("initial_capital", 100000.0)
                estimated_stop_loss_amount = initial_capital * self.current_stop_loss_pct
                
                # Check if we have trade statistics
                avg_loss = getattr(metrics, 'average_loss', 0.0)
                avg_win = getattr(metrics, 'average_win', 0.0)
                total_trades = snapshot.total_trades
                
                # Calculate performance-based adjustment
                performance_adjustment = 0.0
                adjustment_reason = ""
                
                # FIX: Add drawdown-based stop loss adjustment
                # If drawdown is high, tighten stop loss to limit further losses
                max_dd = 0.0
                if metrics and hasattr(metrics, 'max_drawdown'):
                    max_dd = getattr(metrics, 'max_drawdown', 0.0)
                elif hasattr(snapshot, 'max_drawdown'):
                    max_dd = getattr(snapshot, 'max_drawdown', 0.0)
                
                if max_dd > 0.10:  # Drawdown > 10%
                    # Aggressively tighten stop loss
                    performance_adjustment -= 0.005  # Reduce by 0.5%
                    adjustment_reason = f"High drawdown ({max_dd*100:.1f}%) - tightening stop loss to limit losses"
                    print(warn(f"[ADAPT] High drawdown detected ({max_dd*100:.1f}%) - tightening stop loss"))
                elif max_dd > 0.08:  # Drawdown > 8%
                    # Moderately tighten stop loss
                    performance_adjustment -= 0.003  # Reduce by 0.3%
                    adjustment_reason = f"Elevated drawdown ({max_dd*100:.1f}%) - tightening stop loss"
                    print(warn(f"[ADAPT] Elevated drawdown ({max_dd*100:.1f}%) - tightening stop loss"))
                elif max_dd > 0.05:  # Drawdown > 5%
                    # Slightly tighten stop loss
                    performance_adjustment -= 0.001  # Reduce by 0.1%
                    adjustment_reason = f"Moderate drawdown ({max_dd*100:.1f}%) - slight stop loss tightening"
                
                # If avg_loss is consistently at stop loss threshold, stop may be too tight
                if avg_loss > 0 and total_trades >= 10:
                    loss_ratio = avg_loss / estimated_stop_loss_amount if estimated_stop_loss_amount > 0 else 1.0
                    
                    # If losses are consistently at stop loss level (90-110%), we may be hitting stop too often
                    if 0.90 <= loss_ratio <= 1.10:
                        # Likely hitting stop loss frequently - may need wider stops in high vol, or tighter if low vol
                        if volatility_percentile > 70:
                            # High volatility - widen stops
                            performance_adjustment = 0.003  # Increase by 0.3%
                            adjustment_reason = f"High volatility ({volatility_percentile:.0f}th percentile) and frequent stop hits - widening stops"
                        elif volatility_percentile < 30:
                            # Low volatility - stops may be too tight, but don't widen much
                            performance_adjustment = 0.001  # Small increase
                            adjustment_reason = f"Low volatility ({volatility_percentile:.0f}th percentile) but frequent stop hits - slight widening"
                    # If avg_loss is significantly less than stop loss, stops may be too wide
                    elif loss_ratio < 0.7 and volatility_percentile < 40:
                        # Low volatility and losses are small - can tighten stops
                        performance_adjustment = -0.002  # Decrease by 0.2%
                        adjustment_reason = f"Low volatility ({volatility_percentile:.0f}th percentile) and small losses - tightening stops"
                
                # Combine volatility and performance adjustments
                # Start with current stop loss, apply volatility multiplier, then performance adjustment
                vol_based_stop = self.current_stop_loss_pct * volatility_multiplier
                
                # Apply performance adjustment
                new_stop_loss = vol_based_stop + performance_adjustment
                
                # Enforce hard minimum and maximum
                new_stop_loss = max(
                    self.adaptive_config.min_stop_loss_pct,
                    min(self.adaptive_config.max_stop_loss_pct, new_stop_loss)
                )
                
                # Only adjust if change is significant (at least 0.001 = 0.1%)
                if abs(new_stop_loss - self.current_stop_loss_pct) >= 0.001:
                    old_stop_loss = self.current_stop_loss_pct
                    self.current_stop_loss_pct = new_stop_loss
                    
                    adjustments["stop_loss_pct"] = {
                        "old": old_stop_loss,
                        "new": self.current_stop_loss_pct,
                        "volatility_multiplier": volatility_multiplier,
                        "volatility_percentile": volatility_percentile,
                        "performance_adjustment": performance_adjustment,
                        "reason": adjustment_reason or f"Volatility-based adjustment (volatility: {volatility_percentile:.0f}th percentile, multiplier: {volatility_multiplier:.2f}x)"
                    }
                    print(f"[ADAPT] Adjusted stop loss: {old_stop_loss:.3f} ({old_stop_loss*100:.1f}%) -> {self.current_stop_loss_pct:.3f} ({self.current_stop_loss_pct*100:.1f}%)")
                    print(f"        Volatility: {volatility_percentile:.0f}th percentile, Multiplier: {volatility_multiplier:.2f}x")
                    if adjustment_reason:
                        print(f"        Reason: {adjustment_reason}")
                    
                    # Update reward config with new stop loss
                    self._update_reward_config()
            except Exception as e:
                print(warn(f"[WARN] Stop loss adjustment failed: {e}"))
                import traceback
                traceback.print_exc()
        
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
                },
                # NEW: Adaptive stop loss
                "stop_loss_pct": self.current_stop_loss_pct,
                # NEW: Enforcement floor (absolute minimum R:R to allow trades)
                "min_rr_floor": self.adaptive_config.min_rr_floor
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
    
    def _save_adjustment(self, timestep: int, adjustments: Dict, episode: Optional[int] = None):
        """Save configuration adjustments"""
        record = {
            "timestep": timestep,
            "timestamp": datetime.now().isoformat(),
            "adjustments": adjustments
        }
        if episode is not None:
            record["episode"] = episode
        # Always include episode if available (critical for filtering when timestep is stuck at 0)
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

