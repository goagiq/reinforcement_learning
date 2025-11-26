"""
Trading Journal Integration - Non-intrusive service that connects trainer to journal

This service runs in the background and:
1. Reads data from trainer (non-blocking)
2. Writes to trading journal (async)
3. Captures equity curve points
4. Logs episode summaries
"""

import threading
import time
from typing import Optional, Dict, Any
from datetime import datetime
from src.trading_journal import get_journal, TradingJournal


class JournalIntegration:
    """
    Non-intrusive integration between trainer and trading journal.
    
    Runs in background thread, reads trainer data periodically,
    and writes to journal without blocking training.
    """
    
    def __init__(self):
        self.journal = get_journal()
        self.running = False
        self.background_thread: Optional[threading.Thread] = None
        self.trainer = None
        self.last_episode = -1
        self.last_equity_log_step = {}
        
    def start(self, trainer):
        """Start background monitoring of trainer"""
        self.trainer = trainer
        self.running = True
        self.background_thread = threading.Thread(
            target=self._monitor_trainer,
            daemon=True,
            name="JournalIntegration"
        )
        self.background_thread.start()
        print("[OK] Trading Journal Integration started")
    
    def stop(self):
        """Stop background monitoring"""
        self.running = False
        if self.background_thread:
            self.background_thread.join(timeout=2.0)
        print("[OK] Trading Journal Integration stopped")
    
    def _monitor_trainer(self):
        """Background thread that monitors trainer and logs to journal"""
        while self.running:
            try:
                if self.trainer:
                    self._process_episodes()
                    self._process_equity_curve()
                
                # Sleep 1 second between checks (non-intrusive)
                time.sleep(1.0)
            except Exception as e:
                print(f"[ERROR] Journal integration error: {e}")
                time.sleep(5.0)
    
    def _process_episodes(self):
        """Process completed episodes"""
        if not hasattr(self.trainer, 'episode'):
            return
        
        current_episode = self.trainer.episode
        
        # Check for new completed episodes
        if current_episode > self.last_episode:
            # Process episodes that completed since last check
            for episode_num in range(self.last_episode + 1, current_episode + 1):
                self._log_episode_summary(episode_num)
            
            self.last_episode = current_episode
    
    def _log_episode_summary(self, episode_num: int):
        """Log summary for a completed episode"""
        try:
            # Get episode data from trainer
            if not hasattr(self.trainer, 'episode_pnls'):
                return
            
            if episode_num >= len(self.trainer.episode_pnls):
                return  # Episode not completed yet
            
            episode_pnl = self.trainer.episode_pnls[episode_num] if episode_num < len(self.trainer.episode_pnls) else 0.0
            episode_trades = self.trainer.episode_trades[episode_num] if episode_num < len(self.trainer.episode_trades) else 0
            episode_equity = self.trainer.episode_equities[episode_num] if episode_num < len(self.trainer.episode_equities) else 0.0
            episode_win_rate = self.trainer.episode_win_rates[episode_num] if episode_num < len(self.trainer.episode_win_rates) else 0.0
            episode_max_dd = self.trainer.episode_max_drawdowns[episode_num] if episode_num < len(self.trainer.episode_max_drawdowns) else 0.0
            
            # Log episode summary
            self.journal.log_episode_summary(
                episode_number=episode_num,
                start_timestamp=datetime.now().isoformat(),  # Approximate
                end_timestamp=datetime.now().isoformat(),
                total_trades=episode_trades,
                total_pnl=episode_pnl,
                final_equity=episode_equity,
                max_drawdown=episode_max_dd,
                win_rate=episode_win_rate
            )
        except Exception as e:
            pass  # Don't let logging break training
    
    def _process_equity_curve(self):
        """Process equity curve from environment"""
        try:
            if not hasattr(self.trainer, 'env'):
                return
            
            env = self.trainer.env
            if not hasattr(env, 'equity_curve'):
                return
            
            current_episode = getattr(self.trainer, 'episode', 0)
            current_step = getattr(env, 'current_step', 0)
            
            # Log equity points periodically (every 100 steps to reduce overhead)
            last_logged = self.last_equity_log_step.get(current_episode, -1)
            if current_step - last_logged >= 100 and len(env.equity_curve) > 0:
                equity = env.equity_curve[-1]
                cumulative_pnl = getattr(env.state, 'total_pnl', 0.0) if hasattr(env, 'state') and env.state else 0.0
                
                self.journal.log_equity_point(
                    episode=current_episode,
                    step=current_step,
                    equity=equity,
                    cumulative_pnl=cumulative_pnl
                )
                
                self.last_equity_log_step[current_episode] = current_step
        except Exception:
            pass  # Don't let logging break training
    
    def setup_trade_callback(self, env):
        """Setup trade callback in environment"""
        def trade_callback(episode, step, entry_price, exit_price, position_size, pnl, commission, entry_step=None, action_confidence=None):
            """Callback function for trade logging"""
            try:
                # Determine strategy (RL only during training, but can be enhanced)
                # During training, DecisionGate may be used but swarm is disabled
                strategy = "RL"
                # FIXED: Use actual action confidence if provided, otherwise fallback to position size
                if action_confidence is not None:
                    strategy_confidence = float(action_confidence)
                else:
                    strategy_confidence = abs(position_size)  # Fallback: Use position size as proxy
                
                # If DecisionGate was used, we could detect that here
                # For now, all training trades are "RL" strategy
                
                # Calculate duration
                duration_steps = step - entry_step if entry_step is not None else 0
                
                # Calculate commission if not provided (fallback)
                if commission == 0.0 and position_size != 0:
                    # Estimate commission (0.03% of position value)
                    estimated_commission = abs(position_size) * 100000 * 0.0003  # Using initial capital estimate
                else:
                    estimated_commission = commission
                
                # Log trade
                self.journal.log_trade(
                    episode=episode,
                    step=step,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    position_size=position_size,
                    pnl=pnl,
                    commission=estimated_commission,
                    strategy=strategy,
                    strategy_confidence=strategy_confidence,
                    entry_timestamp=None,  # Could be enhanced
                    exit_timestamp=datetime.now().isoformat(),
                    market_conditions=None,  # Could be enhanced
                    decision_metadata=None  # Could be enhanced
                )
            except Exception:
                pass  # Don't let logging break training
        
        # Add equity logging method to callback
        def log_equity(episode, step, equity, cumulative_pnl):
            try:
                self.journal.log_equity_point(
                    episode=episode,
                    step=step,
                    equity=equity,
                    cumulative_pnl=cumulative_pnl
                )
            except Exception:
                pass
        
        trade_callback.log_equity = log_equity
        
        # Set callback in environment
        env.trade_callback = trade_callback
        env._current_episode = getattr(self.trainer, 'episode', 0)


# Global integration instance
_global_integration: Optional[JournalIntegration] = None


def get_integration() -> JournalIntegration:
    """Get global journal integration instance"""
    global _global_integration
    if _global_integration is None:
        _global_integration = JournalIntegration()
    return _global_integration

