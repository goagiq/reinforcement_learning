"""
Live Trading Execution Module

Coordinates between NT8, RL agent, and reasoning engine for live/paper trading.

Usage:
    python src/live_trading.py --config configs/train_config.yaml --model models/best_model.pt
"""

import argparse
import csv
import yaml
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import time
import json
import threading
import math
from collections import deque
from typing import Dict, Optional, List, Any

from src.data_extraction import DataExtractor, MarketBar
from src.trading_env import TradingEnvironment
from src.rl_agent import PPOAgent
from src.nt8_bridge_server import NT8BridgeServer
from src.reasoning_engine import ReasoningEngine, MarketState, RLRecommendation, TradeAction
from src.risk_manager import RiskManager
from src.drift_monitor import DriftMonitor, TradeMetrics
from src.decision_gate import DecisionGate, DecisionResult
from src.agentic_swarm import SwarmOrchestrator
from src.trading_hours import TradingHoursManager
from src.signal_calculator import SignalCalculator
import asyncio


class MultiTimeframeResampler:
    """
    Resample a base timeframe stream (e.g., 1-minute bars) into higher timeframes
    for live trading inference.
    """

    def __init__(self, base_timeframe: int, target_timeframes: List[int]):
        self.base_timeframe = base_timeframe
        self.target_timeframes = sorted(
            tf for tf in target_timeframes if tf > base_timeframe
        )
        self.window_sizes = {
            tf: max(1, math.ceil(tf / base_timeframe)) for tf in self.target_timeframes
        }
        self.buffers = {
            tf: deque(maxlen=self.window_sizes[tf]) for tf in self.target_timeframes
        }
        self.latest_complete: Dict[int, Optional[MarketBar]] = {
            tf: None for tf in self.target_timeframes
        }
        self.last_timestamp: Optional[datetime] = None

    def update(self, bar: MarketBar) -> Dict[int, Optional[MarketBar]]:
        """Update resampler with the latest base timeframe bar."""
        if self.last_timestamp:
            minutes_diff = (bar.timestamp - self.last_timestamp).total_seconds() / 60.0
            # Reset buffers if we detect a gap larger than expected (session break, etc.)
            if minutes_diff > self.base_timeframe * 1.5:
                for buffer in self.buffers.values():
                    buffer.clear()
        self.last_timestamp = bar.timestamp

        result: Dict[int, Optional[MarketBar]] = {self.base_timeframe: bar}

        for tf in self.target_timeframes:
            window_size = self.window_sizes[tf]
            buffer = self.buffers[tf]
            buffer.append(bar)

            if len(buffer) == window_size and self._is_boundary(bar.timestamp, tf):
                aggregated = self._aggregate(list(buffer))
                self.latest_complete[tf] = aggregated

        for tf in self.target_timeframes:
            result[tf] = self.latest_complete[tf]

        return result

    @staticmethod
    def _is_boundary(timestamp: datetime, timeframe: int) -> bool:
        total_minutes = timestamp.hour * 60 + timestamp.minute
        return (total_minutes % timeframe) == 0

    @staticmethod
    def _aggregate(bars: List[MarketBar]) -> MarketBar:
        return MarketBar(
            timestamp=bars[-1].timestamp,
            open=bars[0].open,
            high=max(bar.high for bar in bars),
            low=min(bar.low for bar in bars),
            close=bars[-1].close,
            volume=sum(bar.volume for bar in bars)
        )


class LiveTradingSystem:
    """
    Main live trading system that coordinates:
    - NT8 market data reception
    - RL agent decision making
    - Reasoning engine validation
    - Risk management
    - Order execution
    """
    
    def __init__(self, config: dict, model_path: str):
        self.config = config
        self.model_path = model_path
        self.running = False
        
        # Load agent
        print(f"Loading RL agent from: {model_path}")
        self.agent = PPOAgent(
            state_dim=config["environment"]["state_features"],
            action_range=tuple(config["environment"]["action_range"]),
            device="cpu"  # CPU for inference (faster startup)
        )
        self.agent.load(model_path)
        self.agent.actor.eval()  # Set to evaluation mode
        self.agent.critic.eval()
        
        # Initialize reasoning engine (if enabled)
        if config.get("reasoning", {}).get("enabled", True):
            print("Initializing reasoning engine...")
            reasoning_config = config.get("reasoning", {})
            
            # Get API key from environment variable if not in config
            import os
            api_key = reasoning_config.get("api_key") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("GROK_API_KEY")
            
            # Kong Gateway configuration
            use_kong = reasoning_config.get("use_kong", False)
            kong_api_key = reasoning_config.get("kong_api_key") or os.getenv("KONG_API_KEY")
            
            self.reasoning_engine = ReasoningEngine(
                provider_type=reasoning_config.get("provider", "ollama"),
                model=reasoning_config.get("model", "deepseek-r1:8b"),
                api_key=api_key,
                base_url=reasoning_config.get("base_url"),
                timeout=int(reasoning_config.get("timeout", 2.0) * 60),  # Convert to seconds
                keep_alive=reasoning_config.get("keep_alive", "10m"),  # Keep model pre-loaded
                use_kong=use_kong,
                kong_api_key=kong_api_key
            )
            self.reasoning_enabled = True
        else:
            self.reasoning_engine = None
            self.reasoning_enabled = False
        
        # Initialize risk manager
        print("Initializing risk manager...")
        self.risk_manager = RiskManager(config["risk_management"])
        
        # Initialize decision gate
        decision_gate_config = config.get("decision_gate", {
            "rl_weight": 0.6,
            "swarm_weight": 0.4,
            "min_combined_confidence": 0.7,
            "conflict_reduction_factor": 0.5,
            "swarm_enabled": True,
            "swarm_timeout": 20.0,
            "fallback_to_rl_only": True
        })
        self.decision_gate = DecisionGate(decision_gate_config)
        
        # Initialize swarm orchestrator (if enabled)
        swarm_config = config.get("agentic_swarm", {})
        if swarm_config.get("enabled", True):
            print("Initializing swarm orchestrator...")
            try:
                self.swarm_orchestrator = SwarmOrchestrator(
                    config=config,
                    reasoning_engine=self.reasoning_engine if self.reasoning_enabled else None,
                    risk_manager=self.risk_manager
                )
                self.swarm_enabled = True
                print("✅ Swarm orchestrator enabled")
            except Exception as e:
                print(f"⚠️  Failed to initialize swarm orchestrator: {e}")
                print("   Falling back to RL-only mode")
                self.swarm_orchestrator = None
                self.swarm_enabled = False
        else:
            self.swarm_orchestrator = None
            self.swarm_enabled = False
        
        # Manual approval tracking
        self.manual_approval_enabled = config.get("manual_approval", {}).get("enabled", False)
        self.pending_approvals = []  # Queue of pending trades awaiting approval
        
        # State tracking
        self.current_state = None
        self.current_position = 0.0
        self.equity_history = []
        self.trade_history = []
        
        # Performance tracking for adaptive learning
        self.performance_start_time = datetime.now()
        self.winning_trades = []
        self.losing_trades = []
        self.max_equity = config.get("environment", {}).get("initial_capital", 100000.0)
        self.max_drawdown = 0.0
        env_config = config.get("environment", {})
        self.instrument = env_config.get("instrument", "default")
        logging_config = config.get("logging", {})
        log_dir = Path(logging_config.get("log_dir", "logs"))
        log_dir.mkdir(exist_ok=True)
        self.decision_log_path = log_dir / "decision_gate_debug.csv"
        if not self.decision_log_path.exists():
            try:
                with open(self.decision_log_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp",
                        "rl_action",
                        "rl_confidence",
                        "decision_action",
                        "decision_confidence",
                        "scale_factor",
                        "confluence_count",
                        "confluence_score",
                        "agreement",
                        "final_action",
                        "current_position",
                        "break_even_active",
                        "avg_entry",
                        "protected_size",
                        "trail_price",
                        "confluence_signals",
                        "risk_state"
                    ])
            except Exception as e:
                print(f"⚠️  Could not initialize decision log: {e}")
        trading_hours_cfg = env_config.get("trading_hours", {})
        self.trading_hours: Optional[TradingHoursManager] = None
        if trading_hours_cfg.get("enabled"):
            self.trading_hours = TradingHoursManager.from_dict(trading_hours_cfg)
        self.timeframes = sorted(env_config.get("timeframes", [1]))
        self.primary_timeframe = self.timeframes[0]
        higher_timeframes = [tf for tf in self.timeframes if tf > self.primary_timeframe]
        if higher_timeframes:
            if self.primary_timeframe != 1:
                print(
                    f"⚠️  Resampler assumes 1-minute base data but primary timeframe is {self.primary_timeframe}."
                    " Ensure NT8 stream matches the smallest timeframe."
                )
            self.multi_timeframe_resampler = MultiTimeframeResampler(
                base_timeframe=self.primary_timeframe,
                target_timeframes=self.timeframes
            )
        else:
            self.multi_timeframe_resampler = None
        self.latest_multi_tf_bars: Optional[Dict[int, MarketBar]] = None
        
        # Statistics
        self.stats = {
            "trades_executed": 0,
            "trades_rejected": 0,
            "total_pnl": 0.0,
            "reasoning_agreements": 0,
            "reasoning_disagreements": 0,
        }
        
        # Initialize drift monitor (if enabled)
        drift_config = config.get("drift_detection", {})
        if drift_config.get("enabled", True):
            baseline_metrics = drift_config.get("baseline_metrics", {
                "win_rate": 0.55,
                "sharpe_ratio": 1.0,
                "profit_factor": 1.5,
                "max_drawdown": 0.15
            })
            self.drift_monitor = DriftMonitor(
                baseline_metrics=baseline_metrics,
                thresholds=drift_config.get("thresholds"),
                window_size=drift_config.get("window_size", 50),
                min_trades_for_drift=drift_config.get("min_trades", 20)
            )
            print("✅ Drift detection enabled")
        else:
            self.drift_monitor = None
        
        # NT8 Bridge Server
        self.bridge_server = None
        
        # Initialize signal calculator for NinjaScript signals
        signal_config = config.get("signals", {})
        self.signal_calculator = SignalCalculator(
            action_change_threshold=signal_config.get("action_change_threshold", 0.15),
            pullback_detection_bars=signal_config.get("pullback_detection_bars", 3),
            trend_strength_threshold=signal_config.get("trend_strength_threshold", 0.5)
        )
        
        # Load Markov regime (if available) - will be updated periodically
        self.markov_regime = None
        self.markov_regime_confidence = 0.0
        self._load_markov_regime()
        
    def start(self):
        """Start the live trading system"""
        print("\n" + "="*60)
        print("Starting Live Trading System")
        print("="*60)
        print(f"Model: {self.model_path}")
        print(f"Reasoning: {'Enabled' if self.reasoning_enabled else 'Disabled'}")
        print(f"Swarm: {'Enabled' if self.swarm_enabled else 'Disabled'}")
        print(f"Manual Approval: {'Enabled' if self.manual_approval_enabled else 'Disabled'}")
        print(f"Mode: {'LIVE' if self.config.get('live_trading', False) else 'PAPER'}")
        print("="*60 + "\n")
        
        # Start NT8 bridge server
        self.bridge_server = NT8BridgeServer(
            host=self.config.get("bridge", {}).get("host", "localhost"),
            port=self.config.get("bridge", {}).get("port", 8888),
            on_market_data=self._handle_market_data,
            on_trade_request=self._handle_trade_request
        )
        
        self.bridge_server.start()
        self.running = True
        
        print("✅ Live trading system started")
        print("✅ Waiting for NT8 connection...")
        print("\nPress Ctrl+C to stop\n")
        
        try:
            # Keep running
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nStopping live trading system...")
            self.stop()
    
    def stop(self):
        """Stop the live trading system"""
        self.running = False
        
        # Stop adaptive learning agent if enabled
        if self.swarm_orchestrator and self.swarm_enabled:
            self.swarm_orchestrator.stop_adaptive_learning()
        
        if self.bridge_server:
            self.bridge_server.stop()
        
        # Save statistics
        self._save_statistics()
        print("\n✅ System stopped")
    
    def _handle_market_data(self, data: Dict):
        """
        Handle incoming market data from NT8.
        
        This is called whenever NT8 sends new market data.
        """
        try:
            # Parse market data
            bar = self._parse_market_data(data)

            if self.trading_hours and not self.trading_hours.is_in_session(bar.timestamp):
                return

            if self.multi_timeframe_resampler:
                bars_by_tf = self.multi_timeframe_resampler.update(bar)
                # Wait until all higher timeframes have at least one completed bar
                missing = [
                    tf for tf in self.timeframes
                    if bars_by_tf.get(tf) is None
                ]
                if missing:
                    return
            else:
                bars_by_tf = {self.primary_timeframe: bar}

            self.latest_multi_tf_bars = bars_by_tf
            # Update state (would need to maintain state buffer)
            # For now, we'll request a trade decision
            self._process_market_update(bars_by_tf)

        except Exception as e:
            print(f"Error handling market data: {e}")

    def _parse_market_data(self, data: Dict) -> MarketBar:
        """Parse market data from NT8"""
        return MarketBar(
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"])
        )
    
    def _process_market_update(self, bars_by_timeframe: Dict[int, MarketBar]):
        """
        Process market update and make trading decision.
        
        In a full implementation, this would:
        1. Update state buffer with new bar
        2. Extract features
        3. Get RL recommendation
        4. Validate with reasoning
        5. Apply risk management
        6. Execute trade
        """
        # TODO: Implement full state update logic
        # For now, this is a placeholder that shows the structure
        
        primary_bar = bars_by_timeframe.get(self.primary_timeframe)
        if primary_bar is None:
            return

        if self.current_state is None:
            # Initialize state (would need historical data)
            return
        
        # Get RL recommendation
        action, value, log_prob = self.agent.select_action(
            self.current_state,
            deterministic=True  # Use mean action for live trading
        )
        
        rl_action_value = float(action[0])
        rl_confidence = min(abs(rl_action_value), 1.0)
        
        # Create RL recommendation object for swarm/reasoning
        rl_rec = RLRecommendation(
            action=TradeAction.BUY if rl_action_value > 0.1 else (TradeAction.SELL if rl_action_value < -0.1 else TradeAction.HOLD),
            confidence=rl_confidence
        )
        
        # Run swarm analysis (async, with timeout)
        swarm_recommendation = None
        if self.swarm_enabled and self.swarm_orchestrator:
            swarm_recommendation = self._run_swarm_analysis(primary_bar, rl_rec)
        
        # Use DecisionGate to combine RL + Swarm
        # Phase 3.4: Pass current timestamp for time-of-day filtering
        decision = self.decision_gate.make_decision(
            rl_action=rl_action_value,
            rl_confidence=rl_confidence,
            swarm_recommendation=swarm_recommendation,
            current_timestamp=primary_bar.timestamp
        )
        
        # Apply risk management (DecisionGate already handles basic risk, but apply final validation)
        # Note: validate_action now returns (position, monte_carlo_result)
        decision_context = {
            "confluence_count": decision.confluence_count,
            "confluence_score": decision.confluence_score,
            "scale_factor": decision.scale_factor,
            "agreement": decision.agreement
        }
        result = self.risk_manager.validate_action(
            decision.action,
            current_position=self.current_position,
            market_data={
                "price": primary_bar.close,
                "high": primary_bar.high,
                "low": primary_bar.low,
                "volume": primary_bar.volume
            },
            current_price=primary_bar.close,
            decision_context=decision_context,
            instrument=self.instrument
        )
        
        # Handle tuple return (position, monte_carlo_result)
        if isinstance(result, tuple):
            final_action, monte_carlo_result = result
            if monte_carlo_result:
                # Store Monte Carlo results for monitoring
                self.monte_carlo_result = monte_carlo_result
        else:
            # Backward compatibility
            final_action = result
        
        # Check if should execute
        if not self.decision_gate.should_execute(decision):
            print(f"⚠️  Decision rejected: confidence={decision.confidence:.2f} < threshold")
            self.stats["trades_rejected"] += 1
            return
        
        # Manual approval workflow (if enabled)
        if self.manual_approval_enabled:
            if not self._request_manual_approval(decision, primary_bar):
                print("⚠️  Trade rejected by manual approval")
                self.stats["trades_rejected"] += 1
                return
        
        # Execute trade
        if abs(final_action - self.current_position) > 0.01:
            self._execute_trade(final_action, primary_bar, decision)
        
        # Store original RL action for signal calculation
        decision.original_rl_action = rl_action_value
        
        # Log decision diagnostics for post-trade analysis
        self._log_decision(
            timestamp=primary_bar.timestamp,
            rl_action=rl_action_value,
            rl_confidence=rl_confidence,
            decision=decision,
            final_action=final_action
        )
    
    def _run_swarm_analysis(
        self,
        bar: MarketBar,
        rl_rec: RLRecommendation
    ) -> Optional[Dict]:
        """
        Run swarm analysis asynchronously with timeout.
        
        Returns:
            Swarm recommendation dict or None if failed/timed out
        """
        if not self.swarm_enabled or not self.swarm_orchestrator:
            return None
        
        try:
            # Prepare market data for swarm
            market_data = {
                "price_data": {
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close
                },
                "volume_data": {"volume": bar.volume},
                "indicators": {},  # Would need to calculate
                "market_regime": "trending",  # Would need to detect
                "timestamp": bar.timestamp.isoformat()
            }
            
            # Prepare RL recommendation for swarm
            rl_recommendation = {
                "action": rl_rec.action.name,
                "confidence": rl_rec.confidence,
                "reasoning": getattr(rl_rec, "reasoning", None)
            }
            
            # Run swarm analysis with timeout
            swarm_result = self.swarm_orchestrator.analyze_sync(
                market_data=market_data,
                rl_recommendation=rl_recommendation,
                current_position=self.current_position,
                timeout=self.decision_gate.swarm_timeout
            )
            
            # Extract recommendation from swarm result
            if swarm_result.get("status") == "success":
                recommendation = swarm_result.get("recommendation", {})
                if recommendation and recommendation.get("action") != "HOLD":
                    return recommendation
            
            # Swarm timed out or failed
            if swarm_result.get("status") == "timeout":
                print(f"⚠️  Swarm analysis timed out after {self.decision_gate.swarm_timeout}s")
            elif swarm_result.get("status") == "error":
                print(f"⚠️  Swarm analysis error: {swarm_result.get('error', 'Unknown error')}")
            
            return None
            
        except Exception as e:
            print(f"Warning: Swarm analysis failed: {e}")
            print("  Falling back to RL-only decision")
            return None
    
    def _request_manual_approval(
        self,
        decision: 'DecisionResult',
        bar: MarketBar
    ) -> bool:
        """
        Request manual approval for trade (if enabled).
        
        Args:
            decision: Decision result from DecisionGate
            bar: Current market bar
        
        Returns:
            True if approved, False if rejected
        """
        # Add to pending approvals queue
        approval_request = {
            "timestamp": bar.timestamp.isoformat(),
            "decision": decision,
            "bar": {
                "price": bar.close,
                "volume": bar.volume
            },
            "approved": None  # None = pending, True/False = decision
        }
        
        self.pending_approvals.append(approval_request)
        
        # TODO: In production, this would:
        # 1. Send notification to UI/API
        # 2. Wait for user response
        # 3. Return True/False based on user decision
        
        # For now, auto-approve (can be changed to require actual approval)
        print(f"📋 Manual approval request: {decision.action:.2f} @ {bar.close:.2f}")
        print(f"   RL: {decision.rl_confidence:.2f}, Swarm: {decision.swarm_confidence:.2f}, Combined: {decision.confidence:.2f}")
        print(f"   Agreement: {decision.agreement}")
        
        # Auto-approve for now (set to False to require manual approval)
        return True
    
    def _execute_trade(self, target_position: float, bar: MarketBar, decision: Optional[DecisionResult] = None):
        """
        Execute trade via NT8 bridge.
        
        Args:
            target_position: Target position size (-1.0 to 1.0)
            bar: Current market bar
            decision: Decision result (optional, for signal calculation)
        """
        position_change = target_position - self.current_position
        
        if abs(position_change) < 0.01:
            return  # No significant change
        
        # Calculate NinjaScript signals if decision is available
        signal_trend = 0
        signal_trade = 0
        
        if decision:
            # Get RL action and confidence from decision
            # Use original RL action if stored, otherwise use target_position as proxy
            rl_action = getattr(decision, 'original_rl_action', target_position)
            rl_confidence = getattr(decision, 'rl_confidence', getattr(decision, 'confidence', 0.0))
            swarm_recommendation = getattr(decision, 'swarm_recommendation', None)
            
            # Calculate signals
            signal_trend, signal_trade = self.signal_calculator.calculate_signals(
                rl_action=rl_action,
                rl_confidence=rl_confidence,
                swarm_recommendation=swarm_recommendation,
                current_position=self.current_position,
                markov_regime=self.markov_regime,
                markov_regime_confidence=self.markov_regime_confidence
            )
        
        # Create trade signal
        signal = {
            "action": "buy" if position_change > 0 else "sell",
            "position_size": target_position,
            "confidence": min(abs(target_position), 1.0),
            "signal_trend": signal_trend,
            "signal_trade": signal_trade,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to NT8
        if self.bridge_server:
            self.bridge_server.send_trade_signal(signal)
            self.stats["trades_executed"] += 1
            print(f"📊 Signal sent: {signal['action']} @ size {target_position:.2f} | Trend: {signal_trend}, Trade: {signal_trade}")
            
            # Update position
            self.current_position = target_position
        else:
            print("⚠️  Bridge server not connected")
    
    def _log_decision(
        self,
        timestamp: datetime,
        rl_action: float,
        rl_confidence: float,
        decision: DecisionResult,
        final_action: float
    ):
        """Append decision diagnostics to a CSV log for offline analysis."""
        if not getattr(self, "decision_log_path", None):
            return
        try:
            risk_state = self.risk_manager.get_position_state_info(self.instrument)
            row = [
                timestamp.isoformat(),
                rl_action,
                rl_confidence,
                decision.action,
                decision.confidence,
                getattr(decision, "scale_factor", None),
                getattr(decision, "confluence_count", None),
                getattr(decision, "confluence_score", None),
                decision.agreement,
                final_action,
                self.current_position,
                risk_state.get("break_even_active"),
                risk_state.get("avg_entry"),
                risk_state.get("protected_size"),
                risk_state.get("trail_price"),
                json.dumps(decision.confluence_signals or {}),
                json.dumps(risk_state),
            ]
            with open(self.decision_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            print(f"⚠️  Failed to log decision: {e}")
    
    def log_completed_trade(
        self,
        entry_price: float,
        exit_price: float,
        position_size: float,
        duration_seconds: float,
        rl_confidence: float,
        reasoning_confidence: Optional[float] = None
    ):
        """
        Log a completed trade for drift monitoring.
        
        This should be called when a trade closes to update drift detection.
        """
        pnl = (exit_price - entry_price) / entry_price * position_size * 100  # Simplified PnL calculation
        
        trade_metrics = TradeMetrics(
            timestamp=datetime.now().isoformat(),
            action="buy" if position_size > 0 else "sell",
            position_size=position_size,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            duration_seconds=duration_seconds,
            rl_confidence=rl_confidence,
            reasoning_confidence=reasoning_confidence,
            agreement="agree" if reasoning_confidence else None
        )
        
        # Update drift monitor if enabled
        if self.drift_monitor:
            self.drift_monitor.update(trade_metrics)
            
            # Check for rollback recommendation
            if self.drift_monitor.should_rollback():
                recommendation = self.drift_monitor.get_rollback_recommendation()
                if recommendation:
                    print(f"\n🚨 {recommendation}\n")
        
        # Update trade history
        self.trade_history.append({
            "timestamp": trade_metrics.timestamp,
            "pnl": pnl,
            "entry": entry_price,
            "exit": exit_price,
            "position_size": position_size
        })
        
        # Update statistics
        self.stats["total_pnl"] += pnl
        
        # Track winning/losing trades for adaptive learning
        if pnl > 0:
            self.winning_trades.append(pnl)
        else:
            self.losing_trades.append(abs(pnl))
        
        # Update max equity and drawdown
        current_equity = self.max_equity + self.stats["total_pnl"]
        if current_equity > self.max_equity:
            self.max_equity = current_equity
        else:
            drawdown = (self.max_equity - current_equity) / self.max_equity if self.max_equity > 0 else 0.0
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
    
    def get_drift_status(self):
        """
        Get current drift detection status.
        
        Returns:
            DriftMetrics or None if drift monitoring disabled
        """
        if self.drift_monitor:
            return self.drift_monitor.get_drift_status()
        return None
    
    def get_performance_data(self) -> Dict[str, Any]:
        """
        Get performance data for adaptive learning agent.
        
        Returns:
            Dict with performance metrics
        """
        total_trades = len(self.trade_history)
        winning_trades_count = len(self.winning_trades)
        losing_trades_count = len(self.losing_trades)
        
        # Calculate averages
        avg_win = sum(self.winning_trades) / max(1, winning_trades_count) if self.winning_trades else 0.0
        avg_loss = sum(self.losing_trades) / max(1, losing_trades_count) if self.losing_trades else 0.0
        
        # Calculate time window
        time_window_seconds = (datetime.now() - self.performance_start_time).total_seconds()
        trades_per_hour = (total_trades / max(1, time_window_seconds)) * 3600 if time_window_seconds > 0 else 0.0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades_count,
            "losing_trades": losing_trades_count,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_drawdown": self.max_drawdown,
            "trades_per_hour": trades_per_hour,
            "time_window_seconds": time_window_seconds,
            "total_pnl": self.stats.get("total_pnl", 0.0),
            "current_equity": self.max_equity + self.stats.get("total_pnl", 0.0)
        }
    
    def get_adaptive_learning_recommendations(self) -> Optional[Dict[str, Any]]:
        """Get latest adaptive learning recommendations"""
        if self.swarm_orchestrator and self.swarm_enabled:
            return self.swarm_orchestrator.get_adaptive_learning_recommendations()
        return None
    
    def apply_adaptive_learning_recommendation(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply an approved adaptive learning recommendation"""
        if self.swarm_orchestrator and self.swarm_enabled:
            return self.swarm_orchestrator.apply_adaptive_learning_recommendation(recommendation)
        return {"status": "error", "message": "Swarm orchestrator not available"}
    
    def _handle_trade_request(self, data: Dict) -> Dict:
        """
        Handle trade request from NT8.
        
        This is called when NT8 explicitly requests a trade signal.
        """
        # Get current recommendation
        if self.current_state is None:
            return {
                "action": "hold",
                "position_size": 0.0,
                "confidence": 0.0
            }
        
        action, value, _ = self.agent.select_action(self.current_state, deterministic=True)
        
        return {
            "action": "buy" if action[0] > 0.1 else ("sell" if action[0] < -0.1 else "hold"),
            "position_size": float(action[0]),
            "confidence": min(abs(action[0]), 1.0)
        }
    
    def _save_statistics(self):
        """Save trading statistics"""
        stats_path = Path("logs") / f"live_trading_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        stats_path.parent.mkdir(exist_ok=True)
        
        stats = {
            **self.stats,
            "end_time": datetime.now().isoformat(),
            "reasoning_agreement_rate": (
                self.stats["reasoning_agreements"] / 
                max(1, self.stats["reasoning_agreements"] + self.stats["reasoning_disagreements"])
            )
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n📊 Statistics saved to: {stats_path}")
    
    def _load_markov_regime(self):
        """Load latest Markov regime from report file (if available)."""
        try:
            import json
            from pathlib import Path
            
            # Look for latest Markov regime report
            report_path = Path("reports/markov_regime_report.json")
            if not report_path.exists():
                return
            
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            # Extract current regime (simplified - would need real-time regime detection in production)
            # For now, use the most probable regime from stationary distribution
            if "stationary_distribution" in report:
                stationary = report["stationary_distribution"]
                if isinstance(stationary, dict):
                    # Find regime with highest probability
                    max_regime = max(stationary.items(), key=lambda x: x[1])
                    self.markov_regime = max_regime[0]
                    self.markov_regime_confidence = float(max_regime[1])
                elif isinstance(stationary, list) and len(stationary) > 0:
                    # If it's a list, use the first regime (would need proper mapping)
                    self.markov_regime = "NEUTRAL"
                    self.markov_regime_confidence = 0.5
        except Exception as e:
            # Silently fail - Markov regime is optional
            pass


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Live Trading System")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Run in paper trading mode (default)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live trading mode (USE WITH CAUTION)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set trading mode
    if args.live:
        config["live_trading"] = True
        print("⚠️  WARNING: LIVE TRADING MODE ENABLED!")
        print("   This will execute real trades with real money!")
        response = input("   Are you sure? Type 'YES' to continue: ")
        if response != "YES":
            print("Aborted.")
            return
    else:
        config["live_trading"] = False
    
    # Create and start system
    system = LiveTradingSystem(config, args.model)
    system.start()


if __name__ == "__main__":
    main()

