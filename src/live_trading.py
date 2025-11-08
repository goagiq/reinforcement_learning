"""
Live Trading Execution Module

Coordinates between NT8, RL agent, and reasoning engine for live/paper trading.

Usage:
    python src/live_trading.py --config configs/train_config.yaml --model models/best_model.pt
"""

import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import time
import json
import threading
from typing import Dict, Optional

from src.data_extraction import DataExtractor, MarketBar
from src.trading_env import TradingEnvironment
from src.rl_agent import PPOAgent
from src.nt8_bridge_server import NT8BridgeServer
from src.reasoning_engine import ReasoningEngine, MarketState, RLRecommendation, TradeAction
from src.risk_manager import RiskManager
from src.drift_monitor import DriftMonitor, TradeMetrics
from src.decision_gate import DecisionGate
from src.agentic_swarm import SwarmOrchestrator
import asyncio


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
            
            # Update state (would need to maintain state buffer)
            # For now, we'll request a trade decision
            self._process_market_update(bar)
            
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
    
    def _process_market_update(self, bar: MarketBar):
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
            swarm_recommendation = self._run_swarm_analysis(bar, rl_rec)
        
        # Use DecisionGate to combine RL + Swarm
        decision = self.decision_gate.make_decision(
            rl_action=rl_action_value,
            rl_confidence=rl_confidence,
            swarm_recommendation=swarm_recommendation
        )
        
        # Apply risk management (DecisionGate already handles basic risk, but apply final validation)
        # Note: validate_action now returns (position, monte_carlo_result)
        result = self.risk_manager.validate_action(
            decision.action,
            current_position=self.current_position,
            market_data={
                "price": bar.close,
                "high": bar.high,
                "low": bar.low,
                "volume": bar.volume
            }
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
            if not self._request_manual_approval(decision, bar):
                print("⚠️  Trade rejected by manual approval")
                self.stats["trades_rejected"] += 1
                return
        
        # Execute trade
        if abs(final_action - self.current_position) > 0.01:
            self._execute_trade(final_action, bar)
    
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
    
    def _execute_trade(self, target_position: float, bar: MarketBar):
        """
        Execute trade via NT8 bridge.
        
        Args:
            target_position: Target position size (-1.0 to 1.0)
            bar: Current market bar
        """
        position_change = target_position - self.current_position
        
        if abs(position_change) < 0.01:
            return  # No significant change
        
        # Create trade signal
        signal = {
            "action": "buy" if position_change > 0 else "sell",
            "position_size": target_position,
            "confidence": min(abs(target_position), 1.0),
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to NT8
        if self.bridge_server:
            self.bridge_server.send_trade_signal(signal)
            self.stats["trades_executed"] += 1
            print(f"📊 Signal sent: {signal['action']} @ size {target_position:.2f}")
            
            # Update position
            self.current_position = target_position
        else:
            print("⚠️  Bridge server not connected")
    
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
            "exit": exit_price
        })
        
        # Update statistics
        self.stats["total_pnl"] += pnl
    
    def get_drift_status(self):
        """
        Get current drift detection status.
        
        Returns:
            DriftMetrics or None if drift monitoring disabled
        """
        if self.drift_monitor:
            return self.drift_monitor.get_drift_status()
        return None
    
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

