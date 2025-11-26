"""
Drift Detection and Model Performance Monitoring

Monitors live trading performance and detects model degradation.
Provides automatic rollback recommendations when thresholds are breached.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque

from src.monitoring import PerformanceMetrics, TradeMetrics


@dataclass
class DriftAlert:
    """Drift detection alert"""
    timestamp: str
    severity: str  # "info", "warning", "critical"
    metric_name: str
    current_value: float
    baseline_value: float
    threshold_value: float
    message: str
    recommendation: str


@dataclass
class DriftMetrics:
    """Current drift metrics"""
    win_rate_drift: float
    sharpe_drift: float
    profit_factor_drift: float
    max_drawdown_drift: float
    consecutive_losses: int
    is_degrading: bool
    severity: str


class DriftMonitor:
    """
    Monitors live trading performance and detects model drift/degradation.
    
    Features:
    - Real-time performance tracking
    - Baseline vs current comparison
    - Configurable alert thresholds
    - Automatic rollback recommendations
    """
    
    def __init__(
        self,
        baseline_metrics: Dict,
        thresholds: Optional[Dict] = None,
        window_size: int = 50,
        min_trades_for_drift: int = 20
    ):
        """
        Initialize drift monitor.
        
        Args:
            baseline_metrics: Baseline performance metrics (from training/test)
            thresholds: Custom drift thresholds (uses defaults if None)
            window_size: Number of recent trades to analyze
            min_trades_for_drift: Minimum trades before drift detection activates
        """
        self.baseline_metrics = baseline_metrics
        self.thresholds = thresholds or self._get_default_thresholds()
        self.window_size = window_size
        self.min_trades_for_drift = min_trades_for_drift
        
        # Current state
        self.recent_trades: deque = deque(maxlen=window_size)
        self.consecutive_losses = 0
        self.alerts: List[DriftAlert] = []
        self.drift_detected = False
        self.rollback_recommended = False
        
        # Initialize baseline values
        self.baseline_win_rate = baseline_metrics.get("win_rate", 0.55)
        self.baseline_sharpe = baseline_metrics.get("sharpe_ratio", 1.0)
        self.baseline_profit_factor = baseline_metrics.get("profit_factor", 1.5)
        self.baseline_max_dd = baseline_metrics.get("max_drawdown", 0.10)
    
    def _get_default_thresholds(self) -> Dict:
        """Get default drift detection thresholds"""
        return {
            "win_rate_drop": 0.10,           # 10% absolute drop (55% -> 45%)
            "sharpe_drop": 0.30,             # 0.3 absolute drop (1.0 -> 0.7)
            "profit_factor_drop": 0.30,      # 30% relative drop
            "max_drawdown_increase": 0.05,   # 5% absolute increase
            "consecutive_losses": 5,         # 5 losses in a row
            "rollback_threshold": "critical" # Severity to trigger rollback
        }
    
    def update(self, trade: TradeMetrics):
        """
        Update drift monitor with new trade.
        
        Args:
            trade: Completed trade metrics
        """
        self.recent_trades.append(trade)
        
        # Update consecutive losses counter
        if trade.pnl <= 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Check for drift if we have enough data
        if len(self.recent_trades) >= self.min_trades_for_drift:
            self._check_drift()
    
    def _check_drift(self):
        """Check for model drift and generate alerts"""
        if not self.recent_trades:
            return
        
        # Calculate current metrics from recent trades
        current_metrics = self._calculate_current_metrics()
        
        # Check each metric against thresholds
        self._check_win_rate_drift(current_metrics)
        self._check_sharpe_drift(current_metrics)
        self._check_profit_factor_drift(current_metrics)
        self._check_drawdown_drift(current_metrics)
        self._check_consecutive_losses()
        
        # Determine overall drift status
        self._update_drift_status()
    
    def _calculate_current_metrics(self) -> Dict:
        """Calculate metrics from recent trades"""
        if not self.recent_trades:
            return {}
        
        winning = [t for t in self.recent_trades if t.pnl > 0]
        losing = [t for t in self.recent_trades if t.pnl <= 0]
        
        metrics = {
            "win_rate": len(winning) / len(self.recent_trades) if self.recent_trades else 0.0,
            "total_trades": len(self.recent_trades)
        }
        
        if winning:
            metrics["average_win"] = sum(t.pnl for t in winning) / len(winning)
        if losing:
            metrics["average_loss"] = abs(sum(t.pnl for t in losing) / len(losing))
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # CRITICAL FIX #5: Sharpe ratio (from percentage returns, not raw PnL)
        if len(self.recent_trades) > 1:
            # Get initial capital (default 100000.0)
            initial_capital = getattr(self, 'initial_capital', 100000.0)
            if initial_capital <= 0:
                initial_capital = 100000.0
            
            # Convert PnL to percentage returns
            pnl_values = [t.pnl for t in self.recent_trades]
            returns = [pnl / initial_capital for pnl in pnl_values]
            
            mean_return = sum(returns) / len(returns)
            std_return = (sum((r - mean_return)**2 for r in returns) / len(returns))**0.5
            risk_free_rate = 0.0  # Default risk-free rate
            
            # Sharpe ratio = (mean_return - risk_free_rate) / std_return * sqrt(periods_per_year)
            if std_return > 0:
                metrics["sharpe_ratio"] = (mean_return - risk_free_rate) / std_return * np.sqrt(252)
            else:
                metrics["sharpe_ratio"] = 0.0
        else:
            metrics["sharpe_ratio"] = 0.0
        
        return metrics
    
    def _check_win_rate_drift(self, current: Dict):
        """Check win rate degradation"""
        win_rate = current.get("win_rate", 0.0)
        drop = self.baseline_win_rate - win_rate
        
        if drop >= self.thresholds["win_rate_drop"]:
            severity = "critical" if drop >= self.thresholds["win_rate_drop"] * 2 else "warning"
            
            alert = DriftAlert(
                timestamp=datetime.now().isoformat(),
                severity=severity,
                metric_name="win_rate",
                current_value=win_rate,
                baseline_value=self.baseline_win_rate,
                threshold_value=self.baseline_win_rate - self.thresholds["win_rate_drop"],
                message=f"Win rate dropped from {self.baseline_win_rate:.1%} to {win_rate:.1%} (drop: {drop:.1%})",
                recommendation="Consider rolling back to previous model version" if severity == "critical" else "Monitor closely"
            )
            self.alerts.append(alert)
    
    def _check_sharpe_drift(self, current: Dict):
        """Check Sharpe ratio degradation"""
        sharpe = current.get("sharpe_ratio", 0.0)
        drop = self.baseline_sharpe - sharpe
        
        if drop >= self.thresholds["sharpe_drop"]:
            severity = "critical" if drop >= self.thresholds["sharpe_drop"] * 2 else "warning"
            
            alert = DriftAlert(
                timestamp=datetime.now().isoformat(),
                severity=severity,
                metric_name="sharpe_ratio",
                current_value=sharpe,
                baseline_value=self.baseline_sharpe,
                threshold_value=self.baseline_sharpe - self.thresholds["sharpe_drop"],
                message=f"Sharpe ratio dropped from {self.baseline_sharpe:.2f} to {sharpe:.2f} (drop: {drop:.2f})",
                recommendation="Rollback recommended" if severity == "critical" else "Review model performance"
            )
            self.alerts.append(alert)
    
    def _check_profit_factor_drift(self, current: Dict):
        """Check profit factor degradation"""
        pf = current.get("profit_factor", 0.0)
        
        if pf == float('inf') or pf <= 0:
            return
        
        drop_ratio = (self.baseline_profit_factor - pf) / self.baseline_profit_factor
        
        if drop_ratio >= self.thresholds["profit_factor_drop"]:
            severity = "critical" if drop_ratio >= 0.5 else "warning"
            
            alert = DriftAlert(
                timestamp=datetime.now().isoformat(),
                severity=severity,
                metric_name="profit_factor",
                current_value=pf,
                baseline_value=self.baseline_profit_factor,
                threshold_value=self.baseline_profit_factor * (1 - self.thresholds["profit_factor_drop"]),
                message=f"Profit factor dropped from {self.baseline_profit_factor:.2f} to {pf:.2f} ({drop_ratio:.1%} drop)",
                recommendation="Rollback recommended" if severity == "critical" else "Monitor closely"
            )
            self.alerts.append(alert)
    
    def _check_drawdown_drift(self, current: Dict):
        """Check max drawdown increase"""
        # This would need to be tracked from equity history
        # For now, we'll skip this check
        pass
    
    def _check_consecutive_losses(self):
        """Check for consecutive losses"""
        if self.consecutive_losses >= self.thresholds["consecutive_losses"]:
            alert = DriftAlert(
                timestamp=datetime.now().isoformat(),
                severity="warning",
                metric_name="consecutive_losses",
                current_value=self.consecutive_losses,
                baseline_value=0,
                threshold_value=self.thresholds["consecutive_losses"],
                message=f"{self.consecutive_losses} consecutive losses detected",
                recommendation="Review recent trades and consider pausing if trend continues"
            )
            self.alerts.append(alert)
    
    def _update_drift_status(self):
        """Update overall drift detection status"""
        # Check if any critical alerts exist
        critical_alerts = [a for a in self.alerts if a.severity == "critical"]
        
        if critical_alerts:
            self.drift_detected = True
            self.rollback_recommended = True
        else:
            # Check warning alerts
            recent_warnings = [a for a in self.alerts 
                             if a.severity == "warning" 
                             and (datetime.now() - datetime.fromisoformat(a.timestamp)).total_seconds() < 3600]
            
            if recent_warnings:
                self.drift_detected = True
                self.rollback_recommended = False
    
    def get_drift_status(self) -> DriftMetrics:
        """
        Get current drift status.
        
        Returns:
            DriftMetrics with current drift information
        """
        current = self._calculate_current_metrics()
        
        return DriftMetrics(
            win_rate_drift=self.baseline_win_rate - current.get("win_rate", 0.0),
            sharpe_drift=self.baseline_sharpe - current.get("sharpe_ratio", 0.0),
            profit_factor_drift=self.baseline_profit_factor - current.get("profit_factor", 0.0),
            max_drawdown_drift=0.0,  # Would need equity history
            consecutive_losses=self.consecutive_losses,
            is_degrading=self.drift_detected,
            severity=self._get_overall_severity()
        )
    
    def _get_overall_severity(self) -> str:
        """Get overall drift severity"""
        if not self.alerts:
            return "none"
        
        recent_alerts = [a for a in self.alerts 
                        if (datetime.now() - datetime.fromisoformat(a.timestamp)).total_seconds() < 3600]
        
        if not recent_alerts:
            return "none"
        
        severities = [a.severity for a in recent_alerts]
        
        if "critical" in severities:
            return "critical"
        elif "warning" in severities:
            return "warning"
        else:
            return "info"
    
    def get_recent_alerts(self, hours: int = 24) -> List[DriftAlert]:
        """
        Get recent alerts within specified time window.
        
        Args:
            hours: Hours of history to retrieve
            
        Returns:
            List of recent alerts
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        return [a for a in self.alerts 
                if datetime.fromisoformat(a.timestamp) >= cutoff]
    
    def should_rollback(self) -> bool:
        """Check if rollback is recommended"""
        return self.rollback_recommended
    
    def get_rollback_recommendation(self) -> Optional[str]:
        """
        Get rollback recommendation message.
        
        Returns:
            Recommendation message or None
        """
        if not self.rollback_recommended:
            return None
        
        critical_alerts = self.get_recent_alerts(hours=1)
        critical_alerts = [a for a in critical_alerts if a.severity == "critical"]
        
        if critical_alerts:
            return (
                f"🚨 Model degradation detected! "
                f"{len(critical_alerts)} critical alerts in the last hour. "
                f"Recommend rolling back to previous stable model."
            )
        
        return None
    
    def save_alerts(self, filepath: Optional[str] = None):
        """Save alerts to file"""
        if filepath is None:
            filepath = Path("logs") / f"drift_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        Path(filepath).parent.mkdir(exist_ok=True)
        
        data = {
            "baseline_metrics": self.baseline_metrics,
            "thresholds": self.thresholds,
            "alerts": [asdict(a) for a in self.alerts],
            "current_status": asdict(self.get_drift_status())
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📊 Drift alerts saved to: {filepath}")
        return filepath
    
    def print_status(self):
        """Print current drift status"""
        status = self.get_drift_status()
        
        print("\n" + "="*60)
        print("DRIFT DETECTION STATUS")
        print("="*60)
        print(f"Status: {'🔴 DEGRADING' if status.is_degrading else '🟢 HEALTHY'}")
        print(f"Severity: {status.severity}")
        print(f"Win Rate Drift: {status.win_rate_drift:.2%}")
        print(f"Sharpe Drift: {status.sharpe_drift:.2f}")
        print(f"Consecutive Losses: {status.consecutive_losses}")
        print(f"Rollback Recommended: {'⚠️ YES' if self.rollback_recommended else '✅ NO'}")
        
        recent = self.get_recent_alerts(hours=1)
        if recent:
            print(f"\nRecent Alerts (Last Hour): {len(recent)}")
            for alert in recent[:5]:  # Show last 5
                print(f"  [{alert.severity.upper()}] {alert.message}")
        
        print("="*60)


# Example usage
if __name__ == "__main__":
    # Example baseline metrics
    baseline = {
        "win_rate": 0.55,
        "sharpe_ratio": 1.20,
        "profit_factor": 1.50,
        "max_drawdown": 0.10
    }
    
    # Create monitor
    monitor = DriftMonitor(baseline_metrics=baseline)
    
    # Simulate degraded performance
    for i in range(30):
        trade = TradeMetrics(
            timestamp=datetime.now().isoformat(),
            action="buy" if i % 3 == 0 else "sell",
            position_size=0.5,
            entry_price=100.0,
            exit_price=98.0 if i % 3 == 0 else 102.0,  # More losses
            pnl=-20.0 if i % 3 == 0 else 50.0,
            duration_seconds=3600,
            rl_confidence=0.7,
            reasoning_confidence=0.75,
            agreement="agree"
        )
        monitor.update(trade)
    
    # Check status
    monitor.print_status()
    
    # Save alerts
    monitor.save_alerts()

