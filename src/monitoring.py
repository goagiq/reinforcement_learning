"""
Performance Monitoring Module

Monitors trading performance, logs metrics, and provides real-time dashboards.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class TradeMetrics:
    """Metrics for a single trade"""
    timestamp: str
    action: str
    position_size: float
    entry_price: float
    exit_price: Optional[float]
    pnl: float
    duration_seconds: float
    rl_confidence: float
    reasoning_confidence: Optional[float]
    agreement: Optional[str]


@dataclass
class PerformanceMetrics:
    """Aggregate performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    average_win: float
    average_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    current_equity: float


class PerformanceMonitor:
    """
    Monitors and logs trading performance.
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Metrics storage
        self.trades: List[TradeMetrics] = []
        self.equity_history: List[Dict] = []
        self.start_time = datetime.now()
        
        # Real-time metrics
        self.current_metrics = PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            average_win=0.0,
            average_loss=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            current_equity=0.0
        )
    
    def log_trade(self, metrics: TradeMetrics):
        """Log a completed trade"""
        self.trades.append(metrics)
        
        # Update aggregate metrics
        self._update_metrics()
        
        # Save to file
        self._save_trade(metrics)
    
    def log_equity(self, equity: float, timestamp: Optional[str] = None):
        """Log current equity"""
        entry = {
            "timestamp": timestamp or datetime.now().isoformat(),
            "equity": equity
        }
        self.equity_history.append(entry)
        
        # Update max drawdown
        if len(self.equity_history) > 1:
            peak_equity = max(e["equity"] for e in self.equity_history)
            self.current_metrics.max_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
    
    def _update_metrics(self):
        """Update aggregate performance metrics"""
        if not self.trades:
            return
        
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]
        
        self.current_metrics.total_trades = len(self.trades)
        self.current_metrics.winning_trades = len(winning)
        self.current_metrics.losing_trades = len(losing)
        self.current_metrics.win_rate = len(winning) / len(self.trades) if self.trades else 0.0
        self.current_metrics.total_pnl = sum(t.pnl for t in self.trades)
        
        if winning:
            self.current_metrics.average_win = sum(t.pnl for t in winning) / len(winning)
        if losing:
            self.current_metrics.average_loss = abs(sum(t.pnl for t in losing) / len(losing))
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        self.current_metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe ratio (simplified)
        if len(self.trades) > 1:
            returns = [t.pnl for t in self.trades]
            mean_return = sum(returns) / len(returns)
            std_return = (sum((r - mean_return)**2 for r in returns) / len(returns))**0.5
            self.current_metrics.sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
        
        # Current equity
        if self.equity_history:
            self.current_metrics.current_equity = self.equity_history[-1]["equity"]
    
    def _save_trade(self, metrics: TradeMetrics):
        """Save trade to JSON log file"""
        trade_log = self.log_dir / "trades.jsonl"
        
        with open(trade_log, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')
    
    def get_summary(self) -> Dict:
        """Get current performance summary"""
        return {
            "metrics": asdict(self.current_metrics),
            "runtime_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            "trades_per_hour": self.current_metrics.total_trades / \
                              max(0.1, (datetime.now() - self.start_time).total_seconds() / 3600)
        }
    
    def print_summary(self):
        """Print performance summary to console"""
        summary = self.get_summary()
        metrics = summary["metrics"]
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Runtime: {summary['runtime_hours']:.2f} hours")
        print(f"Trades: {metrics['total_trades']} ({summary['trades_per_hour']:.2f}/hour)")
        print(f"Win Rate: {metrics['win_rate']*100:.1f}% ({metrics['winning_trades']}W / {metrics['losing_trades']}L)")
        print(f"Total PnL: ${metrics['total_pnl']:.2f}")
        print(f"Average Win: ${metrics['average_win']:.2f}")
        print(f"Average Loss: ${metrics['average_loss']:.2f}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"Current Equity: ${metrics['current_equity']:.2f}")
        print("="*60)
    
    def save_report(self, filepath: Optional[str] = None):
        """Save performance report to file"""
        if filepath is None:
            filepath = self.log_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "summary": self.get_summary(),
            "trades": [asdict(t) for t in self.trades],
            "equity_history": self.equity_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Performance report saved to: {filepath}")
        return filepath
    
    def plot_equity_curve(self, save_path: Optional[str] = None):
        """Plot equity curve"""
        if not self.equity_history:
            print("No equity data to plot")
            return
        
        df = pd.DataFrame(self.equity_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['equity'], label='Equity')
        plt.title('Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"📈 Equity curve saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()


# Example usage
if __name__ == "__main__":
    monitor = PerformanceMonitor()
    
    # Simulate some trades
    monitor.log_equity(100000.0)
    
    trade1 = TradeMetrics(
        timestamp=datetime.now().isoformat(),
        action="buy",
        position_size=0.5,
        entry_price=100.0,
        exit_price=102.0,
        pnl=100.0,
        duration_seconds=3600,
        rl_confidence=0.8,
        reasoning_confidence=0.85,
        agreement="agree"
    )
    monitor.log_trade(trade1)
    monitor.log_equity(100100.0)
    
    trade2 = TradeMetrics(
        timestamp=datetime.now().isoformat(),
        action="sell",
        position_size=-0.3,
        entry_price=101.0,
        exit_price=100.5,
        pnl=-15.0,
        duration_seconds=1800,
        rl_confidence=0.6,
        reasoning_confidence=0.7,
        agreement="modify"
    )
    monitor.log_trade(trade2)
    monitor.log_equity(100085.0)
    
    # Print summary
    monitor.print_summary()
    
    # Save report
    monitor.save_report()

