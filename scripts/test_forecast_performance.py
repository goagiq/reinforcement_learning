"""
Test Forecast Features Performance

This script helps compare performance with/without forecast features.
It analyzes training metrics and trade journal data to determine if
forecast features improve trading performance.

Usage:
    python scripts/test_forecast_performance.py [--with-forecast] [--without-forecast] [--compare]
"""

import sys
import io
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

# Configure stdout for Windows Unicode support
if sys.platform == 'win32':
    try:
        if not isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except (AttributeError, ValueError):
        pass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_trade_journal(db_path: str = "logs/trading_journal.db") -> pd.DataFrame:
    """Load trades from trading journal"""
    try:
        conn = sqlite3.connect(db_path)
        query = """
            SELECT 
                timestamp, episode, strategy, entry_price, exit_price, 
                pnl, net_pnl, strategy_confidence, is_win
            FROM trades
            ORDER BY timestamp DESC
            LIMIT 1000
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load trade journal: {e}")
        return pd.DataFrame()


def analyze_performance(trades: pd.DataFrame, label: str = "") -> Dict:
    """Analyze trading performance metrics"""
    if len(trades) == 0:
        return {
            "label": label,
            "total_trades": 0,
            "error": "No trades found"
        }
    
    total_trades = len(trades)
    winning_trades = trades[trades['is_win'] == 1]
    losing_trades = trades[trades['is_win'] == 0]
    
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    win_rate = win_count / total_trades if total_trades > 0 else 0.0
    
    total_pnl = trades['net_pnl'].sum()
    avg_pnl = trades['net_pnl'].mean()
    
    avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0.0
    avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0.0
    
    profit_factor = abs(winning_trades['net_pnl'].sum() / losing_trades['net_pnl'].sum()) if len(losing_trades) > 0 and losing_trades['net_pnl'].sum() != 0 else 0.0
    
    # Calculate Sharpe-like ratio (simplified)
    if len(trades) > 1:
        returns = trades['net_pnl'].values
        sharpe_like = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252) if np.std(returns) > 0 else 0.0
    else:
        sharpe_like = 0.0
    
    # Max drawdown
    cumulative = trades['net_pnl'].cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0
    
    return {
        "label": label,
        "total_trades": total_trades,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "sharpe_like": sharpe_like,
        "max_drawdown": max_drawdown
    }


def check_forecast_features_usage() -> Dict:
    """Check if forecast features are being used in current config"""
    try:
        import yaml
        
        config_path = project_root / "configs/train_config_adaptive.yaml"
        if not config_path.exists():
            return {"error": "Config file not found"}
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check config structure - features can be in different locations
        env_config = config.get('environment', {})
        features_config = env_config.get('features', {})
        reward_config = env_config.get('reward', {})
        
        # Check all possible locations (features dict, reward dict, or direct in environment)
        forecast_enabled = (
            features_config.get('include_forecast_features', False) or 
            reward_config.get('include_forecast_features', False) or
            env_config.get('include_forecast_features', False)
        )
        regime_enabled = (
            features_config.get('include_regime_features', False) or 
            reward_config.get('include_regime_features', False) or
            env_config.get('include_regime_features', False)
        )
        state_features = env_config.get('state_features', 900)
        
        return {
            "forecast_enabled": forecast_enabled,
            "regime_enabled": regime_enabled,
            "state_features": state_features,
            "expected_state_dim": 900 + (5 if regime_enabled else 0) + (3 if forecast_enabled else 0)
        }
    except Exception as e:
        return {"error": str(e)}


def compare_performance(with_forecast_trades: pd.DataFrame, without_forecast_trades: pd.DataFrame) -> Dict:
    """Compare performance between two sets of trades"""
    with_stats = analyze_performance(with_forecast_trades, "With Forecast Features")
    without_stats = analyze_performance(without_forecast_trades, "Without Forecast Features")
    
    comparison = {
        "with_forecast": with_stats,
        "without_forecast": without_stats,
        "improvements": {}
    }
    
    # Calculate improvements
    if with_stats.get("total_trades", 0) > 0 and without_stats.get("total_trades", 0) > 0:
        win_rate_diff = with_stats["win_rate"] - without_stats["win_rate"]
        profit_factor_diff = with_stats["profit_factor"] - without_stats["profit_factor"]
        total_pnl_diff = with_stats["total_pnl"] - without_stats["total_pnl"]
        sharpe_diff = with_stats["sharpe_like"] - without_stats["sharpe_like"]
        
        comparison["improvements"] = {
            "win_rate_change": win_rate_diff,
            "win_rate_change_pct": (win_rate_diff / without_stats["win_rate"] * 100) if without_stats["win_rate"] > 0 else 0.0,
            "profit_factor_change": profit_factor_diff,
            "total_pnl_change": total_pnl_diff,
            "sharpe_change": sharpe_diff,
            "forecast_better": (
                win_rate_diff > 0 and 
                profit_factor_diff > 0 and 
                total_pnl_diff > 0
            )
        }
    
    return comparison


def print_performance_report(stats: Dict):
    """Print formatted performance report"""
    print(f"\n{'='*80}")
    print(f"PERFORMANCE REPORT: {stats.get('label', 'Unknown')}")
    print(f"{'='*80}")
    
    if "error" in stats:
        print(f"  [ERROR] {stats['error']}")
        return
    
    print(f"\n[Trading Metrics]:")
    print(f"   Total Trades: {stats['total_trades']}")
    print(f"   Win Rate: {stats['win_rate']:.2%} ({stats['win_count']} wins, {stats['loss_count']} losses)")
    print(f"   Profit Factor: {stats['profit_factor']:.2f}")
    
    print(f"\n[PnL Metrics]:")
    print(f"   Total PnL: ${stats['total_pnl']:,.2f}")
    print(f"   Avg PnL: ${stats['avg_pnl']:,.2f}")
    print(f"   Avg Win: ${stats['avg_win']:,.2f}")
    print(f"   Avg Loss: ${stats['avg_loss']:,.2f}")
    
    print(f"\n[Risk Metrics]:")
    print(f"   Sharpe-like Ratio: {stats['sharpe_like']:.2f}")
    print(f"   Max Drawdown: ${stats['max_drawdown']:,.2f}")
    
    print(f"\n{'='*80}\n")


def print_comparison_report(comparison: Dict):
    """Print formatted comparison report"""
    print(f"\n{'='*80}")
    print(f"FORECAST FEATURES COMPARISON")
    print(f"{'='*80}")
    
    print_performance_report(comparison["with_forecast"])
    print_performance_report(comparison["without_forecast"])
    
    if "improvements" in comparison and comparison["improvements"]:
        improvements = comparison["improvements"]
        print(f"{'='*80}")
        print(f"IMPROVEMENT ANALYSIS")
        print(f"{'='*80}")
        
        print(f"\n[Metric Changes]:")
        print(f"   Win Rate: {improvements.get('win_rate_change', 0):.2%} ({improvements.get('win_rate_change_pct', 0):+.1f}%)")
        print(f"   Profit Factor: {improvements.get('profit_factor_change', 0):+.2f}")
        print(f"   Total PnL: ${improvements.get('total_pnl_change', 0):+,.2f}")
        print(f"   Sharpe Ratio: {improvements.get('sharpe_change', 0):+.2f}")
        
        forecast_better = improvements.get("forecast_better", False)
        print(f"\n[CONCLUSION]:")
        if forecast_better:
            print(f"   [OK] Forecast features appear to IMPROVE performance")
        else:
            print(f"   [WARN] Forecast features do NOT clearly improve performance")
            print(f"   [TIP] Consider disabling if metrics don't improve after more training")
        
        print(f"\n{'='*80}\n")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test forecast features performance")
    parser.add_argument("--check-config", action="store_true", help="Check current config settings")
    parser.add_argument("--analyze-current", action="store_true", help="Analyze current trades from journal")
    parser.add_argument("--compare", action="store_true", help="Compare with/without forecast (requires separate training runs)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("FORECAST FEATURES PERFORMANCE TEST")
    print("="*80)
    
    # Check config
    if args.check_config or not any([args.analyze_current, args.compare]):
        print("\n[CHECK] Current Configuration:")
        config_info = check_forecast_features_usage()
        
        if "error" in config_info:
            print(f"  [ERROR] {config_info['error']}")
        else:
            print(f"  Forecast Features: {'[ENABLED]' if config_info['forecast_enabled'] else '[DISABLED]'}")
            print(f"  Regime Features: {'[ENABLED]' if config_info['regime_enabled'] else '[DISABLED]'}")
            print(f"  State Features: {config_info['state_features']}")
            print(f"  Expected State Dim: {config_info['expected_state_dim']}")
            
            if config_info['state_features'] != config_info['expected_state_dim']:
                print(f"  [WARN] WARNING: State dimension mismatch!")
                print(f"     Config says {config_info['state_features']}, but should be {config_info['expected_state_dim']}")
    
    # Analyze current trades
    if args.analyze_current:
        print("\n[ANALYZE] Current Trades from Journal:")
        trades = load_trade_journal()
        
        if len(trades) == 0:
            print("  [WARN] No trades found in journal")
        else:
            config_info = check_forecast_features_usage()
            label = "With Forecast Features" if config_info.get("forecast_enabled", False) else "Without Forecast Features"
            stats = analyze_performance(trades, label)
            print_performance_report(stats)
            
            # Save to file
            output_file = project_root / "reports" / "forecast_performance_analysis.json"
            output_file.parent.mkdir(exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, default=str)
            print(f"  [SAVED] Analysis saved to: {output_file}")
    
    # Compare (requires manual setup - separate training runs)
    if args.compare:
        print("\n[COMPARE] Performance Comparison:")
        print("  [INFO] This requires separate training runs with/without forecast features")
        print("  [INFO] For now, analyzing current trades...")
        
        trades = load_trade_journal()
        if len(trades) == 0:
            print("  [WARN] No trades found in journal")
        else:
            config_info = check_forecast_features_usage()
            label = "With Forecast Features" if config_info.get("forecast_enabled", False) else "Without Forecast Features"
            stats = analyze_performance(trades, label)
            print_performance_report(stats)
            
            print("\n  [TIP] To properly compare:")
            print("     1. Train with forecast features enabled (current)")
            print("     2. Train with forecast features disabled")
            print("     3. Compare metrics from both runs")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("""
1. Monitor performance over next 1000+ trades
2. Compare key metrics:
   - Win rate (target: >50%)
   - Profit factor (target: >1.2)
   - Total PnL (should be positive)
   - Sharpe ratio (target: >1.0)

3. If forecast features don't improve metrics after sufficient training:
   - Disable via Settings panel
   - Or set include_forecast_features: false in config
   - State dimension will automatically adjust

4. Re-run this script periodically to track progress:
   python scripts/test_forecast_performance.py --analyze-current
    """)


if __name__ == "__main__":
    main()

