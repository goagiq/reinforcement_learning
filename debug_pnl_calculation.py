"""
Debug script to check how PnL is being calculated in training vs monitoring
"""

import requests
import json

def check_metrics():
    """Check both training status and monitoring performance endpoints"""
    
    print("=" * 80)
    print("PNL CALCULATION DEBUG")
    print("=" * 80)
    print()
    
    # Get training status
    print("1. TRAINING STATUS (from /api/training/status):")
    print("-" * 80)
    try:
        response = requests.get("http://localhost:8200/api/training/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            metrics = data.get("metrics", {})
            
            print(f"Current Episode PnL: ${metrics.get('current_episode_pnl', 0):.2f}")
            print(f"Mean PnL (Last 10 Episodes): ${metrics.get('mean_pnl_10', 0):.2f}")
            print(f"Total Trades: {metrics.get('total_trades', 0)}")
            print(f"Overall Win Rate: {metrics.get('overall_win_rate', 0):.1f}%")
            
            # Show episode PnLs if available
            if "episode_pnls" in str(data):
                print(f"\nNote: Episode PnLs are stored per episode")
                print(f"Total PnL = sum of all episode PnLs")
        else:
            print(f"Error: Status code {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")
    
    print()
    print("2. PERFORMANCE MONITORING (from /api/monitoring/performance):")
    print("-" * 80)
    try:
        response = requests.get("http://localhost:8200/api/monitoring/performance", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                metrics = data.get("metrics", {})
                source = metrics.get("source", "unknown")
                
                print(f"Source: {source}")
                print(f"Total P&L: ${metrics.get('total_pnl', 0):.2f}")
                print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
                print(f"Win Rate: {metrics.get('win_rate', 0):.4f} ({metrics.get('win_rate', 0)*100:.1f}%)")
                print(f"Profit Factor: {metrics.get('profit_factor', 0):.4f}")
                print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.4f} ({metrics.get('max_drawdown', 0)*100:.2f}%)")
                print(f"Total Trades: {metrics.get('total_trades', 0)}")
                print(f"Average Trade: ${metrics.get('average_trade', 0):.2f}")
                
                if source == "training":
                    print(f"\nMean PnL (Last 10): ${metrics.get('mean_pnl_10', 0):.2f}")
                    print(f"Risk/Reward Ratio: {metrics.get('risk_reward_ratio', 0):.4f}")
                    print("\n[EXPLANATION]")
                    print("  Total P&L = Sum of all episode PnLs (cumulative across all episodes)")
                    print("  This is different from:")
                    print("    - Current Episode PnL (just the current episode)")
                    print("    - Mean PnL (Last 10) (average of recent episodes)")
            else:
                print(f"Error: {data.get('message', 'Unknown error')}")
        else:
            print(f"Error: Status code {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Performance Monitoring shows:
  - Total P&L: Cumulative sum of ALL episode PnLs (all episodes combined)
  - This is the total profit/loss across the entire training run

Training Progress shows:
  - Current Episode PnL: Just the current (in-progress) episode
  - Mean PnL (Last 10): Average of the last 10 completed episodes
  
These are different metrics:
  - Total P&L (Monitoring) = Sum of all episode PnLs (can be positive even if recent episodes are negative)
  - Current Episode PnL (Training) = Just the current episode (can be negative)
  - Mean PnL (Last 10) = Average of recent episodes (shows recent trend)
  
Example:
  - If you had 100 episodes with PnLs: [100, 200, -50, -30, -10, ...]
  - Total P&L = 100 + 200 - 50 - 30 - 10 + ... = 34,065.22 (positive)
  - Current Episode PnL = -10.96 (negative, current episode is losing)
  - Mean PnL (Last 10) = -410.20 (negative, recent episodes are losing money)
  
This is normal - early episodes may have been profitable, but recent episodes are losing.
""")

if __name__ == "__main__":
    check_metrics()

