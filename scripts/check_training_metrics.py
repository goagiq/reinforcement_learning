"""
Check if training metrics are updating properly
"""
import requests
import json
import time

def check_training_metrics():
    print("=" * 80)
    print("TRAINING METRICS DIAGNOSTIC")
    print("=" * 80)
    print()
    
    try:
        r = requests.get('http://localhost:8200/api/training/status', timeout=5)
        if r.status_code != 200:
            print(f"Error: HTTP {r.status_code}")
            return
        
        data = r.json()
        metrics = data.get('metrics', {})
        
        print("Training Status:")
        print(f"  Status: {data.get('status')}")
        print(f"  Message: {data.get('message', 'N/A')}")
        print()
        
        print("Episode Info:")
        print(f"  Current Episode: {metrics.get('episode', 'N/A')}")
        print(f"  Completed Episodes: {metrics.get('completed_episodes', 'N/A')}")
        print(f"  Current Episode Length: {metrics.get('current_episode_length', 'N/A')}")
        print(f"  Has Active Episode: {metrics.get('current_episode_length', 0) > 0}")
        print()
        
        print("Current Episode Metrics (should update during episode):")
        print(f"  Current Episode Trades: {metrics.get('current_episode_trades', 'N/A')}")
        print(f"  Current PnL: ${metrics.get('current_episode_pnl', 'N/A')}")
        print(f"  Current Equity: ${metrics.get('current_episode_equity', 'N/A')}")
        print(f"  Current Win Rate: {metrics.get('current_episode_win_rate', 'N/A')}%")
        print(f"  Current Max Drawdown: {metrics.get('current_episode_max_drawdown', 'N/A')}%")
        print()
        
        print("Latest Completed Episode Metrics:")
        print(f"  Latest Episode Length: {metrics.get('latest_episode_length', 'N/A')}")
        print(f"  Latest Reward: {metrics.get('latest_reward', 'N/A')}")
        print()
        
        print("Overall Metrics:")
        print(f"  Total Trades: {metrics.get('total_trades', 'N/A')}")
        print(f"  Overall Win Rate: {metrics.get('overall_win_rate', 'N/A')}%")
        print(f"  Mean PnL (Last 10): ${metrics.get('mean_pnl_10', 'N/A')}")
        print()
        
        # Check if values are stuck
        has_active = metrics.get('current_episode_length', 0) > 0
        if not has_active:
            print("[WARN] No active episode - values will show last completed episode")
            print("  This is normal if episode just ended or hasn't started yet")
        else:
            print("[OK] Active episode detected - values should update during training")
        
        print()
        print("=" * 80)
        print("DIAGNOSIS:")
        print("=" * 80)
        
        # Check if episode is stuck
        episode_length = metrics.get('current_episode_length', 0)
        if episode_length > 0 and episode_length < 100:
            print("[OK] Episode is active and updating")
        elif episode_length == 0:
            print("[INFO] Episode has ended or not started - values show last completed episode")
        else:
            print(f"[INFO] Episode length: {episode_length} steps")
        
        # Check if values are all zeros (might indicate reset issue)
        if has_active:
            trades = metrics.get('current_episode_trades', 0)
            pnl = metrics.get('current_episode_pnl', 0)
            if trades == 0 and pnl == 0:
                print("[WARN] Active episode but all metrics are zero - might indicate:")
                print("  - Episode just started (no trades yet)")
                print("  - Values not being updated from step_info")
                print("  - Episode reset issue")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_training_metrics()

