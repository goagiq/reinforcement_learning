"""
Check backend activity: API server status, training status, and recent activity
"""

import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime

def check_api_server():
    """Check if API server is running and get training status"""
    print("=" * 80)
    print("BACKEND ACTIVITY CHECK")
    print("=" * 80)
    
    try:
        import requests
        r = requests.get('http://localhost:8000/api/training/status', timeout=2)
        print("\n[1/4] API Server Status: RUNNING")
        print(f"   Status Code: {r.status_code}")
        
        data = r.json()
        status = data.get('status', 'N/A')
        print(f"\n[2/4] Training Status: {status}")
        
        if status != 'idle':
            print(f"   Current Episode: {data.get('current_episode', 'N/A')}")
            print(f"   Timesteps: {data.get('timestep', 'N/A'):,}" if data.get('timestep') else "   Timesteps: N/A")
            print(f"   Current Episode Trades: {data.get('current_episode_trades', 'N/A')}")
            print(f"   Current PnL: ${data.get('current_pnl', 0):.2f}" if data.get('current_pnl') else "   Current PnL: N/A")
            print(f"   Current Equity: ${data.get('current_equity', 0):.2f}" if data.get('current_equity') else "   Current Equity: N/A")
            
            # Get performance data
            try:
                perf_r = requests.get('http://localhost:8000/api/performance', timeout=2)
                perf_data = perf_r.json()
                print(f"\n   Performance Summary:")
                print(f"   - Total Trades: {perf_data.get('total_trades', 'N/A')}")
                if perf_data.get('win_rate'):
                    print(f"   - Win Rate: {perf_data.get('win_rate', 0)*100:.1f}%")
                if perf_data.get('total_pnl'):
                    print(f"   - Total PnL: ${perf_data.get('total_pnl', 0):.2f}")
            except:
                pass
        else:
            print("   No training in progress")
        
        return True
    except requests.exceptions.ConnectionError:
        print("\n[1/4] API Server Status: NOT RUNNING (connection refused)")
        print("   The backend server is not running on port 8000")
        return False
    except Exception as e:
        print(f"\n[1/4] API Server Status: ERROR - {e}")
        return False

def check_database():
    """Check trading journal database for recent activity"""
    print("\n[3/4] Database Activity:")
    
    db_path = Path('data/trading_journal.db')
    if not db_path.exists():
        print("   Database not found")
        return False
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get total counts
        cursor.execute('SELECT COUNT(*) FROM trades')
        total_trades = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM episodes')
        total_episodes = cursor.fetchone()[0]
        
        print(f"   Total Trades: {total_trades}")
        print(f"   Total Episodes: {total_episodes}")
        
        # Get recent episodes
        cursor.execute('SELECT episode, total_trades, total_pnl, timestamp FROM episodes ORDER BY episode DESC LIMIT 5')
        recent_episodes = cursor.fetchall()
        
        if recent_episodes:
            print(f"\n   Recent Episodes (last 5):")
            for ep in recent_episodes:
                episode_num, trades, pnl, timestamp = ep
                print(f"   - Episode {episode_num}: {trades} trades, PnL=${pnl:.2f}, Time: {timestamp}")
        
        # Get recent trades
        cursor.execute('SELECT COUNT(*) FROM trades WHERE timestamp > datetime("now", "-1 hour")')
        recent_trades = cursor.fetchone()[0]
        print(f"\n   Trades in last hour: {recent_trades}")
        
        conn.close()
        return True
    except Exception as e:
        print(f"   Error accessing database: {e}")
        return False

def check_adaptive_learning():
    """Check adaptive learning adjustments"""
    print("\n[4/4] Adaptive Learning Activity:")
    
    adj_file = Path('logs/adaptive_training/config_adjustments.jsonl')
    if not adj_file.exists():
        print("   No adaptive learning adjustments file found")
        return False
    
    try:
        with open(adj_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            print("   No adjustments recorded")
            return False
        
        print(f"   Total adjustments: {len(lines)}")
        print(f"\n   Recent adjustments (last 5):")
        
        for line in lines[-5:]:
            try:
                data = json.loads(line.strip())
                timestamp = data.get('timestamp', 'N/A')
                episode = data.get('episode', 'N/A')
                adjustments = data.get('adjustments', {})
                adj_keys = list(adjustments.keys())
                print(f"   - [{timestamp}] Episode {episode}: {', '.join(adj_keys)}")
            except:
                pass
        
        return True
    except Exception as e:
        print(f"   Error reading adjustments: {e}")
        return False

def main():
    """Run all checks"""
    api_running = check_api_server()
    check_database()
    check_adaptive_learning()
    
    print("\n" + "=" * 80)
    if api_running:
        print("SUMMARY: Backend is active and running")
    else:
        print("SUMMARY: Backend server is not running")
    print("=" * 80)

if __name__ == "__main__":
    main()

