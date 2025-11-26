"""
Get performance metrics since checkpoint 1000000 was loaded
"""
import sqlite3
import requests
import json

def get_checkpoint_performance():
    # Connect to trading journal
    conn = sqlite3.connect('logs/trading_journal.db')
    cursor = conn.cursor()
    
    # Get first trade timestamp (when training started after loading checkpoint)
    cursor.execute('SELECT MIN(timestamp) FROM trades')
    first_trade_time = cursor.fetchone()[0]
    
    conn.close()
    
    if not first_trade_time:
        print("No trades found in journal")
        return
    
    print("=" * 80)
    print("PERFORMANCE SINCE CHECKPOINT 1000000")
    print("=" * 80)
    print(f"\nCheckpoint 1000000 was loaded and training started at:")
    print(f"  {first_trade_time}")
    print()
    
    # Get performance metrics
    url = f"http://localhost:8200/api/monitoring/performance?since={first_trade_time}"
    print(f"Fetching: {url}")
    print()
    
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            metrics = data.get('metrics', {})
            
            print("Performance Metrics (Since Checkpoint 1000000):")
            print("-" * 80)
            print(f"Total Trades: {metrics.get('total_trades', 0):,}")
            print(f"Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")
            print(f"Total PnL: ${metrics.get('total_pnl', 0):,.2f}")
            print(f"Average Trade: ${metrics.get('average_trade', 0):.2f}")
            print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: ${metrics.get('max_drawdown', 0):,.2f}")
            print(f"Risk/Reward Ratio: {metrics.get('risk_reward_ratio', 0):.2f}")
            print(f"Mean PnL (Last 10): ${metrics.get('mean_pnl_10', 0):.2f}")
            print()
            print("=" * 80)
            print("CURL COMMAND:")
            print("=" * 80)
            print()
            print("PowerShell:")
            print(f'python -c "import requests; import json; r = requests.get(\'http://localhost:8200/api/monitoring/performance?since={first_trade_time}\'); print(json.dumps(r.json(), indent=2))"')
            print()
            print("Linux/Mac:")
            print(f'curl "http://localhost:8200/api/monitoring/performance?since={first_trade_time}" | python -m json.tool')
            print()
        else:
            print(f"Error: HTTP {r.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_checkpoint_performance()

