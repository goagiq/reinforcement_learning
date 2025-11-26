"""Quick check for winning trades"""
import requests
import time

API_BASE = "http://localhost:8200"

# Get current metrics
perf = requests.get(f"{API_BASE}/api/monitoring/performance", timeout=5).json()
status = requests.get(f"{API_BASE}/api/training/status", timeout=5).json()

m = perf.get('metrics', {})
s = status

print("="*60)
print("CURRENT TRAINING STATUS")
print("="*60)
print(f"Training: {s.get('status', 'unknown')}")
print(f"Timestep: {s.get('timestep', 0):,}")
print(f"Episode: {s.get('episode', 0)}")
print()
print("TRADE METRICS:")
print(f"  Total Trades: {m.get('total_trades', 0)}")
print(f"  Winning Trades: {m.get('winning_trades', 0)}")
print(f"  Losing Trades: {m.get('losing_trades', 0)}")
print(f"  Win Rate: {m.get('win_rate', 0)*100:.2f}%")
print(f"  Total P&L: ${m.get('total_pnl', 0):,.2f}")
print("="*60)

if m.get('winning_trades', 0) > 0:
    print("\n✅ SUCCESS: WINNING TRADES FOUND!")
    exit(0)
else:
    print("\n⚠️  Still 0% win rate - monitoring for 5 minutes...")
    print()
    
    last_trades = m.get('total_trades', 0)
    last_wins = m.get('winning_trades', 0)
    start = time.time()
    
    while time.time() - start < 300:  # 5 minutes
        time.sleep(30)
        
        try:
            perf = requests.get(f"{API_BASE}/api/monitoring/performance", timeout=5).json()
            m = perf.get('metrics', {})
            trades = m.get('total_trades', 0)
            wins = m.get('winning_trades', 0)
            elapsed = (time.time() - start) / 60
            
            new_trades = trades - last_trades
            new_wins = wins - last_wins
            
            print(f"[{elapsed:.1f}m] Trades: {trades} | Wins: {wins} | Win Rate: {m.get('win_rate', 0)*100:.1f}% | P&L: ${m.get('total_pnl', 0):,.2f}")
            if new_trades > 0:
                print(f"  🆕 NEW: {new_trades} trades ({new_wins} new wins)")
            
            if wins > 0:
                print(f"\n✅ SUCCESS: Found {wins} winning trades!")
                exit(0)
            
            last_trades = trades
            last_wins = wins
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)
    
    print("\n⚠️  Still 0% win rate after 5 minutes")
    exit(1)

