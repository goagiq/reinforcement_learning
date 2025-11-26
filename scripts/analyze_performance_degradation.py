"""
Analyze performance degradation after forecast caching changes
"""
import sqlite3
import pandas as pd
from pathlib import Path

def analyze_performance():
    db_path = Path("logs/trading_journal.db")
    if not db_path.exists():
        print("Trading journal not found")
        return
    
    conn = sqlite3.connect(str(db_path))
    
    # Get all trades
    df = pd.read_sql("""
        SELECT timestamp, net_pnl, is_win, position_size, entry_price, exit_price
        FROM trades
        ORDER BY timestamp DESC
    """, conn)
    
    if len(df) == 0:
        print("No trades found")
        conn.close()
        return
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Analyze recent performance (last 100 trades)
    recent = df.head(100)
    
    print("=" * 80)
    print("PERFORMANCE ANALYSIS - Recent 100 Trades")
    print("=" * 80)
    print(f"\nTotal Trades: {len(recent)}")
    print(f"Win Rate: {recent['is_win'].mean()*100:.1f}%")
    print(f"Total PnL: ${recent['net_pnl'].sum():.2f}")
    print(f"Average PnL: ${recent['net_pnl'].mean():.2f}")
    print(f"Winning Trades: {recent['is_win'].sum()}")
    print(f"Losing Trades: {(~recent['is_win'].astype(bool)).sum()}")
    
    if len(recent) > 0:
        wins = recent[recent['is_win'] == 1]
        losses = recent[recent['is_win'] == 0]
        if len(wins) > 0:
            print(f"Average Win: ${wins['net_pnl'].mean():.2f}")
        if len(losses) > 0:
            print(f"Average Loss: ${losses['net_pnl'].mean():.2f}")
            if len(wins) > 0:
                profit_factor = wins['net_pnl'].sum() / abs(losses['net_pnl'].sum())
                print(f"Profit Factor: {profit_factor:.2f}")
    
    # Compare last 20 vs previous 20
    if len(recent) >= 40:
        last_20 = recent.head(20)
        prev_20 = recent.iloc[20:40]
        
        print("\n" + "=" * 80)
        print("COMPARISON: Last 20 vs Previous 20 Trades")
        print("=" * 80)
        print(f"\nLast 20 Trades:")
        print(f"  Win Rate: {last_20['is_win'].mean()*100:.1f}%")
        print(f"  Total PnL: ${last_20['net_pnl'].sum():.2f}")
        print(f"  Avg PnL: ${last_20['net_pnl'].mean():.2f}")
        
        print(f"\nPrevious 20 Trades:")
        print(f"  Win Rate: {prev_20['is_win'].mean()*100:.1f}%")
        print(f"  Total PnL: ${prev_20['net_pnl'].sum():.2f}")
        print(f"  Avg PnL: ${prev_20['net_pnl'].mean():.2f}")
        
        pnl_diff = last_20['net_pnl'].sum() - prev_20['net_pnl'].sum()
        print(f"\nChange: ${pnl_diff:.2f} ({pnl_diff/abs(prev_20['net_pnl'].sum())*100 if prev_20['net_pnl'].sum() != 0 else 0:.1f}%)")
    
    # Check for large losses
    large_losses = recent[recent['net_pnl'] < -500]
    if len(large_losses) > 0:
        print("\n" + "=" * 80)
        print(f"LARGE LOSSES (>$500): {len(large_losses)} trades")
        print("=" * 80)
        for idx, trade in large_losses.iterrows():
            print(f"  {trade['timestamp']}: ${trade['net_pnl']:.2f} (Size: {trade['position_size']:.2f})")
    
    conn.close()

if __name__ == "__main__":
    analyze_performance()

