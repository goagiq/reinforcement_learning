"""Check recent trades to see win rate"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

conn = sqlite3.connect('logs/trading_journal.db')

# Check last 30 minutes
cutoff = (datetime.now() - timedelta(minutes=30)).isoformat()
df = pd.read_sql_query(
    'SELECT timestamp, entry_price, exit_price, position_size, pnl, net_pnl, is_win FROM trades WHERE timestamp >= ? ORDER BY timestamp DESC',
    conn,
    params=(cutoff,)
)

conn.close()

print(f"Trades in last 30 minutes: {len(df)}")
if len(df) > 0:
    wins = len(df[df['is_win'] == 1])
    print(f"Winning trades: {wins}")
    print(f"Win rate: {wins/len(df)*100:.1f}%")
    print()
    print("Recent trades:")
    print(df.head(20)[['timestamp', 'entry_price', 'exit_price', 'net_pnl', 'is_win']].to_string())
else:
    print("No trades in last 30 minutes")

