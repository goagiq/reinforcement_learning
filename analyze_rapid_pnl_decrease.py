"""Analyze why P&L is decreasing so rapidly"""
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

project_root = Path(__file__).parent
db_path = project_root / "logs/trading_journal.db"

if db_path.exists():
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Get trades from last hour
    one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
    
    print("="*80)
    print("RAPID P&L DECREASE ANALYSIS")
    print("="*80)
    
    # Get recent trades
    cursor.execute("""
        SELECT timestamp, pnl, commission, net_pnl, position_size
        FROM trades
        WHERE timestamp >= ?
        ORDER BY timestamp DESC
        LIMIT 100
    """, (one_hour_ago,))
    
    recent_trades = cursor.fetchall()
    
    if recent_trades:
        print(f"\nFound {len(recent_trades)} trades in last hour")
        
        # Calculate statistics
        # row format: (timestamp, pnl, commission, net_pnl, position_size)
        total_gross = sum(row[1] for row in recent_trades)  # pnl (index 1) = gross PnL
        total_commission = sum(row[2] for row in recent_trades)  # commission (index 2)
        total_pnl = sum(row[3] for row in recent_trades)  # net_pnl (index 3) = gross - commission
        
        print(f"\nTotal Gross PnL: ${total_gross:,.2f}")
        print(f"Total Commission: ${total_commission:,.2f}")
        print(f"Total Net PnL: ${total_pnl:,.2f}")
        
        # Check commission per trade
        commissions = [row[2] for row in recent_trades]
        avg_commission = sum(commissions) / len(commissions) if commissions else 0
        print(f"\nAverage Commission per Trade: ${avg_commission:.2f}")
        
        # Check if commission seems too high
        if avg_commission > 20:
            print(f"[WARNING] Average commission (${avg_commission:.2f}) seems very high!")
        
        # Check trade frequency
        if len(recent_trades) > 0:
            first_trade_time = datetime.fromisoformat(recent_trades[-1][0])
            last_trade_time = datetime.fromisoformat(recent_trades[0][0])
            time_span = (last_trade_time - first_trade_time).total_seconds() / 60  # minutes
            trades_per_minute = len(recent_trades) / max(time_span, 1)
            print(f"\nTrade Frequency: {trades_per_minute:.2f} trades/minute")
            
            if trades_per_minute > 1:
                print(f"⚠️ WARNING: Very high trade frequency ({trades_per_minute:.2f} trades/min)")
        
        # Show sample trades
        print("\n" + "="*80)
        print("SAMPLE RECENT TRADES (Last 10)")
        print("="*80)
        for i, row in enumerate(recent_trades[:10]):
            ts, pnl, comm, net_pnl, pos_size = row
            print(f"{i+1}. {ts}: Gross=${pnl:.2f}, Comm=${comm:.2f}, Net=${net_pnl:.2f}, Size={pos_size:.3f}")
    
    # Check for trades with suspiciously high commission
    cursor.execute("""
        SELECT COUNT(*) 
        FROM trades
        WHERE commission > 50
    """)
    high_comm_count = cursor.fetchone()[0]
    if high_comm_count > 0:
        print(f"\n⚠️ WARNING: {high_comm_count} trades have commission > $50")
    
    conn.close()
else:
    print(f"Database not found at {db_path}")

