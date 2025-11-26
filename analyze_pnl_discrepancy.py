"""Analyze PnL discrepancy between Training Progress and Performance Monitoring"""
import sqlite3
from pathlib import Path

project_root = Path(__file__).parent
db_path = project_root / "logs/trading_journal.db"

if db_path.exists():
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Get summary statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as total_trades,
            SUM(pnl) as gross_pnl,
            SUM(commission) as total_commission,
            SUM(net_pnl) as net_pnl_sum,
            SUM(pnl - commission) as calculated_net_pnl,
            AVG(pnl) as avg_gross_pnl,
            AVG(commission) as avg_commission,
            AVG(net_pnl) as avg_net_pnl
        FROM trades
    """)
    
    summary = cursor.fetchone()
    if summary:
        total_trades, gross_pnl, total_commission, net_pnl_sum, calculated_net, avg_gross, avg_comm, avg_net = summary
        
        print("="*80)
        print("PNL DISCREPANCY ANALYSIS")
        print("="*80)
        print(f"\nTotal Trades: {total_trades or 0:,}")
        print(f"\nGross PnL (sum of pnl column): ${gross_pnl or 0:,.2f}")
        print(f"Total Commission (sum of commission column): ${total_commission or 0:,.2f}")
        print(f"Net PnL (sum of net_pnl column): ${net_pnl_sum or 0:,.2f}")
        print(f"Calculated Net PnL (sum of pnl - commission): ${calculated_net or 0:,.2f}")
        print(f"\nDifference: ${(net_pnl_sum or 0) - (calculated_net or 0):,.2f}")
        
        print(f"\nAverage Gross PnL: ${avg_gross or 0:,.2f}")
        print(f"Average Commission: ${avg_comm or 0:,.2f}")
        print(f"Average Net PnL: ${avg_net or 0:,.2f}")
        
        # Check if commission is being double-counted
        expected_net = (gross_pnl or 0) - (total_commission or 0)
        print(f"\nExpected Net PnL (gross - total commission): ${expected_net:,.2f}")
        print(f"Actual Net PnL from database: ${net_pnl_sum or 0:,.2f}")
        print(f"Discrepancy: ${(net_pnl_sum or 0) - expected_net:,.2f}")
        
        # Check for trades where net_pnl doesn't match pnl - commission
        cursor.execute("""
            SELECT COUNT(*) 
            FROM trades 
            WHERE ABS(net_pnl - (pnl - commission)) > 0.01
        """)
        mismatch_count = cursor.fetchone()[0]
        print(f"\nTrades with net_pnl mismatch (>$0.01): {mismatch_count}")
        
        if mismatch_count > 0:
            print("\nSample mismatches:")
            cursor.execute("""
                SELECT pnl, commission, net_pnl, (pnl - commission) as calculated, 
                       ABS(net_pnl - (pnl - commission)) as diff
                FROM trades 
                WHERE ABS(net_pnl - (pnl - commission)) > 0.01
                LIMIT 10
            """)
            for row in cursor.fetchall():
                print(f"  PnL: ${row[0]:.2f}, Commission: ${row[1]:.2f}, "
                      f"Net PnL (DB): ${row[2]:.2f}, Calculated: ${row[3]:.2f}, "
                      f"Diff: ${row[4]:.2f}")
    
    # Check recent trades
    print("\n" + "="*80)
    print("RECENT 10 TRADES")
    print("="*80)
    cursor.execute("""
        SELECT timestamp, pnl, commission, net_pnl, (pnl - commission) as calc_net
        FROM trades
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    for row in cursor.fetchall():
        ts, pnl, comm, net, calc = row
        match = "OK" if abs(net - calc) < 0.01 else "MISMATCH"
        print(f"{ts}: PnL=${pnl:.2f}, Comm=${comm:.2f}, Net=${net:.2f}, Calc=${calc:.2f} [{match}]")
    
    conn.close()
else:
    print(f"Database not found at {db_path}")

