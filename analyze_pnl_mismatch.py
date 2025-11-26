"""Analyze P&L mismatch between equity curve, trading journal, and Performance Monitor"""
import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd

project_root = Path(__file__).parent
db_path = project_root / "logs/trading_journal.db"

if db_path.exists():
    conn = sqlite3.connect(str(db_path))
    
    print("="*80)
    print("PNL MISMATCH ANALYSIS")
    print("="*80)
    
    # 1. Check total P&L from trades table
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            COUNT(*) as total_trades,
            SUM(pnl) as total_gross_pnl,
            SUM(commission) as total_commission,
            SUM(net_pnl) as total_net_pnl,
            MIN(timestamp) as first_trade,
            MAX(timestamp) as last_trade
        FROM trades
    """)
    trade_summary = cursor.fetchone()
    
    if trade_summary:
        total_trades, gross_pnl, total_comm, net_pnl, first_trade, last_trade = trade_summary
        print(f"\n1. TRADING JOURNAL SUMMARY (trades table):")
        print(f"   Total Trades: {total_trades or 0:,}")
        print(f"   Total Gross PnL: ${gross_pnl or 0:,.2f}")
        print(f"   Total Commission: ${total_comm or 0:,.2f}")
        print(f"   Total Net PnL: ${net_pnl or 0:,.2f}")
        print(f"   First Trade: {first_trade}")
        print(f"   Last Trade: {last_trade}")
    
    # 2. Check equity curve
    cursor.execute("""
        SELECT 
            COUNT(*) as equity_points,
            MIN(equity) as min_equity,
            MAX(equity) as max_equity,
            equity
        FROM equity_curve
        ORDER BY timestamp DESC
        LIMIT 1
    """)
    equity_result = cursor.fetchone()
    
    if equity_result:
        equity_points, min_equity, max_equity, latest_equity = equity_result
        print(f"\n2. EQUITY CURVE SUMMARY:")
        print(f"   Total Equity Points: {equity_points or 0:,}")
        print(f"   Min Equity: ${min_equity or 0:,.2f}")
        print(f"   Max Equity: ${max_equity or 0:,.2f}")
        print(f"   Latest Equity: ${latest_equity or 0:,.2f}")
        
        # Calculate change from initial capital
        initial_capital = 100000.0
        equity_change = (latest_equity or initial_capital) - initial_capital
        print(f"   Equity Change: ${equity_change:,.2f}")
        
        # Check if equity matches net PnL
        expected_equity = initial_capital + (net_pnl or 0)
        equity_diff = (latest_equity or initial_capital) - expected_equity
        print(f"   Expected Equity (100K + net PnL): ${expected_equity:,.2f}")
        print(f"   Equity Difference: ${equity_diff:,.2f}")
        if abs(equity_diff) > 100:
            print(f"   [WARNING] Equity doesn't match net PnL! Difference: ${equity_diff:,.2f}")
    
    # 3. Check for duplicate trades or double counting
    cursor.execute("""
        SELECT 
            timestamp,
            COUNT(*) as count
        FROM trades
        GROUP BY timestamp
        HAVING COUNT(*) > 1
        ORDER BY count DESC
        LIMIT 10
    """)
    duplicates = cursor.fetchall()
    
    if duplicates:
        print(f"\n3. DUPLICATE TRADES (same timestamp):")
        for ts, count in duplicates:
            print(f"   {ts}: {count} trades")
    else:
        print(f"\n3. DUPLICATE TRADES: None found")
    
    # 4. Check cumulative P&L calculation
    cursor.execute("""
        SELECT 
            timestamp,
            pnl,
            commission,
            net_pnl
        FROM trades
        ORDER BY timestamp ASC
        LIMIT 10
    """)
    first_trades = cursor.fetchall()
    
    if first_trades:
        print(f"\n4. FIRST 10 TRADES (chronological):")
        cumulative_pnl = 0.0
        cumulative_commission = 0.0
        for i, (ts, pnl, comm, net) in enumerate(first_trades[:10]):
            cumulative_pnl += (net or 0)
            cumulative_commission += (comm or 0)
            print(f"   {i+1}. {ts}: PnL=${pnl:.2f}, Comm=${comm:.2f}, Net=${net:.2f}, Cumulative=${cumulative_pnl:.2f}")
    
    # 5. Check recent trades vs cumulative
    cursor.execute("""
        SELECT 
            timestamp,
            pnl,
            commission,
            net_pnl
        FROM trades
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    recent_trades = cursor.fetchall()
    
    if recent_trades:
        print(f"\n5. LAST 10 TRADES (most recent):")
        for i, (ts, pnl, comm, net) in enumerate(recent_trades[:10]):
            print(f"   {i+1}. {ts}: PnL=${pnl:.2f}, Comm=${comm:.2f}, Net=${net:.2f}")
    
    # 6. Check if net_pnl = pnl - commission
    cursor.execute("""
        SELECT COUNT(*) 
        FROM trades
        WHERE ABS(net_pnl - (pnl - commission)) > 0.01
    """)
    mismatch_count = cursor.fetchone()[0]
    
    if mismatch_count > 0:
        print(f"\n6. NET_PNL MISMATCH:")
        print(f"   {mismatch_count} trades where net_pnl != (pnl - commission)")
        
        cursor.execute("""
            SELECT timestamp, pnl, commission, net_pnl, (pnl - commission) as calculated
            FROM trades
            WHERE ABS(net_pnl - (pnl - commission)) > 0.01
            LIMIT 10
        """)
        mismatches = cursor.fetchall()
        for ts, pnl, comm, net, calc in mismatches:
            print(f"   {ts}: PnL=${pnl:.2f}, Comm=${comm:.2f}, Net=${net:.2f}, Calc=${calc:.2f}, Diff=${abs(net-calc):.2f}")
    else:
        print(f"\n6. NET_PNL CALCULATION: OK (all trades match)")
    
    # 7. Check for trades with zero or missing values
    cursor.execute("""
        SELECT COUNT(*) 
        FROM trades
        WHERE pnl IS NULL OR commission IS NULL OR net_pnl IS NULL
    """)
    null_count = cursor.fetchone()[0]
    
    if null_count > 0:
        print(f"\n7. NULL VALUES: {null_count} trades have NULL pnl/commission/net_pnl")
    else:
        print(f"\n7. NULL VALUES: None found")
    
    # 8. Calculate cumulative net PnL step by step
    cursor.execute("""
        SELECT net_pnl
        FROM trades
        ORDER BY timestamp ASC
    """)
    all_net_pnls = [row[0] for row in cursor.fetchall() if row[0] is not None]
    
    if all_net_pnls:
        cumulative_by_sum = sum(all_net_pnls)
        print(f"\n8. CUMULATIVE CALCULATION:")
        print(f"   Sum of all net_pnl: ${cumulative_by_sum:,.2f}")
        print(f"   Database SUM(net_pnl): ${net_pnl or 0:,.2f}")
        if abs(cumulative_by_sum - (net_pnl or 0)) > 0.01:
            print(f"   [ERROR] Mismatch! Difference: ${abs(cumulative_by_sum - (net_pnl or 0)):,.2f}")
    
    conn.close()
else:
    print(f"Database not found at {db_path}")

