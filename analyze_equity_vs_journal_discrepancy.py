"""Analyze the massive discrepancy between equity curve and trading journal"""
import sqlite3
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent
db_path = project_root / "logs/trading_journal.db"

if db_path.exists():
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    print("="*80)
    print("EQUITY vs JOURNAL DISCREPANCY ANALYSIS")
    print("="*80)
    
    # 1. Check latest equity from equity_curve
    cursor.execute("""
        SELECT equity, cumulative_pnl, episode, step, timestamp
        FROM equity_curve
        ORDER BY timestamp DESC
        LIMIT 1
    """)
    latest_equity = cursor.fetchone()
    
    if latest_equity:
        equity, cum_pnl, episode, step, ts = latest_equity
        print(f"\n1. LATEST EQUITY CURVE ENTRY:")
        print(f"   Equity: ${equity:,.2f}")
        print(f"   Cumulative PnL: ${cum_pnl:,.2f}")
        print(f"   Episode: {episode}")
        print(f"   Step: {step}")
        print(f"   Timestamp: {ts}")
        print(f"   Implied Initial Capital: ${equity - cum_pnl:,.2f}")
    
    # 2. Check total net PnL from trades
    cursor.execute("""
        SELECT SUM(net_pnl) as total
        FROM trades
    """)
    total_net_pnl = cursor.fetchone()[0]
    print(f"\n2. TOTAL NET PNL FROM ALL TRADES:")
    print(f"   Sum of all net_pnl: ${total_net_pnl or 0:,.2f}")
    
    # 3. Check equity curve entries per episode
    cursor.execute("""
        SELECT episode, 
               COUNT(*) as equity_points,
               MIN(equity) as min_equity,
               MAX(equity) as max_equity,
               MIN(cumulative_pnl) as min_cum_pnl,
               MAX(cumulative_pnl) as max_cum_pnl
        FROM equity_curve
        GROUP BY episode
        ORDER BY episode DESC
        LIMIT 10
    """)
    episode_equities = cursor.fetchall()
    
    print(f"\n3. EQUITY CURVE BY EPISODE (Last 10):")
    for ep, points, min_eq, max_eq, min_cum, max_cum in episode_equities:
        print(f"   Episode {ep}: {points} points, Equity: ${min_eq:.2f}-${max_eq:.2f}, CumPnL: ${min_cum:.2f}-${max_cum:.2f}")
    
    # 4. Check trades per episode
    cursor.execute("""
        SELECT episode,
               COUNT(*) as trades,
               SUM(net_pnl) as episode_net_pnl,
               SUM(commission) as episode_commission
        FROM trades
        WHERE episode IS NOT NULL
        GROUP BY episode
        ORDER BY episode DESC
        LIMIT 10
    """)
    episode_trades = cursor.fetchall()
    
    print(f"\n4. TRADES BY EPISODE (Last 10):")
    for ep, trades, net_pnl, comm in episode_trades:
        print(f"   Episode {ep}: {trades} trades, Net PnL: ${net_pnl or 0:.2f}, Commission: ${comm or 0:.2f}")
    
    # 5. Check if equity uses per-episode PnL (which resets)
    cursor.execute("""
        SELECT episode, 
               MAX(cumulative_pnl) as max_cum_pnl
        FROM equity_curve
        WHERE episode IN (SELECT DISTINCT episode FROM equity_curve ORDER BY episode DESC LIMIT 5)
        GROUP BY episode
    """)
    recent_episode_pnls = cursor.fetchall()
    
    print(f"\n5. MAX CUMULATIVE PNL PER EPISODE (Last 5):")
    episode_pnl_sum = 0.0
    for ep, max_pnl in recent_episode_pnls:
        episode_pnl_sum += (max_pnl or 0)
        print(f"   Episode {ep}: Max Cum PnL: ${max_pnl or 0:.2f}")
    print(f"   Sum of Last 5 Episodes: ${episode_pnl_sum:.2f}")
    
    # 6. Check for duplicate trade logging
    cursor.execute("""
        SELECT timestamp, episode, step, COUNT(*) as count
        FROM trades
        GROUP BY timestamp, episode, step
        HAVING COUNT(*) > 1
        ORDER BY count DESC, timestamp DESC
        LIMIT 20
    """)
    duplicates = cursor.fetchall()
    
    if duplicates:
        print(f"\n6. DUPLICATE TRADES (same timestamp, episode, step):")
        total_duplicate_pnl = 0.0
        for ts, ep, st, count in duplicates:
            cursor.execute("""
                SELECT SUM(net_pnl) 
                FROM trades 
                WHERE timestamp = ? AND episode = ? AND step = ?
            """, (ts, ep, st))
            dup_pnl = cursor.fetchone()[0] or 0
            total_duplicate_pnl += dup_pnl * (count - 1)  # Extra duplicates
            print(f"   {ts} (Ep {ep}, Step {st}): {count} duplicates, Extra PnL: ${dup_pnl * (count - 1):.2f}")
        print(f"   Total Extra PnL from duplicates: ${total_duplicate_pnl:.2f}")
    
    # 7. Calculate what equity SHOULD be
    initial_capital = 100000.0
    expected_equity = initial_capital + (total_net_pnl or 0)
    if latest_equity:
        actual_equity = latest_equity[0]
        discrepancy = actual_equity - expected_equity
        print(f"\n7. EQUITY DISCREPANCY:")
        print(f"   Initial Capital: ${initial_capital:,.2f}")
        print(f"   Total Net PnL (all trades): ${total_net_pnl or 0:,.2f}")
        print(f"   Expected Equity: ${expected_equity:,.2f}")
        print(f"   Actual Equity (from curve): ${actual_equity:,.2f}")
        print(f"   DISCREPANCY: ${discrepancy:,.2f}")
    
    conn.close()
else:
    print(f"Database not found at {db_path}")

