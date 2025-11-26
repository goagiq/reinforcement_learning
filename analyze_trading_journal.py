"""
Analyze Trading Journal to understand performance issues
"""

import sqlite3
import pandas as pd
from pathlib import Path
import numpy as np

def analyze_trades():
    """Analyze trades from trading journal"""
    db_path = Path("logs/trading_journal.db")
    
    if not db_path.exists():
        print(f"❌ Trading journal not found at {db_path}")
        return
    
    conn = sqlite3.connect(str(db_path))
    
    # Get all trades
    query = """
        SELECT 
            trade_id, timestamp, episode, step,
            entry_price, exit_price, position_size,
            pnl, commission, net_pnl, is_win,
            strategy, strategy_confidence, duration_steps
        FROM trades
        ORDER BY timestamp ASC
    """
    
    trades_df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(trades_df) == 0:
        print("❌ No trades found in journal")
        return
    
    print(f"\n{'='*70}")
    print(f"TRADING JOURNAL ANALYSIS")
    print(f"{'='*70}\n")
    
    print(f"Total Trades: {len(trades_df)}")
    print(f"Date Range: {trades_df['timestamp'].min()} to {trades_df['timestamp'].max()}\n")
    
    # Basic statistics
    winning_trades = trades_df[trades_df['is_win'] == 1]
    losing_trades = trades_df[trades_df['is_win'] == 0]
    
    print(f"{'='*70}")
    print(f"WIN/LOSS STATISTICS")
    print(f"{'='*70}")
    print(f"Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(trades_df)*100:.2f}%)")
    print(f"Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(trades_df)*100:.2f}%)")
    
    if len(winning_trades) > 0:
        print(f"\nAverage Win: ${winning_trades['net_pnl'].mean():.2f}")
        print(f"Max Win: ${winning_trades['net_pnl'].max():.2f}")
        print(f"Min Win: ${winning_trades['net_pnl'].min():.2f}")
    
    if len(losing_trades) > 0:
        print(f"\nAverage Loss: ${losing_trades['net_pnl'].mean():.2f}")
        print(f"Max Loss: ${losing_trades['net_pnl'].max():.2f}")
        print(f"Min Loss: ${losing_trades['net_pnl'].min():.2f}")
    
    # PnL statistics
    print(f"\n{'='*70}")
    print(f"P&L STATISTICS")
    print(f"{'='*70}")
    print(f"Total P&L: ${trades_df['net_pnl'].sum():.2f}")
    print(f"Average Trade: ${trades_df['net_pnl'].mean():.2f}")
    
    gross_profit = winning_trades['net_pnl'].sum() if len(winning_trades) > 0 else 0.0
    gross_loss = abs(losing_trades['net_pnl'].sum()) if len(losing_trades) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    print(f"Gross Profit: ${gross_profit:.2f}")
    print(f"Gross Loss: ${gross_loss:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    
    if len(winning_trades) > 0 and len(losing_trades) > 0:
        avg_win = winning_trades['net_pnl'].mean()
        avg_loss = abs(losing_trades['net_pnl'].mean())
        risk_reward = avg_win / avg_loss if avg_loss > 0 else 0.0
        print(f"Risk/Reward Ratio: {risk_reward:.2f}")
    
    # Commission impact
    total_commission = trades_df['commission'].sum()
    print(f"\nTotal Commission: ${total_commission:.2f}")
    print(f"Commission as % of P&L: {(total_commission/abs(trades_df['net_pnl'].sum())*100):.2f}%" if trades_df['net_pnl'].sum() != 0 else "N/A")
    
    # Trade size analysis
    print(f"\n{'='*70}")
    print(f"POSITION SIZE ANALYSIS")
    print(f"{'='*70}")
    print(f"Average Position Size: {trades_df['position_size'].mean():.4f}")
    print(f"Min Position Size: {trades_df['position_size'].min():.4f}")
    print(f"Max Position Size: {trades_df['position_size'].max():.4f}")
    
    # Strategy breakdown
    print(f"\n{'='*70}")
    print(f"STRATEGY BREAKDOWN")
    print(f"{'='*70}")
    strategy_stats = trades_df.groupby('strategy').agg({
        'trade_id': 'count',
        'net_pnl': ['sum', 'mean'],
        'is_win': 'sum'
    }).round(2)
    print(strategy_stats)
    
    # Recent trades
    print(f"\n{'='*70}")
    print(f"RECENT 10 TRADES (Most Recent First)")
    print(f"{'='*70}")
    recent = trades_df.tail(10)[::-1]  # Reverse to show most recent first
    for idx, trade in recent.iterrows():
        win_loss = "WIN" if trade['is_win'] == 1 else "LOSS"
        print(f"Trade {trade['trade_id']}: {win_loss} | "
              f"Entry: ${trade['entry_price']:.2f} → Exit: ${trade['exit_price']:.2f} | "
              f"PnL: ${trade['net_pnl']:.2f} | "
              f"Size: {trade['position_size']:.4f} | "
              f"Episode: {trade['episode']}")
    
    # Episode breakdown
    print(f"\n{'='*70}")
    print(f"EPISODE BREAKDOWN (Top 10 by P&L)")
    print(f"{'='*70}")
    episode_stats = trades_df.groupby('episode').agg({
        'trade_id': 'count',
        'net_pnl': 'sum'
    }).sort_values('net_pnl', ascending=False).head(10)
    print(episode_stats)
    
    # Price analysis
    print(f"\n{'='*70}")
    print(f"PRICE MOVEMENT ANALYSIS")
    print(f"{'='*70}")
    trades_df['price_change'] = trades_df['exit_price'] - trades_df['entry_price']
    trades_df['price_change_pct'] = (trades_df['price_change'] / trades_df['entry_price']) * 100
    
    print(f"Average Price Change: ${trades_df['price_change'].mean():.2f} ({trades_df['price_change_pct'].mean():.2f}%)")
    print(f"Average Price Change (Wins): ${winning_trades['price_change'].mean():.2f} ({winning_trades['price_change_pct'].mean():.2f}%)" if len(winning_trades) > 0 else "No wins")
    print(f"Average Price Change (Losses): ${losing_trades['price_change'].mean():.2f} ({losing_trades['price_change_pct'].mean():.2f}%)" if len(losing_trades) > 0 else "No losses")
    
    # Check for patterns
    print(f"\n{'='*70}")
    print(f"POTENTIAL ISSUES")
    print(f"{'='*70}")
    
    if profit_factor < 1.0:
        print(f"⚠️  Profit Factor < 1.0: Agent is losing more than winning")
    
    if len(winning_trades) > 0 and len(losing_trades) > 0:
        avg_win = winning_trades['net_pnl'].mean()
        avg_loss = abs(losing_trades['net_pnl'].mean())
        if avg_loss > avg_win:
            print(f"⚠️  Average Loss (${avg_loss:.2f}) > Average Win (${avg_win:.2f})")
    
    # Check if losses are due to commissions
    if total_commission > abs(gross_profit - gross_loss):
        print(f"⚠️  Commission costs (${total_commission:.2f}) are significant relative to net P&L")
    
    # Check position sizing
    if trades_df['position_size'].std() < 0.01:
        print(f"⚠️  Position sizes are very consistent (std={trades_df['position_size'].std():.4f}) - may indicate over-regularization")
    
    # Check duration
    if 'duration_steps' in trades_df.columns:
        avg_duration = trades_df['duration_steps'].mean()
        print(f"\nAverage Trade Duration: {avg_duration:.1f} steps")
        if avg_duration < 5:
            print(f"⚠️  Very short average trade duration ({avg_duration:.1f} steps) - may be overtrading")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    analyze_trades()

