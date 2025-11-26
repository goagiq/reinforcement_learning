"""
Analyze performance before and after checkpoint resume
"""
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

def analyze_performance():
    db_path = Path("logs/trading_journal.db")
    if not db_path.exists():
        print("[ERROR] Trading journal database not found")
        return
    
    conn = sqlite3.connect(str(db_path))
    
    # Get all trades
    query = """
        SELECT 
            trade_id, timestamp, episode, step, 
            pnl, net_pnl, is_win, strategy
        FROM trades
        ORDER BY timestamp ASC
    """
    trades_df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(trades_df) == 0:
        print("[ERROR] No trades found in journal")
        return
    
    print(f"[INFO] Total trades in journal: {len(trades_df)}")
    print(f"   First trade: {trades_df['timestamp'].min()}")
    print(f"   Last trade: {trades_df['timestamp'].max()}")
    print(f"   Episodes: {trades_df['episode'].min()} to {trades_df['episode'].max()}")
    print()
    
    # Try to identify checkpoint resume point
    # Look for gaps in episode numbers or timestamps
    trades_df['timestamp_dt'] = pd.to_datetime(trades_df['timestamp'])
    trades_df = trades_df.sort_values('timestamp_dt')
    
    # Calculate time gaps between trades
    trades_df['time_gap'] = trades_df['timestamp_dt'].diff()
    
    # Find large gaps (potential checkpoint resume)
    large_gaps = trades_df[trades_df['time_gap'] > pd.Timedelta(hours=1)]
    
    if len(large_gaps) > 0:
        print("[DETECT] Found potential checkpoint resume points (gaps > 1 hour):")
        for idx, row in large_gaps.iterrows():
            print(f"   Gap at {row['timestamp']}: {row['time_gap']}")
            print(f"      Episode {row['episode']}, Trade ID {row['trade_id']}")
        print()
        
        # Use the first large gap as checkpoint resume point
        checkpoint_resume_time = large_gaps.iloc[0]['timestamp_dt']
        print(f"[INFO] Using checkpoint resume time: {checkpoint_resume_time}")
    else:
        # If no large gaps, check episode numbers
        # Look for episode number reset or jump
        episode_changes = trades_df[trades_df['episode'].diff() < 0]
        if len(episode_changes) > 0:
            checkpoint_resume_time = episode_changes.iloc[0]['timestamp_dt']
            print(f"[INFO] Found episode reset at: {checkpoint_resume_time}")
        else:
            # Default: use middle point or ask user
            print("[WARN] Could not automatically detect checkpoint resume point")
            print("   Please provide the timestamp when checkpoint 1000000 was resumed")
            print(f"   Format: YYYY-MM-DDTHH:MM:SS (e.g., {trades_df['timestamp'].iloc[len(trades_df)//2]})")
            return
    
    # Split trades into before and after checkpoint
    before_checkpoint = trades_df[trades_df['timestamp_dt'] < checkpoint_resume_time]
    after_checkpoint = trades_df[trades_df['timestamp_dt'] >= checkpoint_resume_time]
    
    print()
    print("=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)
    print()
    
    # Before checkpoint
    if len(before_checkpoint) > 0:
        print("[BEFORE] BEFORE CHECKPOINT RESUME:")
        print(f"   Trades: {len(before_checkpoint)}")
        print(f"   Total PnL: ${before_checkpoint['net_pnl'].sum():.2f}")
        print(f"   Win Rate: {(before_checkpoint['is_win'].sum() / len(before_checkpoint) * 100):.2f}%")
        print(f"   Avg Trade: ${before_checkpoint['net_pnl'].mean():.2f}")
        print(f"   Profit Factor: {calculate_profit_factor(before_checkpoint):.2f}")
        print()
    
    # After checkpoint
    if len(after_checkpoint) > 0:
        print("[AFTER] AFTER CHECKPOINT RESUME:")
        print(f"   Trades: {len(after_checkpoint)}")
        print(f"   Total PnL: ${after_checkpoint['net_pnl'].sum():.2f}")
        print(f"   Win Rate: {(after_checkpoint['is_win'].sum() / len(after_checkpoint) * 100):.2f}%")
        print(f"   Avg Trade: ${after_checkpoint['net_pnl'].mean():.2f}")
        print(f"   Profit Factor: {calculate_profit_factor(after_checkpoint):.2f}")
        print()
        
        # Calculate additional metrics
        wins = after_checkpoint[after_checkpoint['is_win'] == 1]
        losses = after_checkpoint[after_checkpoint['is_win'] == 0]
        avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0.0
        avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0.0
        
        print("   Detailed Metrics:")
        print(f"   - Winning trades: {len(wins)}")
        print(f"   - Losing trades: {len(losses)}")
        print(f"   - Avg Win: ${avg_win:.2f}")
        print(f"   - Avg Loss: ${avg_loss:.2f}")
        print(f"   - Risk/Reward Ratio: {abs(avg_win / avg_loss) if avg_loss < 0 else 0.0:.2f}")
        
        # Calculate Sharpe-like ratio
        if len(after_checkpoint) > 1:
            returns = after_checkpoint['net_pnl'].values
            mean_return = returns.mean()
            std_return = returns.std()
            sharpe_like = (mean_return / std_return * (252**0.5)) if std_return > 0 else 0.0
            print(f"   - Sharpe-like Ratio: {sharpe_like:.2f}")
        
        # Max drawdown
        cumulative = after_checkpoint['net_pnl'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
        print(f"   - Max Drawdown: ${max_drawdown:.2f}")
        print()
    
    # Overall (current - mixing both)
    print("[OVERALL] OVERALL (MIXED - CURRENT MONITORING):")
    print(f"   Trades: {len(trades_df)}")
    print(f"   Total PnL: ${trades_df['net_pnl'].sum():.2f}")
    print(f"   Win Rate: {(trades_df['is_win'].sum() / len(trades_df) * 100):.2f}%")
    print(f"   Avg Trade: ${trades_df['net_pnl'].mean():.2f}")
    print(f"   Profit Factor: {calculate_profit_factor(trades_df):.2f}")
    print()
    
    print("=" * 80)
    print("[RECOMMENDATION]")
    print("=" * 80)
    print("The Monitoring tab is showing MIXED metrics (old + new trades).")
    print("To see only performance AFTER checkpoint resume, use:")
    print(f"   curl 'http://localhost:8200/api/monitoring/performance?since={checkpoint_resume_time.isoformat()}'")
    print()
    print("Or modify the frontend to filter by timestamp when resuming from checkpoint.")

def calculate_profit_factor(trades_df):
    """Calculate profit factor"""
    wins = trades_df[trades_df['is_win'] == 1]
    losses = trades_df[trades_df['is_win'] == 0]
    
    gross_profit = wins['net_pnl'].sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 0.0
    
    if gross_loss > 0:
        return gross_profit / gross_loss
    elif gross_profit > 0:
        return float('inf')
    else:
        return 0.0

if __name__ == "__main__":
    analyze_performance()
