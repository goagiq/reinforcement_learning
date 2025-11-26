"""Analyze why 45% win rate isn't profitable"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

# Connect to trading journal
db_path = "logs/trading_journal.db"
conn = sqlite3.connect(db_path)

# Get all trades
query = """
    SELECT 
        timestamp, entry_price, exit_price, position_size, 
        pnl, net_pnl, commission, is_win
    FROM trades
    ORDER BY timestamp DESC
"""
df = pd.read_sql_query(query, conn)
conn.close()

print("="*80)
print("PROFITABILITY ANALYSIS")
print("="*80)
print()

if len(df) == 0:
    print("No trades found in database")
    exit(1)

# Overall stats
total_trades = len(df)
winning_trades = df[df['is_win'] == 1]
losing_trades = df[df['is_win'] == 0]

win_count = len(winning_trades)
loss_count = len(losing_trades)
win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0

print(f"OVERALL STATISTICS")
print(f"   Total Trades: {total_trades:,}")
print(f"   Winning Trades: {win_count:,}")
print(f"   Losing Trades: {loss_count:,}")
print(f"   Win Rate: {win_rate:.2f}%")
print()

# P&L Analysis
total_pnl = df['net_pnl'].sum()
total_gross_pnl = df['pnl'].sum()
total_commission = df['commission'].sum() if 'commission' in df.columns else 0

gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0

avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
avg_loss_abs = abs(avg_loss)

profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
rr_ratio = avg_win / avg_loss_abs if avg_loss_abs > 0 else 0.0

print(f"P&L ANALYSIS")
print(f"   Total P&L (Net): ${total_pnl:,.2f}")
print(f"   Total P&L (Gross): ${total_gross_pnl:,.2f}")
print(f"   Total Commission: ${total_commission:,.2f}")
print()

print(f"WIN/LOSS BREAKDOWN")
print(f"   Average Win: ${avg_win:,.2f}")
print(f"   Average Loss: ${avg_loss:,.2f} (${avg_loss_abs:,.2f} absolute)")
print(f"   Risk/Reward Ratio: {rr_ratio:.2f}:1")
print(f"   Profit Factor: {profit_factor:.2f}")
print()

# Calculate expected value
if total_trades > 0:
    expected_value_per_trade = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
    expected_value_total = expected_value_per_trade * total_trades
    
    print(f"EXPECTED VALUE ANALYSIS")
    print(f"   Expected Value per Trade: ${expected_value_per_trade:,.2f}")
    print(f"   Expected Value (Total): ${expected_value_total:,.2f}")
    print()

# Commission impact
avg_commission = total_commission / total_trades if total_trades > 0 else 0
commission_pct_of_avg_trade = (avg_commission / abs(avg_win)) * 100 if avg_win > 0 else 0

print(f"COMMISSION ANALYSIS")
print(f"   Total Commission: ${total_commission:,.2f}")
print(f"   Average Commission per Trade: ${avg_commission:,.4f}")
print(f"   Commission as % of Avg Win: {commission_pct_of_avg_trade:.2f}%")
print()

# Problem diagnosis
print("="*80)
print("PROBLEM DIAGNOSIS")
print("="*80)

issues = []

if profit_factor < 1.0:
    issues.append(f"[ERROR] Profit Factor {profit_factor:.2f} < 1.0 (gross losses > gross profits)")

if rr_ratio < 1.2:
    issues.append(f"[ERROR] Risk/Reward Ratio {rr_ratio:.2f}:1 < 1.2:1 (average loss too close to average win)")

if avg_loss_abs > avg_win:
    issues.append(f"[ERROR] Average Loss (${avg_loss_abs:,.2f}) > Average Win (${avg_win:,.2f})")

if commission_pct_of_avg_trade > 5:
    issues.append(f"[WARN]  Commission ({commission_pct_of_avg_trade:.2f}% of avg win) may be too high")

# Calculate break-even R:R needed
if win_rate > 0:
    break_even_rr = (100 - win_rate) / win_rate
    if rr_ratio < break_even_rr:
        issues.append(f"[ERROR] Current R:R {rr_ratio:.2f}:1 < Break-even R:R {break_even_rr:.2f}:1 needed for {win_rate:.1f}% win rate")

if len(issues) == 0:
    print("[OK] No obvious issues found - all metrics look good!")
else:
    for issue in issues:
        print(f"   {issue}")

print()

# Recommendations
print("="*80)
print("RECOMMENDATIONS")
print("="*80)

if rr_ratio < 1.5:
    print(f"1. [WARN]  R:R RATIO TOO LOW: Current {rr_ratio:.2f}:1")
    print(f"   - Need R:R of at least 1.5:1 (ideally 2.0:1+) to be profitable with {win_rate:.1f}% win rate")
    print(f"   - Average win should be ${avg_loss_abs * 1.5:,.2f}+ (currently ${avg_win:,.2f})")
    print(f"   - Consider tightening stop loss or letting winners run longer")
    print()

if profit_factor < 1.0:
    print(f"2. [WARN] PROFIT FACTOR TOO LOW: {profit_factor:.2f}")
    print(f"   - Gross losses (${gross_loss:,.2f}) exceed gross profits (${gross_profit:,.2f})")
    print(f"   - This means losses are too large relative to wins")
    print(f"   - Need better R:R ratio (see recommendation #1)")
    print()

if commission_pct_of_avg_trade > 3:
    print(f"3. [WARN] COMMISSION COSTS: {commission_pct_of_avg_trade:.2f}% of avg win")
    print(f"   - Commission may be eating into profits")
    print(f"   - Consider reducing transaction costs in config")
    print()

# Check recent trades for trends
print()
print("="*80)
print("RECENT TRADES (Last 50)")
print("="*80)
recent = df.head(50)
recent_wins = recent[recent['is_win'] == 1]
recent_losses = recent[recent['is_win'] == 0]

if len(recent) > 0:
    recent_win_rate = (len(recent_wins) / len(recent) * 100) if len(recent) > 0 else 0.0
    recent_avg_win = recent_wins['net_pnl'].mean() if len(recent_wins) > 0 else 0
    recent_avg_loss = abs(recent_losses['net_pnl'].mean()) if len(recent_losses) > 0 else 0
    recent_rr = recent_avg_win / recent_avg_loss if recent_avg_loss > 0 else 0
    
    print(f"   Recent Win Rate: {recent_win_rate:.2f}%")
    print(f"   Recent Avg Win: ${recent_avg_win:,.2f}")
    print(f"   Recent Avg Loss: ${recent_avg_loss:,.2f}")
    print(f"   Recent R:R: {recent_rr:.2f}:1")
    
    if recent_rr < rr_ratio:
        print(f"   [WARN]  Recent R:R ({recent_rr:.2f}:1) is WORSE than overall ({rr_ratio:.2f}:1)")
    elif recent_rr > rr_ratio:
        print(f"   [OK] Recent R:R ({recent_rr:.2f}:1) is BETTER than overall ({rr_ratio:.2f}:1) - improving!")
