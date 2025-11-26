"""Calculate the effective R:R ratio accounting for commission and slippage"""
import sqlite3
import pandas as pd

# Connect to trading journal
db_path = "logs/trading_journal.db"
conn = sqlite3.connect(db_path)

# Get all trades with commission data
query = """
    SELECT 
        pnl, net_pnl, commission, is_win
    FROM trades
    ORDER BY timestamp DESC
"""
df = pd.read_sql_query(query, conn)
conn.close()

print("="*80)
print("R:R RATIO ANALYSIS - ACCOUNTING FOR COMMISSION")
print("="*80)
print()

if len(df) == 0:
    print("No trades found")
    exit(1)

# Separate wins and losses
winning_trades = df[df['is_win'] == 1]
losing_trades = df[df['is_win'] == 0]

# Current R:R (using net PnL - already includes commission)
avg_win_net = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
avg_loss_net = abs(losing_trades['net_pnl'].mean()) if len(losing_trades) > 0 else 0
current_rr = avg_win_net / avg_loss_net if avg_loss_net > 0 else 0

print(f"CURRENT R:R CALCULATION (Net PnL - Commission Already Deducted):")
print(f"   Average Win (Net): ${avg_win_net:,.2f}")
print(f"   Average Loss (Net): ${avg_loss_net:,.2f}")
print(f"   Current R:R: {current_rr:.2f}:1")
print()

# Commission analysis
avg_commission = df['commission'].mean() if 'commission' in df.columns else 0
print(f"COMMISSION IMPACT:")
print(f"   Average Commission per Trade: ${avg_commission:,.2f}")
print(f"   Commission as % of Avg Win: {(avg_commission/avg_win_net*100) if avg_win_net > 0 else 0:.2f}%")
print()

# Calculate what R:R would be WITHOUT commission (gross)
avg_win_gross = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
avg_loss_gross = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
gross_rr = avg_win_gross / avg_loss_gross if avg_loss_gross > 0 else 0

print(f"GROSS R:R (Before Commission Deduction):")
print(f"   Average Win (Gross): ${avg_win_gross:,.2f}")
print(f"   Average Loss (Gross): ${avg_loss_gross:,.2f}")
print(f"   Gross R:R: {gross_rr:.2f}:1")
print()

# Calculate break-even R:R needed
win_rate = len(winning_trades) / len(df) * 100
break_even_rr_gross = (100 - win_rate) / win_rate
break_even_rr_net = break_even_rr_gross  # Same calculation, but with commission accounted for

print(f"BREAK-EVEN ANALYSIS:")
print(f"   Win Rate: {win_rate:.2f}%")
print(f"   Break-even R:R (gross): {break_even_rr_gross:.2f}:1")
print(f"   Break-even R:R (net): {break_even_rr_net:.2f}:1")
print()

# Account for commission in break-even calculation
# Break-even: win_rate * (avg_win_gross - commission) - (1-win_rate) * (avg_loss_gross + commission) = 0
# Solving for required gross R:R:
# win_rate * (R:R * avg_loss_gross - commission) - (1-win_rate) * (avg_loss_gross + commission) = 0
# win_rate * R:R * avg_loss_gross - win_rate * commission - avg_loss_gross + win_rate * avg_loss_gross - commission + win_rate * commission = 0
# win_rate * R:R * avg_loss_gross = avg_loss_gross + commission
# R:R = (avg_loss_gross + commission) / (win_rate * avg_loss_gross)
# R:R = 1/win_rate + commission/(win_rate * avg_loss_gross)

if avg_loss_gross > 0:
    # Account for commission (paid on both entry and exit, so 2x commission per round trip)
    # But commission is already in net PnL, so we need to account for it in break-even calculation
    commission_impact_rr = avg_commission / (win_rate/100 * avg_loss_gross) if win_rate > 0 else 0
    break_even_rr_with_commission = break_even_rr_net + commission_impact_rr
    
    print(f"BREAK-EVEN R:R ACCOUNTING FOR COMMISSION:")
    print(f"   Base break-even R:R: {break_even_rr_net:.2f}:1")
    print(f"   Commission impact: +{commission_impact_rr:.2f}")
    print(f"   Required R:R (with commission): {break_even_rr_with_commission:.2f}:1")
    print()

# Calculate minimum profitable R:R (target 1.5:1 net, but need to account for commission)
target_rr_net = 1.5  # Target net R:R
if avg_loss_gross > 0:
    # To achieve 1.5:1 net R:R, gross R:R needs to be higher to cover commission
    required_gross_win = target_rr_net * avg_loss_net + avg_commission * 2  # Commission on entry and exit
    required_gross_rr = required_gross_win / avg_loss_gross if avg_loss_gross > 0 else 0
    
    print(f"TARGET R:R ANALYSIS:")
    print(f"   Target Net R:R: {target_rr_net:.2f}:1")
    print(f"   Required Gross Win: ${required_gross_win:,.2f} (to achieve {target_rr_net:.2f}:1 net after commission)")
    print(f"   Required Gross R:R: {required_gross_rr:.2f}:1")
    print()

print("="*80)
print("RECOMMENDATIONS")
print("="*80)
print()

if current_rr < break_even_rr_net:
    print(f"[CRITICAL] Current R:R ({current_rr:.2f}:1) is BELOW break-even ({break_even_rr_net:.2f}:1)")
    print(f"   - System is losing money even with {win_rate:.1f}% win rate")
    print(f"   - Need minimum R:R of {break_even_rr_net:.2f}:1 just to break even")
    print()

if current_rr < 1.5:
    print(f"[WARNING] Current R:R ({current_rr:.2f}:1) is below target (1.5:1)")
    print(f"   - With commission costs, should aim for 1.5:1+ net R:R")
    print(f"   - Current setting min_risk_reward_ratio: 1.5 is correct, but actual R:R is only {current_rr:.2f}:1")
    print(f"   - Enforcement is too lenient or reward function isn't encouraging better R:R")
    print()

print(f"[INFO] Commission reduces effective R:R:")
print(f"   - Commission of ${avg_commission:,.2f} per trade reduces net win by {avg_commission/avg_win_gross*100 if avg_win_gross > 0 else 0:.1f}%")
print(f"   - To maintain 1.5:1 net R:R, need {required_gross_rr:.2f}:1 gross R:R")
print()

