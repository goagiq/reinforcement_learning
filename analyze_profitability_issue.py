"""
Analyze why system is losing money despite higher win rate
"""

def analyze_profitability_issue():
    """Analyze the profitability problem"""
    print("\n" + "=" * 60)
    print("PROFITABILITY ISSUE ANALYSIS")
    print("=" * 60)
    
    # Current metrics
    total_trades = 77
    winning_trades = 29
    losing_trades = 48
    win_rate = 37.7
    mean_pnl_10 = 228.43
    current_pnl = -307.23
    
    print("\n[1] CURRENT METRICS")
    print("-" * 60)
    print(f"  Total Trades: {total_trades}")
    print(f"  Winning Trades: {winning_trades}")
    print(f"  Losing Trades: {losing_trades}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Mean PnL (Last 10): ${mean_pnl_10:.2f}")
    print(f"  Current Episode PnL: ${current_pnl:.2f}")
    
    print("\n[2] PROFITABILITY CALCULATION")
    print("-" * 60)
    
    # Estimate average win and loss
    # If we have 29 wins and 48 losses, and we're losing money with 37.7% win rate
    # We can estimate: (win_rate * avg_win) - ((1 - win_rate) * avg_loss) - commissions < 0
    
    # Let's calculate what avg_win/avg_loss ratio we need
    # For profitability: (0.377 * avg_win) - (0.623 * avg_loss) - commissions > 0
    # If we're losing, then: (0.377 * avg_win) - (0.623 * avg_loss) - commissions < 0
    
    # Estimate commission cost
    commission_per_trade = 3.0  # $3 per trade (0.03% of $100k)
    total_commission = total_trades * commission_per_trade
    print(f"  Estimated Total Commission: ${total_commission:.2f} ({total_trades} trades * ${commission_per_trade})")
    
    # If mean PnL is $228.43 for last 10 episodes, but we're losing overall
    # This suggests the issue is with risk/reward ratio
    
    print("\n[3] ROOT CAUSE ANALYSIS")
    print("-" * 60)
    
    print("\n  Problem: High Win Rate but Losing Money")
    print("  This typically means:")
    print("    1. Risk/Reward ratio is poor (losses are larger than wins)")
    print("    2. Commission costs are eating into profits")
    print("    3. Average loss > Average win (even with higher win rate)")
    
    # Calculate required risk/reward ratio
    # For profitability with 37.7% win rate:
    # (0.377 * avg_win) - (0.623 * avg_loss) - commission > 0
    # 0.377 * avg_win > 0.623 * avg_loss + commission
    # avg_win > (0.623 * avg_loss + commission) / 0.377
    # avg_win / avg_loss > (0.623 + commission/avg_loss) / 0.377
    
    # If avg_loss = $50, commission = $3
    # avg_win / avg_loss > (0.623 + 3/50) / 0.377
    # avg_win / avg_loss > (0.623 + 0.06) / 0.377
    # avg_win / avg_loss > 0.683 / 0.377
    # avg_win / avg_loss > 1.81
    
    print("\n  Required Risk/Reward Ratio:")
    print("    For 37.7% win rate to be profitable:")
    print("    avg_win / avg_loss must be > 1.8:1 (approximately)")
    print("    Meaning: Average win must be at least 1.8x the average loss")
    
    print("\n[4] LIKELY ISSUES")
    print("-" * 60)
    
    issues = []
    
    issues.append(("CRITICAL", "Poor Risk/Reward Ratio"))
    print("  1. [CRITICAL] Poor Risk/Reward Ratio")
    print("     - Average loss is likely larger than average win")
    print("     - Even with 37.7% win rate, if avg_loss > avg_win, you lose money")
    print("     - Need to ensure avg_win >= 1.8x avg_loss")
    
    issues.append(("HIGH", "High Commission Costs"))
    print("\n  2. [HIGH] High Commission Costs")
    print(f"     - {total_trades} trades = ${total_commission:.2f} in commissions")
    print(f"     - With 22.5 trades/episode, commissions add up quickly")
    print("     - Need to reduce trade count OR increase trade quality")
    
    issues.append(("HIGH", "No Stop Loss / Risk Management"))
    print("\n  3. [HIGH] Missing Stop Loss / Risk Management")
    print("     - If losses are not capped, they can exceed wins")
    print("     - Need to implement proper stop loss")
    print("     - Need to limit position size based on risk")
    
    issues.append(("MEDIUM", "Position Sizing Issues"))
    print("\n  4. [MEDIUM] Position Sizing Issues")
    print("     - Position sizes might be too large for losing trades")
    print("     - Position sizes might be too small for winning trades")
    print("     - Need dynamic position sizing based on confidence")
    
    print("\n[5] SOLUTIONS")
    print("-" * 60)
    
    print("\n  1. [URGENT] Implement Risk/Reward Ratio Check")
    print("     - Add minimum risk/reward ratio requirement (e.g., 1.5:1 or 2:1)")
    print("     - Reject trades that don't meet minimum R:R ratio")
    print("     - This ensures avg_win >= 1.5x or 2x avg_loss")
    
    print("\n  2. [URGENT] Implement Stop Loss")
    print("     - Add stop loss to limit loss size")
    print("     - Ensure stop loss is tighter than take profit")
    print("     - This caps maximum loss per trade")
    
    print("\n  3. [HIGH] Reduce Trade Count")
    print("     - Current: 22.5 trades/episode (very high)")
    print("     - Target: 0.5-1.0 trades/episode")
    print("     - Tighten filters to reduce trade count")
    print("     - Focus on quality over quantity")
    
    print("\n  4. [HIGH] Improve Position Sizing")
    print("     - Use smaller positions for lower confidence trades")
    print("     - Use larger positions for higher confidence trades")
    print("     - Implement Kelly Criterion or similar")
    
    print("\n  5. [MEDIUM] Track Average Win/Loss")
    print("     - Monitor avg_win and avg_loss")
    print("     - Ensure avg_win / avg_loss >= 1.5:1")
    print("     - Adjust filters if ratio is too low")
    
    print("\n[6] IMMEDIATE ACTIONS")
    print("-" * 60)
    
    print("\n  1. Add Risk/Reward Ratio Filter")
    print("     - In DecisionGate or TradingEnvironment")
    print("     - Reject trades with R:R < 1.5:1")
    print("     - This will improve avg_win / avg_loss ratio")
    
    print("\n  2. Implement Stop Loss")
    print("     - Add stop loss to RiskManager")
    print("     - Set stop loss at 1-2% of entry price")
    print("     - This caps maximum loss per trade")
    
    print("\n  3. Tighten Quality Filters")
    print("     - Increase min_quality_score")
    print("     - Increase min_combined_confidence")
    print("     - This will reduce trade count and improve quality")
    
    print("\n  4. Monitor Average Win/Loss")
    print("     - Track avg_win and avg_loss in metrics")
    print("     - Display in dashboard")
    print("     - Alert if ratio drops below 1.5:1")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    print("\n[SUMMARY]")
    print("=" * 60)
    print("  Problem: High win rate (37.7%) but losing money")
    print("  Root Cause: Poor risk/reward ratio (avg_loss > avg_win)")
    print("  Solution: Implement risk/reward ratio check and stop loss")
    print("  Action: Add R:R filter and stop loss immediately")

if __name__ == "__main__":
    analyze_profitability_issue()

