"""
Analyze current training metrics and identify concerns
"""

def analyze_metrics():
    """Analyze the training metrics from the screenshot"""
    print("\n" + "=" * 60)
    print("TRAINING METRICS ANALYSIS")
    print("=" * 60)
    
    # Metrics from screenshot
    current_episode = 361
    total_trades = 9
    winning_trades = 2
    losing_trades = 7
    overall_win_rate = 22.2  # %
    current_episode_trades = 6
    current_win_rate = 40.0  # %
    mean_pnl_10 = 710.24
    mean_win_rate_10 = 25.0  # %
    latest_episode_length = 20  # steps
    mean_episode_length = 9980.0  # steps
    
    print("\n[1] TRADE COUNT ANALYSIS")
    print("-" * 60)
    trades_per_episode = total_trades / current_episode if current_episode > 0 else 0
    print(f"  Total Episodes: {current_episode}")
    print(f"  Total Trades: {total_trades}")
    print(f"  Average Trades per Episode: {trades_per_episode:.3f}")
    print(f"  Current Episode Trades: {current_episode_trades}")
    
    # Expected vs Actual
    expected_trades_per_episode = 0.5  # Conservative estimate (should be 1-2 per episode)
    expected_total_trades = current_episode * expected_trades_per_episode
    
    print(f"\n  Expected Trades (at 0.5/episode): {expected_total_trades:.0f}")
    print(f"  Actual Trades: {total_trades}")
    print(f"  Gap: {expected_total_trades - total_trades:.0f} trades missing")
    
    if trades_per_episode < 0.1:
        print(f"  [CRITICAL] Trade count is EXTREMELY low!")
        print(f"            Only {trades_per_episode:.3f} trades per episode")
        print(f"            System is being too conservative")
    
    print("\n[2] WIN RATE ANALYSIS")
    print("-" * 60)
    print(f"  Overall Win Rate: {overall_win_rate:.1f}%")
    print(f"  Current Episode Win Rate: {current_win_rate:.1f}%")
    print(f"  Mean Win Rate (Last 10): {mean_win_rate_10:.1f}%")
    
    target_win_rate = 60.0  # %
    print(f"\n  Target Win Rate: {target_win_rate:.1f}%")
    print(f"  Gap: {target_win_rate - overall_win_rate:.1f}% below target")
    
    if overall_win_rate < 30:
        print(f"  [CRITICAL] Win rate is VERY LOW ({overall_win_rate:.1f}%)")
        print(f"            With commissions, this is likely unprofitable")
        print(f"            Need to improve trade quality")
    
    # Calculate breakeven win rate
    # Assuming avg_win = $100, avg_loss = $50, commission = $3
    avg_win = 100.0  # Estimate
    avg_loss = 50.0  # Estimate
    commission = 3.0
    breakeven_wr = (avg_loss + commission) / (avg_win + avg_loss + 2*commission) * 100
    print(f"\n  Estimated Breakeven Win Rate: {breakeven_wr:.1f}%")
    if overall_win_rate < breakeven_wr:
        print(f"  [WARN] Current win rate ({overall_win_rate:.1f}%) is below breakeven ({breakeven_wr:.1f}%)")
    
    print("\n[3] EPISODE LENGTH ANALYSIS")
    print("-" * 60)
    print(f"  Latest Episode Length: {latest_episode_length} steps")
    print(f"  Mean Episode Length: {mean_episode_length:.1f} steps")
    
    if latest_episode_length < 100:
        print(f"  [WARN] Latest episode is very short ({latest_episode_length} steps)")
        print(f"         Episodes should be ~{mean_episode_length:.0f} steps")
        print(f"         Short episodes might indicate early termination or issues")
    
    print("\n[4] PROFITABILITY ANALYSIS")
    print("-" * 60)
    print(f"  Mean PnL (Last 10): ${mean_pnl_10:.2f}")
    
    if mean_pnl_10 > 0:
        print(f"  [OK] Mean PnL is positive")
    else:
        print(f"  [WARN] Mean PnL is negative")
    
    # Calculate expected profitability
    # With 9 trades, 2 wins, 7 losses, and 22.2% win rate
    # If avg_win = $100, avg_loss = $50, commission = $3
    estimated_total_pnl = (2 * 100) - (7 * 50) - (9 * 3)
    print(f"  Estimated Total PnL (9 trades): ${estimated_total_pnl:.2f}")
    
    print("\n[5] KEY CONCERNS")
    print("-" * 60)
    concerns = []
    
    if trades_per_episode < 0.1:
        concerns.append(("CRITICAL", f"Extremely low trade count: {trades_per_episode:.3f} trades/episode"))
    
    if overall_win_rate < 30:
        concerns.append(("CRITICAL", f"Very low win rate: {overall_win_rate:.1f}% (target: 60%+)"))
    
    if latest_episode_length < 100:
        concerns.append(("HIGH", f"Very short latest episode: {latest_episode_length} steps"))
    
    if total_trades < 10 and current_episode > 100:
        concerns.append(("HIGH", f"Only {total_trades} trades in {current_episode} episodes - system too conservative"))
    
    if not concerns:
        print("  [OK] No major concerns identified")
    else:
        for severity, concern in concerns:
            print(f"  [{severity}] {concern}")
    
    print("\n[6] RECOMMENDATIONS")
    print("-" * 60)
    
    if trades_per_episode < 0.1:
        print("  1. [URGENT] Trade count is too low - system is too conservative")
        print("     - Consider reducing action_threshold (currently 0.05)")
        print("     - Consider reducing min_combined_confidence (currently 0.5)")
        print("     - Consider reducing quality filter thresholds")
        print("     - Check if DecisionGate is rejecting too many trades")
    
    if overall_win_rate < 30:
        print("  2. [URGENT] Win rate is too low - need better trade quality")
        print("     - Current filters may not be effective")
        print("     - May need to improve quality scoring")
        print("     - Consider increasing min_quality_score to filter out bad trades")
    
    if latest_episode_length < 100:
        print("  3. [MEDIUM] Investigate why latest episode was so short")
        print("     - Check if episodes are terminating early")
        print("     - Verify max_episode_steps configuration")
    
    print("\n[7] POSITIVE SIGNS")
    print("-" * 60)
    positives = []
    
    if mean_pnl_10 > 0:
        positives.append(f"Mean PnL (Last 10) is positive: ${mean_pnl_10:.2f}")
    
    if current_win_rate > overall_win_rate:
        positives.append(f"Current episode win rate ({current_win_rate:.1f}%) is better than overall ({overall_win_rate:.1f}%)")
    
    if current_episode_trades > 0:
        positives.append(f"Current episode has trades: {current_episode_trades}")
    
    if positives:
        for positive in positives:
            print(f"  [OK] {positive}")
    else:
        print("  [WARN] No positive signs identified")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    analyze_metrics()

