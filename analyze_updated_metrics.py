"""
Analyze updated training metrics after fixes
"""

def analyze_updated_metrics():
    """Compare metrics before and after fixes"""
    print("\n" + "=" * 60)
    print("UPDATED METRICS ANALYSIS (After Fixes)")
    print("=" * 60)
    
    # Previous metrics (Episode 361)
    prev_episode = 361
    prev_total_trades = 9
    prev_win_rate = 22.2
    prev_mean_win_rate_10 = 25.0
    prev_latest_episode_length = 20
    
    # Current metrics (Episode 362)
    current_episode = 362
    current_total_trades = 15
    current_win_rate = 26.7
    current_mean_win_rate_10 = 32.5
    current_latest_episode_length = 40
    current_mean_pnl_10 = 716.17
    current_episode_trades = 4
    
    print("\n[1] TRADE COUNT PROGRESS")
    print("-" * 60)
    trades_added = current_total_trades - prev_total_trades
    episodes_passed = current_episode - prev_episode
    trades_per_episode_new = trades_added / episodes_passed if episodes_passed > 0 else 0
    
    print(f"  Previous (Episode {prev_episode}): {prev_total_trades} trades")
    print(f"  Current (Episode {current_episode}): {current_total_trades} trades")
    print(f"  Trades Added: +{trades_added} trades in {episodes_passed} episode(s)")
    print(f"  Trades per Episode (New): {trades_per_episode_new:.2f}")
    
    if trades_added > 0:
        improvement_pct = (trades_added / prev_total_trades) * 100 if prev_total_trades > 0 else 0
        print(f"  Improvement: +{improvement_pct:.1f}% increase in trade count")
        print(f"  [OK] Trade count is INCREASING - fixes are working!")
    else:
        print(f"  [WARN] No new trades added")
    
    # Projection
    if trades_per_episode_new > 0:
        projected_trades_100_episodes = trades_per_episode_new * 100
        print(f"\n  Projection (if trend continues):")
        print(f"    Next 100 episodes: ~{projected_trades_100_episodes:.0f} trades")
        print(f"    Total after 100 episodes: ~{current_total_trades + projected_trades_100_episodes:.0f} trades")
    
    print("\n[2] WIN RATE PROGRESS")
    print("-" * 60)
    win_rate_improvement = current_win_rate - prev_win_rate
    mean_win_rate_improvement = current_mean_win_rate_10 - prev_mean_win_rate_10
    
    print(f"  Overall Win Rate:")
    print(f"    Previous: {prev_win_rate:.1f}%")
    print(f"    Current: {current_win_rate:.1f}%")
    print(f"    Change: {win_rate_improvement:+.1f}%")
    
    print(f"\n  Mean Win Rate (Last 10):")
    print(f"    Previous: {prev_mean_win_rate_10:.1f}%")
    print(f"    Current: {current_mean_win_rate_10:.1f}%")
    print(f"    Change: {mean_win_rate_improvement:+.1f}%")
    
    target_win_rate = 60.0
    gap_to_target = target_win_rate - current_win_rate
    print(f"\n  Target: {target_win_rate:.1f}%")
    print(f"  Gap to Target: {gap_to_target:.1f}%")
    
    if win_rate_improvement > 0:
        print(f"  [OK] Win rate is IMPROVING!")
    else:
        print(f"  [WARN] Win rate decreased")
    
    if current_mean_win_rate_10 > prev_mean_win_rate_10:
        print(f"  [OK] Mean win rate (last 10) is IMPROVING!")
    
    print("\n[3] EPISODE LENGTH PROGRESS")
    print("-" * 60)
    print(f"  Previous Latest: {prev_latest_episode_length} steps")
    print(f"  Current Latest: {current_latest_episode_length} steps")
    print(f"  Mean Length: 9980.0 steps")
    
    if current_latest_episode_length > prev_latest_episode_length:
        print(f"  [OK] Latest episode length increased (from {prev_latest_episode_length} to {current_latest_episode_length})")
    else:
        print(f"  [WARN] Latest episode still short ({current_latest_episode_length} steps)")
    
    if current_latest_episode_length < 100:
        print(f"  [WARN] Latest episode is still very short (should be ~10,000 steps)")
    
    print("\n[4] PROFITABILITY ANALYSIS")
    print("-" * 60)
    print(f"  Mean PnL (Last 10): ${current_mean_pnl_10:.2f}")
    
    if current_mean_pnl_10 > 0:
        print(f"  [OK] Mean PnL is positive - recent episodes are profitable")
    else:
        print(f"  [WARN] Mean PnL is negative")
    
    # Calculate estimated profitability
    # With 15 trades, 4 wins, 11 losses, 26.7% win rate
    # Assuming avg_win = $100, avg_loss = $50, commission = $3
    estimated_total_pnl = (4 * 100) - (11 * 50) - (15 * 3)
    print(f"  Estimated Total PnL (15 trades): ${estimated_total_pnl:.2f}")
    
    print("\n[5] OVERALL ASSESSMENT")
    print("-" * 60)
    
    positives = []
    concerns = []
    
    if trades_added > 0:
        positives.append(f"Trade count increased: {prev_total_trades} -> {current_total_trades} (+{trades_added})")
    
    if win_rate_improvement > 0:
        positives.append(f"Overall win rate improved: {prev_win_rate:.1f}% -> {current_win_rate:.1f}%")
    
    if mean_win_rate_improvement > 0:
        positives.append(f"Mean win rate (last 10) improved: {prev_mean_win_rate_10:.1f}% -> {current_mean_win_rate_10:.1f}%")
    
    if current_mean_pnl_10 > 0:
        positives.append(f"Mean PnL (last 10) is positive: ${current_mean_pnl_10:.2f}")
    
    if current_win_rate < 30:
        concerns.append(f"Overall win rate still low: {current_win_rate:.1f}% (target: 60%+)")
    
    if current_mean_win_rate_10 < 40:
        concerns.append(f"Mean win rate (last 10) still below target: {current_mean_win_rate_10:.1f}% (target: 60%+)")
    
    if current_latest_episode_length < 100:
        concerns.append(f"Latest episode still very short: {current_latest_episode_length} steps (expected: ~10,000)")
    
    trades_per_episode_overall = current_total_trades / current_episode
    if trades_per_episode_overall < 0.1:
        concerns.append(f"Overall trade count still low: {trades_per_episode_overall:.3f} trades/episode (target: 0.5-1.0)")
    
    print("\n  POSITIVE SIGNS:")
    if positives:
        for positive in positives:
            print(f"    [OK] {positive}")
    else:
        print("    [WARN] No positive signs identified")
    
    print("\n  CONCERNS:")
    if concerns:
        for concern in concerns:
            print(f"    [WARN] {concern}")
    else:
        print("    [OK] No major concerns")
    
    print("\n[6] RECOMMENDATIONS")
    print("-" * 60)
    
    if trades_per_episode_new < 0.3:
        print("  1. [CONTINUE] Trade count is increasing but still low")
        print("     - Monitor for next 50 episodes")
        print("     - If still low, may need to reduce thresholds further")
    
    if current_win_rate < 40:
        print("  2. [MONITOR] Win rate is improving but still below target")
        print("     - Continue monitoring trend")
        print("     - May need to improve quality scoring if win rate doesn't improve")
    
    if current_latest_episode_length < 100:
        print("  3. [INVESTIGATE] Latest episode still very short")
        print("     - Check why episodes are terminating early")
        print("     - Verify episode termination logic")
    
    print("\n  4. [CONTINUE] Keep monitoring metrics")
    print("     - Track trade count trend over next 50 episodes")
    print("     - Track win rate trend")
    print("     - Ensure mean PnL remains positive")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    analyze_updated_metrics()

