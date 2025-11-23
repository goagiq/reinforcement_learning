"""
Analyze current training status (Episode 365)
"""

def analyze_current_status():
    """Analyze current metrics and trends"""
    print("\n" + "=" * 60)
    print("CURRENT TRAINING STATUS ANALYSIS (Episode 365)")
    print("=" * 60)
    
    # Episode 363 metrics
    ep363_trades = 19
    ep363_win_rate = 21.1
    ep363_mean_win_rate_10 = 21.7
    ep363_mean_pnl_10 = 191.21
    ep363_episode_length = 60
    
    # Episode 365 metrics (current)
    ep365_trades = 32
    ep365_win_rate = 21.9
    ep365_mean_win_rate_10 = 24.3
    ep365_mean_pnl_10 = 344.13
    ep365_episode_length = 100
    ep365_current_trades = 27
    ep365_current_win_rate = 53.8
    ep365_mean_episode_length = 9980.0
    
    print("\n[1] TRADE COUNT PROGRESS")
    print("-" * 60)
    trades_added = ep365_trades - ep363_trades
    episodes_passed = 365 - 363
    
    print(f"  Episode 363: {ep363_trades} trades")
    print(f"  Episode 365: {ep365_trades} trades (+{trades_added} in {episodes_passed} episodes)")
    print(f"  Current Episode: {ep365_current_trades} trades")
    
    trades_per_episode = trades_added / episodes_passed if episodes_passed > 0 else 0
    print(f"  Trades per Episode (Recent): {trades_per_episode:.1f}")
    
    if ep365_current_trades > 20:
        print(f"  [WARN] Current episode has {ep365_current_trades} trades (very high!)")
        print(f"         This might indicate filters are too permissive")
    else:
        print(f"  [OK] Current episode trade count is reasonable")
    
    print("\n[2] WIN RATE PROGRESS")
    print("-" * 60)
    print(f"  Overall Win Rate:")
    print(f"    Episode 363: {ep363_win_rate:.1f}%")
    print(f"    Episode 365: {ep365_win_rate:.1f}% ({ep365_win_rate - ep363_win_rate:+.1f}%)")
    
    print(f"\n  Mean Win Rate (Last 10):")
    print(f"    Episode 363: {ep363_mean_win_rate_10:.1f}%")
    print(f"    Episode 365: {ep365_mean_win_rate_10:.1f}% ({ep365_mean_win_rate_10 - ep363_mean_win_rate_10:+.1f}%)")
    
    print(f"\n  Current Episode Win Rate: {ep365_current_win_rate:.1f}%")
    print(f"  [OK] Current episode win rate ({ep365_current_win_rate:.1f}%) is much better than overall ({ep365_win_rate:.1f}%)")
    
    if ep365_mean_win_rate_10 > ep363_mean_win_rate_10:
        print(f"  [OK] Mean win rate (last 10) is improving!")
    
    print("\n[3] EPISODE LENGTH PROGRESS")
    print("-" * 60)
    print(f"  Latest Episode Length:")
    print(f"    Episode 363: {ep363_episode_length} steps")
    print(f"    Episode 365: {ep365_episode_length} steps (+{ep365_episode_length - ep363_episode_length})")
    
    print(f"\n  Mean Episode Length: {ep365_mean_episode_length:.1f} steps")
    
    if ep365_episode_length > ep363_episode_length:
        print(f"  [OK] Latest episode length is increasing (60 -> 100 steps)")
        print(f"       Boundary check fix appears to be helping!")
    
    if ep365_mean_episode_length >= 9980:
        print(f"  [OK] Mean episode length is {ep365_mean_episode_length:.1f} steps")
        print(f"       This suggests many episodes ARE completing fully!")
        print(f"       The 100-step latest episode might be an outlier")
    else:
        print(f"  [WARN] Mean episode length is below expected 10,000 steps")
    
    if ep365_episode_length < 1000:
        print(f"  [WARN] Latest episode is still short ({ep365_episode_length} steps)")
        print(f"         Expected: ~10,000 steps")
    
    print("\n[4] PROFITABILITY PROGRESS")
    print("-" * 60)
    print(f"  Mean PnL (Last 10):")
    print(f"    Episode 363: ${ep363_mean_pnl_10:.2f}")
    print(f"    Episode 365: ${ep365_mean_pnl_10:.2f} ({ep365_mean_pnl_10 - ep363_mean_pnl_10:+.2f})")
    
    if ep365_mean_pnl_10 > 0:
        print(f"  [OK] Mean PnL is positive: ${ep365_mean_pnl_10:.2f}")
        if ep365_mean_pnl_10 > ep363_mean_pnl_10:
            print(f"  [OK] Mean PnL is improving!")
    
    print("\n[5] KEY OBSERVATIONS")
    print("-" * 60)
    
    observations = []
    
    # Trade count
    if ep365_trades > ep363_trades:
        observations.append(("POSITIVE", f"Trade count increased: {ep363_trades} -> {ep365_trades} (+{trades_added})"))
    
    if ep365_current_trades > 20:
        observations.append(("WARN", f"Current episode has {ep365_current_trades} trades (very high - may need to tighten filters)"))
    
    # Win rate
    if ep365_mean_win_rate_10 > ep363_mean_win_rate_10:
        observations.append(("POSITIVE", f"Mean win rate (last 10) improved: {ep363_mean_win_rate_10:.1f}% -> {ep365_mean_win_rate_10:.1f}%"))
    
    if ep365_current_win_rate > 50:
        observations.append(("POSITIVE", f"Current episode win rate is good: {ep365_current_win_rate:.1f}%"))
    
    # Episode length
    if ep365_episode_length > ep363_episode_length:
        observations.append(("POSITIVE", f"Latest episode length increased: {ep363_episode_length} -> {ep365_episode_length} steps"))
    
    if ep365_mean_episode_length >= 9980:
        observations.append(("POSITIVE", f"Mean episode length is {ep365_mean_episode_length:.1f} steps (many episodes completing fully)"))
    
    # Profitability
    if ep365_mean_pnl_10 > 0:
        observations.append(("POSITIVE", f"Mean PnL (last 10) is positive: ${ep365_mean_pnl_10:.2f}"))
    
    for category, observation in observations:
        if category == "POSITIVE":
            print(f"  [OK] {observation}")
        elif category == "WARN":
            print(f"  [WARN] {observation}")
        else:
            print(f"  [INFO] {observation}")
    
    print("\n[6] BOUNDARY CHECK FIX IMPACT")
    print("-" * 60)
    
    print("  Episode Length Improvement:")
    print(f"    Before fix: 60 steps (Episode 363)")
    print(f"    After fix: 100 steps (Episode 365)")
    print(f"    Improvement: +{ep365_episode_length - ep363_episode_length} steps")
    
    print(f"\n  Mean Episode Length: {ep365_mean_episode_length:.1f} steps")
    print(f"  [OK] Mean length suggests many episodes ARE completing fully!")
    print(f"       The 100-step latest episode might be an outlier or early termination")
    
    print("\n  [INFO] Boundary check fix appears to be helping:")
    print("         - Latest episode length increased")
    print("         - Mean episode length is near 10,000 steps")
    print("         - Some episodes are completing fully")
    
    print("\n[7] RECOMMENDATIONS")
    print("-" * 60)
    
    print("\n  1. [MONITOR] Episode Length")
    print("     - Latest episode is 100 steps (still short)")
    print("     - But mean is 9980.0 steps (suggests many complete)")
    print("     - Monitor if latest episode length continues to increase")
    print("     - Check if 100-step episodes are outliers or pattern")
    
    print("\n  2. [CONSIDER] Current Episode Trade Count")
    print(f"     - Current episode has {ep365_current_trades} trades (very high)")
    print("     - This might indicate filters are too permissive")
    print("     - However, current win rate is 53.8% (good!)")
    print("     - Monitor if this high trade count continues")
    print("     - If win rate stays high, might be OK")
    
    print("\n  3. [CONTINUE] Current Approach")
    print("     - Trade count is increasing (good)")
    print("     - Win rate is improving (mean last 10: 24.3%)")
    print("     - Mean PnL is positive ($344.13)")
    print("     - Episode length is improving")
    print("     - Continue monitoring for 50-100 more episodes")
    
    print("\n  4. [WATCH] Win Rate Trend")
    print("     - Current episode: 53.8% (excellent!)")
    print("     - Mean (last 10): 24.3% (improving)")
    print("     - Overall: 21.9% (still low, but improving)")
    print("     - If current episode trend continues, overall will improve")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    analyze_current_status()

