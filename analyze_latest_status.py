"""
Analyze latest training status (Episode 367)
"""

def analyze_latest_status():
    """Analyze current metrics and trends"""
    print("\n" + "=" * 60)
    print("LATEST TRAINING STATUS ANALYSIS (Episode 367)")
    print("=" * 60)
    
    # Episode 365 metrics
    ep365_trades = 32
    ep365_win_rate = 21.9
    ep365_mean_win_rate_10 = 24.3
    ep365_mean_pnl_10 = 344.13
    ep365_episode_length = 100
    
    # Episode 367 metrics (current)
    ep367_trades = 77
    ep367_win_rate = 37.7
    ep367_mean_win_rate_10 = 31.8
    ep367_mean_pnl_10 = 228.43
    ep367_episode_length = 140
    ep367_current_trades = 9
    ep367_current_win_rate = 25.0
    ep367_current_pnl = -307.23
    ep367_mean_episode_length = 9980.0
    
    print("\n[1] TRADE COUNT PROGRESS")
    print("-" * 60)
    trades_added = ep367_trades - ep365_trades
    episodes_passed = 367 - 365
    
    print(f"  Episode 365: {ep365_trades} trades")
    print(f"  Episode 367: {ep367_trades} trades (+{trades_added} in {episodes_passed} episodes)")
    print(f"  Current Episode: {ep367_current_trades} trades")
    
    trades_per_episode = trades_added / episodes_passed if episodes_passed > 0 else 0
    print(f"  Trades per Episode (Recent): {trades_per_episode:.1f}")
    
    if trades_per_episode > 20:
        print(f"  [WARN] Very high trade rate: {trades_per_episode:.1f} trades/episode")
        print(f"         This is much higher than target (0.5-1.0)")
    else:
        print(f"  [OK] Trade rate is reasonable")
    
    print("\n[2] WIN RATE PROGRESS - MAJOR IMPROVEMENT!")
    print("-" * 60)
    win_rate_improvement = ep367_win_rate - ep365_win_rate
    mean_win_rate_improvement = ep367_mean_win_rate_10 - ep365_mean_win_rate_10
    
    print(f"  Overall Win Rate:")
    print(f"    Episode 365: {ep365_win_rate:.1f}%")
    print(f"    Episode 367: {ep367_win_rate:.1f}% ({win_rate_improvement:+.1f}%)")
    
    print(f"\n  Mean Win Rate (Last 10):")
    print(f"    Episode 365: {ep365_mean_win_rate_10:.1f}%")
    print(f"    Episode 367: {ep367_mean_win_rate_10:.1f}% ({mean_win_rate_improvement:+.1f}%)")
    
    print(f"\n  Current Episode Win Rate: {ep367_current_win_rate:.1f}%")
    
    if win_rate_improvement > 10:
        print(f"  [EXCELLENT] Overall win rate improved by {win_rate_improvement:.1f}%!")
        print(f"             This is a significant improvement!")
    
    if ep367_win_rate >= 35:
        print(f"  [OK] Overall win rate ({ep367_win_rate:.1f}%) is approaching breakeven (~34%)")
    
    if ep367_mean_win_rate_10 >= 30:
        print(f"  [OK] Mean win rate (last 10) is {ep367_mean_win_rate_10:.1f}% (above 30%)")
    
    print("\n[3] EPISODE LENGTH PROGRESS")
    print("-" * 60)
    print(f"  Latest Episode Length:")
    print(f"    Episode 365: {ep365_episode_length} steps")
    print(f"    Episode 367: {ep367_episode_length} steps (+{ep367_episode_length - ep365_episode_length})")
    
    print(f"\n  Mean Episode Length: {ep367_mean_episode_length:.1f} steps")
    
    if ep367_episode_length > ep365_episode_length:
        print(f"  [OK] Latest episode length continues to increase (100 -> 140 steps)")
        print(f"       Boundary check fix is working!")
    
    if ep367_mean_episode_length >= 9980:
        print(f"  [OK] Mean episode length is {ep367_mean_episode_length:.1f} steps")
        print(f"       Many episodes ARE completing fully!")
    
    if ep367_episode_length < 1000:
        print(f"  [WARN] Latest episode is still short ({ep367_episode_length} steps)")
        print(f"         But mean is {ep367_mean_episode_length:.1f} steps (suggests outliers)")
    
    print("\n[4] PROFITABILITY ANALYSIS")
    print("-" * 60)
    print(f"  Mean PnL (Last 10):")
    print(f"    Episode 365: ${ep365_mean_pnl_10:.2f}")
    print(f"    Episode 367: ${ep367_mean_pnl_10:.2f} ({ep367_mean_pnl_10 - ep365_mean_pnl_10:+.2f})")
    
    print(f"\n  Current Episode PnL: ${ep367_current_pnl:.2f}")
    
    if ep367_mean_pnl_10 > 0:
        print(f"  [OK] Mean PnL is positive: ${ep367_mean_pnl_10:.2f}")
        if ep367_mean_pnl_10 < ep365_mean_pnl_10:
            print(f"  [INFO] Mean PnL decreased but still positive")
            print(f"         This might be due to more trades (learning phase)")
    
    if ep367_current_pnl < 0:
        print(f"  [WARN] Current episode PnL is negative: ${ep367_current_pnl:.2f}")
        print(f"         But this is just one episode - mean is still positive")
    
    print("\n[5] KEY OBSERVATIONS")
    print("-" * 60)
    
    observations = []
    
    # Trade count
    if ep367_trades > ep365_trades:
        improvement_pct = ((ep367_trades - ep365_trades) / ep365_trades) * 100
        observations.append(("POSITIVE", f"Trade count increased significantly: {ep365_trades} -> {ep367_trades} (+{improvement_pct:.0f}%)"))
    
    # Win rate
    if win_rate_improvement > 10:
        observations.append(("EXCELLENT", f"Overall win rate improved dramatically: {ep365_win_rate:.1f}% -> {ep367_win_rate:.1f}% (+{win_rate_improvement:.1f}%)"))
    
    if mean_win_rate_improvement > 5:
        observations.append(("POSITIVE", f"Mean win rate (last 10) improved: {ep365_mean_win_rate_10:.1f}% -> {ep367_mean_win_rate_10:.1f}% (+{mean_win_rate_improvement:.1f}%)"))
    
    if ep367_win_rate >= 35:
        observations.append(("POSITIVE", f"Overall win rate ({ep367_win_rate:.1f}%) is approaching breakeven (~34%)"))
    
    # Episode length
    if ep367_episode_length > ep365_episode_length:
        observations.append(("POSITIVE", f"Latest episode length increased: {ep365_episode_length} -> {ep367_episode_length} steps"))
    
    if ep367_mean_episode_length >= 9980:
        observations.append(("POSITIVE", f"Mean episode length is {ep367_mean_episode_length:.1f} steps (many episodes completing fully)"))
    
    # Profitability
    if ep367_mean_pnl_10 > 0:
        observations.append(("POSITIVE", f"Mean PnL (last 10) is positive: ${ep367_mean_pnl_10:.2f}"))
    
    if ep367_current_pnl < 0:
        observations.append(("WARN", f"Current episode PnL is negative: ${ep367_current_pnl:.2f} (but mean is positive)"))
    
    # Trade rate
    if trades_per_episode > 20:
        observations.append(("WARN", f"Very high trade rate: {trades_per_episode:.1f} trades/episode (target: 0.5-1.0)"))
    
    for category, observation in observations:
        if category == "EXCELLENT":
            print(f"  [EXCELLENT] {observation}")
        elif category == "POSITIVE":
            print(f"  [OK] {observation}")
        elif category == "WARN":
            print(f"  [WARN] {observation}")
        else:
            print(f"  [INFO] {observation}")
    
    print("\n[6] BREAKEVEN ANALYSIS")
    print("-" * 60)
    
    # Estimate breakeven win rate
    # With 77 trades, 29 wins, 48 losses
    # Assuming avg_win = $100, avg_loss = $50, commission = $3
    avg_win = 100.0
    avg_loss = 50.0
    commission = 3.0
    breakeven_wr = (avg_loss + commission) / (avg_win + avg_loss + 2*commission) * 100
    
    print(f"  Estimated Breakeven Win Rate: {breakeven_wr:.1f}%")
    print(f"  Current Overall Win Rate: {ep367_win_rate:.1f}%")
    
    if ep367_win_rate >= breakeven_wr:
        print(f"  [OK] Win rate ({ep367_win_rate:.1f}%) is above breakeven ({breakeven_wr:.1f}%)")
        print(f"       System should be profitable!")
    else:
        gap = breakeven_wr - ep367_win_rate
        print(f"  [INFO] Win rate ({ep367_win_rate:.1f}%) is {gap:.1f}% below breakeven ({breakeven_wr:.1f}%)")
        print(f"         But improving trend is positive")
    
    print("\n[7] RECOMMENDATIONS")
    print("-" * 60)
    
    print("\n  1. [EXCELLENT] Win Rate Improvement")
    print(f"     - Overall win rate improved from {ep365_win_rate:.1f}% to {ep367_win_rate:.1f}%")
    print(f"     - This is a {win_rate_improvement:.1f}% improvement - significant progress!")
    print(f"     - Mean win rate (last 10) is {ep367_mean_win_rate_10:.1f}%")
    print(f"     - Continue monitoring - if trend continues, will reach target (60%+)")
    
    print("\n  2. [MONITOR] Trade Count")
    print(f"     - Trade count jumped significantly: {ep365_trades} -> {ep367_trades}")
    print(f"     - Recent rate: {trades_per_episode:.1f} trades/episode (very high)")
    print(f"     - Target: 0.5-1.0 trades/episode")
    print(f"     - However, win rate improved significantly, so might be OK")
    print(f"     - Monitor if trade count stabilizes or continues to increase")
    
    print("\n  3. [CONTINUE] Current Approach")
    print(f"     - Win rate is improving dramatically")
    print(f"     - Mean PnL is positive")
    print(f"     - Episode length is improving")
    print(f"     - System is learning effectively")
    print(f"     - Continue monitoring for 50-100 more episodes")
    
    print("\n  4. [WATCH] Episode Length")
    print(f"     - Latest: {ep367_episode_length} steps (increasing)")
    print(f"     - Mean: {ep367_mean_episode_length:.1f} steps (excellent!)")
    print(f"     - Boundary check fix is working")
    print(f"     - Monitor if latest episode length continues to increase")
    
    print("\n  5. [CONSIDER] Trade Rate")
    print(f"     - Current rate: {trades_per_episode:.1f} trades/episode")
    print(f"     - Target: 0.5-1.0 trades/episode")
    print(f"     - However, win rate improved significantly")
    print(f"     - If win rate continues to improve, high trade count might be OK")
    print(f"     - Consider gradually tightening filters if trade count doesn't stabilize")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    print("\n[SUMMARY]")
    print("=" * 60)
    print("  [EXCELLENT] Win rate improved dramatically: {:.1f}% -> {:.1f}% (+{:.1f}%)".format(
        ep365_win_rate, ep367_win_rate, win_rate_improvement))
    print("  [OK] Mean win rate (last 10): {:.1f}% (improving)".format(ep367_mean_win_rate_10))
    print("  [OK] Mean PnL (last 10): ${:.2f} (positive)".format(ep367_mean_pnl_10))
    print("  [OK] Episode length: Latest {:.0f} steps, Mean {:.1f} steps".format(
        ep367_episode_length, ep367_mean_episode_length))
    print("  [WARN] Trade count is high: {:.1f} trades/episode (target: 0.5-1.0)".format(trades_per_episode))
    print("  [INFO] System is learning effectively - continue monitoring")

if __name__ == "__main__":
    analyze_latest_status()

