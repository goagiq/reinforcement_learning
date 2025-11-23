"""
Analyze Latest Training Metrics

Key observations from the dashboard:
- Latest Episode Length: 180 steps (very short vs mean 9980)
- Latest Reward: -0.01 (negative)
- Mean Reward (Last 10): -1.38 (negative)
- Overall Win Rate: 37.8% (below breakeven with commissions)
- Current PnL: -$171.76 (losing money)
- Mean PnL (Last 10): $166.64 (positive - inconsistent)
- Max Drawdown: 0.7%
"""

def analyze_metrics():
    """Analyze the latest training metrics"""
    
    print("="*60)
    print("TRAINING METRICS ANALYSIS")
    print("="*60)
    
    # Key metrics from dashboard
    metrics = {
        "training_progress": 82.8,  # %
        "current_episode": 369,
        "timesteps": 4140000,
        "total_timesteps": 5000000,
        "latest_reward": -0.01,
        "mean_reward_10": -1.38,
        "total_trades": 111,
        "winning_trades": 42,
        "losing_trades": 69,
        "overall_win_rate": 0.378,  # 37.8%
        "current_episode_trades": 6,
        "current_pnl": -171.76,
        "current_equity": 99828.24,
        "current_win_rate": 0.40,  # 40.0%
        "max_drawdown": 0.007,  # 0.7%
        "mean_pnl_10": 166.64,
        "mean_equity_10": 100166.64,
        "mean_win_rate_10": 0.326,  # 32.6%
        "latest_episode_length": 180,
        "mean_episode_length": 9980.0
    }
    
    print("\n1. EPISODE LENGTH ANALYSIS")
    print("-" * 60)
    print(f"Latest Episode Length: {metrics['latest_episode_length']} steps")
    print(f"Mean Episode Length: {metrics['mean_episode_length']:.0f} steps")
    print(f"Ratio: {metrics['latest_episode_length'] / metrics['mean_episode_length']:.1%}")
    
    if metrics['latest_episode_length'] < metrics['mean_episode_length'] * 0.5:
        print("\n[CRITICAL] Latest episode is significantly shorter than mean!")
        print("Possible causes:")
        print("  - Early termination due to errors")
        print("  - Data boundary issues")
        print("  - Stop loss triggered too early")
        print("  - Consecutive loss limit hit")
    
    print("\n2. REWARD ANALYSIS")
    print("-" * 60)
    print(f"Latest Reward: {metrics['latest_reward']:.2f}")
    print(f"Mean Reward (Last 10): {metrics['mean_reward_10']:.2f}")
    
    if metrics['latest_reward'] < 0 and metrics['mean_reward_10'] < 0:
        print("\n[WARNING] Both latest and mean rewards are negative")
        print("This indicates the agent is not learning effectively")
        print("Possible causes:")
        print("  - Reward function too punitive")
        print("  - Action threshold too high (not enough trades)")
        print("  - Quality filters too strict")
        print("  - Inaction penalty too high")
    
    print("\n3. WIN RATE ANALYSIS")
    print("-" * 60)
    print(f"Overall Win Rate: {metrics['overall_win_rate']:.1%}")
    print(f"Current Win Rate: {metrics['current_win_rate']:.1%}")
    print(f"Mean Win Rate (Last 10): {metrics['mean_win_rate_10']:.1%}")
    
    # Breakeven calculation (assuming 2% commission per round trip)
    breakeven_win_rate = 0.34  # ~34% with commissions
    print(f"\nBreakeven Win Rate (with commissions): ~{breakeven_win_rate:.1%}")
    
    if metrics['overall_win_rate'] < breakeven_win_rate:
        print("[WARNING] Win rate below breakeven - losing money on average")
    elif metrics['overall_win_rate'] < breakeven_win_rate + 0.05:
        print("[CAUTION] Win rate close to breakeven - minimal profit margin")
    else:
        print("[OK] Win rate above breakeven")
    
    print("\n4. PROFITABILITY ANALYSIS")
    print("-" * 60)
    print(f"Current PnL: ${metrics['current_pnl']:.2f}")
    print(f"Mean PnL (Last 10): ${metrics['mean_pnl_10']:.2f}")
    print(f"Current Equity: ${metrics['current_equity']:.2f}")
    print(f"Mean Equity (Last 10): ${metrics['mean_equity_10']:.2f}")
    
    # Calculate expected PnL from win rate
    avg_win = 150.0  # Estimated from previous analysis
    avg_loss = 100.0  # Estimated from previous analysis
    expected_pnl_per_trade = (metrics['overall_win_rate'] * avg_win) - ((1 - metrics['overall_win_rate']) * avg_loss)
    
    print(f"\nExpected PnL per trade (at {metrics['overall_win_rate']:.1%} win rate): ${expected_pnl_per_trade:.2f}")
    
    if metrics['current_pnl'] < 0 and metrics['mean_pnl_10'] > 0:
        print("\n[WARNING] Inconsistent PnL - current negative but mean positive")
        print("This suggests high variance in performance")
        print("Possible causes:")
        print("  - Recent episodes performing poorly")
        print("  - High variance in trade outcomes")
        print("  - Need more consistent risk management")
    
    print("\n5. TRADE COUNT ANALYSIS")
    print("-" * 60)
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Winning Trades: {metrics['winning_trades']}")
    print(f"Losing Trades: {metrics['losing_trades']}")
    print(f"Current Episode Trades: {metrics['current_episode_trades']}")
    
    trades_per_episode = metrics['total_trades'] / metrics['current_episode']
    print(f"\nAverage Trades per Episode: {trades_per_episode:.2f}")
    
    if trades_per_episode < 0.5:
        print("[WARNING] Very few trades per episode")
        print("Possible causes:")
        print("  - Action threshold too high")
        print("  - Quality filters too strict")
        print("  - Min confidence too high")
        print("  - Min quality score too high")
    
    print("\n6. RISK ANALYSIS")
    print("-" * 60)
    print(f"Max Drawdown: {metrics['max_drawdown']:.1%}")
    
    if metrics['max_drawdown'] > 0.10:
        print("[WARNING] High drawdown - risk management may need adjustment")
    elif metrics['max_drawdown'] > 0.05:
        print("[CAUTION] Moderate drawdown - monitor closely")
    else:
        print("[OK] Drawdown within acceptable range")
    
    print("\n7. RECOMMENDATIONS")
    print("-" * 60)
    
    recommendations = []
    
    # Episode length issue
    if metrics['latest_episode_length'] < metrics['mean_episode_length'] * 0.5:
        recommendations.append({
            "priority": "HIGH",
            "issue": "Short episode length (180 vs 9980 mean)",
            "action": "Investigate episode termination logic - check for early exits, data boundary issues, or consecutive loss limits"
        })
    
    # Negative rewards
    if metrics['latest_reward'] < 0 and metrics['mean_reward_10'] < 0:
        recommendations.append({
            "priority": "HIGH",
            "issue": "Negative rewards (latest: -0.01, mean: -1.38)",
            "action": "Review reward function - may need to adjust inaction penalty, reduce quality filter strictness, or lower action threshold"
        })
    
    # Win rate below breakeven
    if metrics['overall_win_rate'] < breakeven_win_rate:
        recommendations.append({
            "priority": "MEDIUM",
            "issue": f"Win rate {metrics['overall_win_rate']:.1%} below breakeven {breakeven_win_rate:.1%}",
            "action": "Focus on trade quality - may need to tighten filters further or improve entry/exit logic"
        })
    
    # Inconsistent PnL
    if metrics['current_pnl'] < 0 and metrics['mean_pnl_10'] > 0:
        recommendations.append({
            "priority": "MEDIUM",
            "issue": "Inconsistent PnL (current negative, mean positive)",
            "action": "Review recent episodes - may indicate need for more consistent risk management or parameter stability"
        })
    
    # Low trade count
    if trades_per_episode < 0.5:
        recommendations.append({
            "priority": "MEDIUM",
            "issue": f"Low trade count ({trades_per_episode:.2f} trades/episode)",
            "action": "Consider relaxing quality filters or lowering action threshold to allow more exploration"
        })
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. [{rec['priority']}] {rec['issue']}")
        print(f"   Action: {rec['action']}")
    
    if not recommendations:
        print("\n[OK] No critical issues detected - training appears to be progressing normally")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    analyze_metrics()
