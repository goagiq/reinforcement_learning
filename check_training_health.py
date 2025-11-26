"""
Comprehensive Training Health and Progress Check

This script provides a detailed analysis of:
1. Current training status and progress
2. Training metrics with explanations
3. Trading performance during training
4. Health assessment and recommendations
5. What the numbers mean

Usage:
    python check_training_health.py
"""

import sys
from pathlib import Path
import requests
import json
from datetime import datetime
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def get_training_status() -> Optional[Dict]:
    """Get current training status from API"""
    try:
        response = requests.get("http://localhost:8200/api/training/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to API server at http://localhost:8200")
        print("  - Is the backend running? Start with: python start_ui.py")
        return None
    except Exception as e:
        print(f"[ERROR] Error getting training status: {e}")
        return None

def explain_metric(name: str, value: any, context: Dict = None) -> str:
    """Explain what a metric means"""
    explanations = {
        "timestep": {
            "desc": "Number of training steps completed",
            "good": "Increasing steadily",
            "bad": "Stuck or not increasing"
        },
        "episode": {
            "desc": "Number of completed trading episodes",
            "good": "Increasing (agent is learning)",
            "bad": "Stuck at 0 or very low"
        },
        "mean_reward": {
            "desc": "Average reward per episode (higher is better)",
            "good": "Positive and increasing",
            "bad": "Negative or decreasing"
        },
        "current_episode_pnl": {
            "desc": "Profit/Loss in current episode",
            "good": "Positive",
            "bad": "Large negative"
        },
        "current_episode_win_rate": {
            "desc": "Percentage of winning trades in current episode",
            "good": "> 50%",
            "bad": "< 30%"
        },
        "overall_win_rate": {
            "desc": "Percentage of winning trades across all episodes",
            "good": "> 50%",
            "bad": "< 30%"
        },
        "total_trades": {
            "desc": "Total number of trades executed",
            "good": "Increasing (agent is trading)",
            "bad": "0 or very low (agent too conservative)"
        },
        "risk_reward_ratio": {
            "desc": "Average win / Average loss (higher is better)",
            "good": "> 2.0",
            "bad": "< 1.0"
        },
        "avg_win": {
            "desc": "Average profit per winning trade",
            "good": "Positive and increasing",
            "bad": "Negative or very small"
        },
        "avg_loss": {
            "desc": "Average loss per losing trade (absolute value)",
            "good": "Smaller than avg_win",
            "bad": "Larger than avg_win"
        },
        "mean_pnl_10": {
            "desc": "Average PnL over last 10 episodes",
            "good": "Positive and trending up",
            "bad": "Negative or trending down"
        },
        "mean_equity_10": {
            "desc": "Average ending equity over last 10 episodes",
            "good": "Above initial capital ($100,000)",
            "bad": "Below initial capital"
        },
        "current_episode_max_drawdown": {
            "desc": "Maximum equity drop in current episode",
            "good": "< 5%",
            "bad": "> 10%"
        }
    }
    
    exp = explanations.get(name, {})
    if not exp:
        return f"{name}: {value}"
    
    status = "GOOD" if value and (isinstance(value, (int, float)) and value > 0) else "CHECK"
    if name in ["mean_reward", "current_episode_pnl", "mean_pnl_10"]:
        status = "GOOD" if value > 0 else "CHECK"
    elif name in ["current_episode_win_rate", "overall_win_rate"]:
        status = "GOOD" if value > 0.5 else "CHECK"
    elif name in ["risk_reward_ratio"]:
        status = "GOOD" if value > 2.0 else "CHECK"
    
    return f"""
  {name}: {value}
    Description: {exp.get('desc', 'N/A')}
    Good: {exp.get('good', 'N/A')}
    Bad: {exp.get('bad', 'N/A')}
    Status: {status}
"""

def analyze_training_health(status: Dict) -> Dict:
    """Analyze training health and provide recommendations"""
    health = {
        "status": "unknown",
        "issues": [],
        "warnings": [],
        "recommendations": [],
        "strengths": []
    }
    
    if not status or status.get("status") != "running":
        health["status"] = "not_running"
        health["issues"].append("Training is not currently running")
        return health
    
    metrics = status.get("metrics", {})
    
    # Check critical issues
    total_trades = metrics.get("total_trades", 0)
    if total_trades == 0:
        health["status"] = "critical"
        health["issues"].append("CRITICAL: No trades executed - agent is too conservative")
        health["recommendations"].append("Increase entropy_coef in config (currently may be too low)")
        health["recommendations"].append("Check action_threshold - may be too high")
        health["recommendations"].append("Review quality filters - may be too strict")
    elif total_trades < 5:
        health["status"] = "warning"
        health["warnings"].append(f"Very few trades ({total_trades}) - agent is conservative")
        health["recommendations"].append("Consider increasing exploration (entropy_coef)")
    
    # Check win rate (already in percentage form 0-100)
    overall_win_rate = metrics.get("overall_win_rate", 0)
    if overall_win_rate > 0:
        if overall_win_rate < 30:
            health["warnings"].append(f"Low win rate ({overall_win_rate:.1f}%)")
            health["recommendations"].append("Model may need more training")
            health["recommendations"].append("Check reward function - may need adjustment")
        elif overall_win_rate > 60:
            health["strengths"].append(f"Good win rate ({overall_win_rate:.1f}%)")
    
    # Check PnL
    mean_pnl = metrics.get("mean_pnl_10", 0)
    if mean_pnl < 0:
        health["warnings"].append(f"Negative average PnL (${mean_pnl:.2f})")
        health["recommendations"].append("Review reward function alignment with PnL")
        health["recommendations"].append("Check if transaction costs are too high")
    elif mean_pnl > 1000:
        health["strengths"].append(f"Positive average PnL (${mean_pnl:.2f})")
    
    # Check risk/reward
    rr_ratio = metrics.get("risk_reward_ratio", 0)
    if rr_ratio > 0:
        if rr_ratio < 1.0:
            health["warnings"].append(f"Poor risk/reward ratio ({rr_ratio:.2f})")
            health["recommendations"].append("Average losses are larger than wins")
            health["recommendations"].append("Consider tightening stop-loss or improving entry timing")
        elif rr_ratio > 2.0:
            health["strengths"].append(f"Good risk/reward ratio ({rr_ratio:.2f})")
    
    # Check progress
    timestep = metrics.get("timestep", 0)
    total_timesteps = metrics.get("total_timesteps", 0)
    if total_timesteps > 0:
        progress = (timestep / total_timesteps) * 100
        if progress < 1:
            health["warnings"].append(f"Very early in training ({progress:.1f}% complete)")
            health["recommendations"].append("Give training more time - early stages can be volatile")
        elif progress > 50:
            health["strengths"].append(f"Training progress: {progress:.1f}% complete")
    
    # Overall status
    if not health["issues"] and not health["warnings"]:
        health["status"] = "healthy"
    elif health["issues"]:
        health["status"] = "critical"
    else:
        health["status"] = "warning"
    
    return health

def print_training_health_report(status: Dict):
    """Print comprehensive training health report"""
    print("\n" + "=" * 80)
    print("TRAINING HEALTH AND PROGRESS REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if not status:
        print("[ERROR] Could not get training status")
        print("  - Is training running?")
        print("  - Is the API server running?")
        return
    
    # Training Status
    print("=" * 80)
    print("TRAINING STATUS")
    print("=" * 80)
    print(f"Status: {status.get('status', 'unknown').upper()}")
    print(f"Message: {status.get('message', 'N/A')}")
    print()
    
    metrics = status.get("metrics", {})
    if not metrics:
        print("[INFO] No metrics available yet")
        return
    
    # Progress Metrics
    print("=" * 80)
    print("PROGRESS METRICS")
    print("=" * 80)
    timestep = metrics.get("timestep", 0)
    total_timesteps = metrics.get("total_timesteps", 0)
    progress = (timestep / total_timesteps * 100) if total_timesteps > 0 else 0
    
    print(f"Timesteps: {timestep:,} / {total_timesteps:,} ({progress:.1f}% complete)")
    print(f"Episodes: {metrics.get('episode', 0)}")
    print(f"Current Episode: {metrics.get('episode', 0) + 1}")
    print()
    
    # Trading Metrics
    print("=" * 80)
    print("TRADING METRICS (What They Mean)")
    print("=" * 80)
    
    print(f"\nTotal Trades: {metrics.get('total_trades', 0)}")
    print("  Description: Total number of trades executed across all episodes")
    print("  Good: Increasing steadily (agent is learning to trade)")
    print("  Bad: 0 or very low (agent too conservative, not trading)")
    print(f"  Status: {'GOOD' if metrics.get('total_trades', 0) > 0 else 'ISSUE - No trades!'}")
    
    print(f"\nWinning Trades: {metrics.get('total_winning_trades', 0)}")
    print(f"Losing Trades: {metrics.get('total_losing_trades', 0)}")
    
    overall_win_rate = metrics.get("overall_win_rate", 0)
    # Win rate is already in percentage form (0-100), not decimal (0-1)
    overall_win_rate_decimal = overall_win_rate / 100.0 if overall_win_rate > 1.0 else overall_win_rate
    print(f"\nOverall Win Rate: {overall_win_rate:.1f}%")
    print("  Description: Percentage of trades that were profitable")
    print("  Good: > 50% (more wins than losses)")
    print("  Bad: < 30% (losing more often than winning)")
    print(f"  Status: {'GOOD' if overall_win_rate > 50 else 'CHECK' if overall_win_rate > 30 else 'ISSUE'}")
    
    current_win_rate = metrics.get("current_episode_win_rate", 0)
    print(f"\nCurrent Episode Win Rate: {current_win_rate:.1f}%")
    print("  Description: Win rate in the current (in-progress) episode")
    print("  Note: This may change as the episode continues")
    
    # PnL Metrics
    print("\n" + "-" * 80)
    print("PROFITABILITY METRICS")
    print("-" * 80)
    
    current_pnl = metrics.get("current_episode_pnl", 0)
    print(f"\nCurrent Episode PnL: ${current_pnl:.2f}")
    print("  Description: Profit/Loss in the current episode")
    print("  Good: Positive (making money)")
    print("  Bad: Large negative (losing money)")
    print(f"  Status: {'GOOD' if current_pnl > 0 else 'CHECK'}")
    
    mean_pnl = metrics.get("mean_pnl_10", 0)
    print(f"\nMean PnL (Last 10 Episodes): ${mean_pnl:.2f}")
    print("  Description: Average profit/loss over the last 10 completed episodes")
    print("  Good: Positive and increasing (trending upward)")
    print("  Bad: Negative or decreasing")
    print(f"  Status: {'GOOD' if mean_pnl > 0 else 'CHECK'}")
    
    current_equity = metrics.get("current_episode_equity", 0)
    print(f"\nCurrent Equity: ${current_equity:,.2f}")
    print("  Description: Current account balance in this episode")
    print("  Good: Above initial capital ($100,000)")
    print("  Bad: Below initial capital")
    print(f"  Status: {'GOOD' if current_equity >= 100000 else 'CHECK'}")
    
    mean_equity = metrics.get("mean_equity_10", 0)
    print(f"\nMean Equity (Last 10 Episodes): ${mean_equity:,.2f}")
    print("  Description: Average ending equity over last 10 episodes")
    
    # Risk Metrics
    print("\n" + "-" * 80)
    print("RISK METRICS")
    print("-" * 80)
    
    rr_ratio = metrics.get("risk_reward_ratio", 0)
    print(f"\nRisk/Reward Ratio: {rr_ratio:.2f}")
    print("  Description: Average win / Average loss")
    print("  Good: > 2.0 (wins are 2x larger than losses)")
    print("  Bad: < 1.0 (losses are larger than wins)")
    print(f"  Status: {'GOOD' if rr_ratio > 2.0 else 'CHECK' if rr_ratio > 1.0 else 'ISSUE'}")
    
    avg_win = metrics.get("avg_win", 0)
    print(f"\nAverage Win: ${avg_win:.2f}")
    print("  Description: Average profit per winning trade")
    print("  Good: Positive and increasing")
    
    avg_loss = metrics.get("avg_loss", 0)
    print(f"\nAverage Loss: ${avg_loss:.2f}")
    print("  Description: Average loss per losing trade (absolute value)")
    print("  Good: Smaller than average win")
    print("  Bad: Larger than average win")
    
    max_drawdown = metrics.get("current_episode_max_drawdown", 0)
    print(f"\nMax Drawdown (Current Episode): {max_drawdown*100:.2f}%")
    print("  Description: Maximum equity drop from peak in current episode")
    print("  Good: < 5% (low risk)")
    print("  Bad: > 10% (high risk)")
    print(f"  Status: {'GOOD' if max_drawdown < 0.05 else 'CHECK' if max_drawdown < 0.10 else 'ISSUE'}")
    
    # Training Metrics
    print("\n" + "-" * 80)
    print("TRAINING METRICS (Learning Progress)")
    print("-" * 80)
    
    latest_reward = metrics.get("latest_reward", 0)
    print(f"\nLatest Episode Reward: {latest_reward:.2f}")
    print("  Description: Reward from the most recent completed episode")
    print("  Good: Positive and increasing")
    print("  Bad: Negative or decreasing")
    
    mean_reward_10 = metrics.get("mean_reward_10", 0)
    print(f"\nMean Reward (Last 10 Episodes): {mean_reward_10:.2f}")
    print("  Description: Average reward over last 10 episodes")
    print("  Good: Positive and trending up")
    print("  Bad: Negative or trending down")
    
    # Health Analysis
    print("\n" + "=" * 80)
    print("HEALTH ANALYSIS")
    print("=" * 80)
    
    health = analyze_training_health(status)
    
    print(f"\nOverall Status: {health['status'].upper()}")
    
    if health["strengths"]:
        print("\n[STRENGTHS]")
        for strength in health["strengths"]:
            print(f"  + {strength}")
    
    if health["issues"]:
        print("\n[CRITICAL ISSUES]")
        for issue in health["issues"]:
            print(f"  ! {issue}")
    
    if health["warnings"]:
        print("\n[WARNINGS]")
        for warning in health["warnings"]:
            print(f"  ⚠ {warning}")
    
    if health["recommendations"]:
        print("\n[RECOMMENDATIONS]")
        for i, rec in enumerate(health["recommendations"], 1):
            print(f"  {i}. {rec}")
    
    if not health["issues"] and not health["warnings"]:
        print("\n[OK] Training appears healthy!")
        print("  - Trades are being executed")
        print("  - Metrics are within reasonable ranges")
        print("  - Continue training and monitor progress")
    
    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)

def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("TRAINING HEALTH CHECK")
    print("=" * 80)
    print("This script analyzes your training progress and explains what the metrics mean")
    print()
    
    status = get_training_status()
    print_training_health_report(status)
    
    print("\n[INFO] Run this script anytime to check training health")
    print("[INFO] The Monitoring tab in the frontend will show this data automatically")

if __name__ == "__main__":
    main()

