"""
Comprehensive Training Performance Analysis
Analyzes training progress, trade performance, and provides remediation recommendations
"""

import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys

# Try to use requests, fallback to file reading if API not available
try:
    import requests
    USE_API = True
except ImportError:
    USE_API = False


def fetch_api_data(endpoint: str) -> Optional[Dict]:
    """Fetch data from API endpoint"""
    if not USE_API:
        return None
    try:
        response = requests.get(f"http://localhost:8200{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"[WARNING] Could not fetch {endpoint}: {e}")
    return None


def load_json_file(filepath: Path) -> Optional[Dict]:
    """Load JSON file"""
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not load {filepath}: {e}")
    return None


def load_jsonl_file(filepath: Path) -> List[Dict]:
    """Load JSONL file"""
    data = []
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        except Exception as e:
            print(f"[WARNING] Could not load {filepath}: {e}")
    return data


def analyze_training_performance():
    """Main analysis function"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TRAINING PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Fetch Training Status
    print("[1] TRAINING STATUS")
    print("-" * 80)
    training_status = fetch_api_data("/api/training/status")
    
    if training_status:
        episode = training_status.get("episode", 0)
        timestep = training_status.get("timestep", 0)
        total_timesteps = training_status.get("total_timesteps", 0)
        progress = training_status.get("progress_percent", 0)
        latest_reward = training_status.get("latest_reward", 0)
        mean_reward_10 = training_status.get("mean_reward_10", 0)
        current_episode_trades = training_status.get("current_episode_trades", 0)
        overall_win_rate = training_status.get("overall_win_rate", 0)
        current_episode_pnl = training_status.get("current_episode_pnl", 0)
        mean_pnl_10 = training_status.get("mean_pnl_10", 0)
        
        print(f"  Episode: {episode}")
        print(f"  Timestep: {timestep:,} / {total_timesteps:,} ({progress:.1f}%)")
        print(f"  Latest Reward: {latest_reward:.4f}")
        print(f"  Mean Reward (Last 10): {mean_reward_10:.4f}")
        print(f"  Current Episode Trades: {current_episode_trades}")
        print(f"  Overall Win Rate: {overall_win_rate*100:.1f}%")
        print(f"  Current Episode PnL: ${current_episode_pnl:.2f}")
        print(f"  Mean PnL (Last 10): ${mean_pnl_10:.2f}")
    else:
        print("  [WARNING] Could not fetch training status from API")
        episode = 0
        overall_win_rate = 0
        mean_pnl_10 = 0
    
    # 2. Fetch Trade Performance
    print("\n[2] TRADE PERFORMANCE")
    print("-" * 80)
    performance = fetch_api_data("/api/monitoring/performance")
    
    if performance and performance.get("status") == "success":
        metrics = performance.get("metrics", {})
        total_pnl = metrics.get("total_pnl", 0)
        total_trades = metrics.get("total_trades", 0)
        win_rate = metrics.get("win_rate", 0)
        profit_factor = metrics.get("profit_factor", 0)
        sharpe_ratio = metrics.get("sharpe_ratio", 0)
        max_drawdown = metrics.get("max_drawdown", 0)
        avg_trade = metrics.get("average_trade", 0)
        
        status = "PROFITABLE" if total_pnl >= 0 else "LOSING"
        print(f"  Total P&L: ${total_pnl:,.2f} [{status}]")
        print(f"  Total Trades: {total_trades}")
        print(f"  Win Rate: {win_rate*100:.1f}%")
        print(f"  Profit Factor: {profit_factor:.2f}")
        print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"  Max Drawdown: ${max_drawdown:,.2f}")
        print(f"  Average Trade: ${avg_trade:.2f}")
    else:
        print("  [WARNING] Could not fetch performance metrics from API")
        total_pnl = 0
        total_trades = 0
        win_rate = 0
        profit_factor = 0
    
    # 3. Analyze Adaptive Learning Adjustments
    print("\n[3] ADAPTIVE LEARNING STATUS")
    print("-" * 80)
    adaptive_config_path = Path("logs/adaptive_training/current_reward_config.json")
    adaptive_config = load_json_file(adaptive_config_path)
    
    if adaptive_config:
        print("  Current Adaptive Parameters:")
        print(f"    Entropy Coefficient: {adaptive_config.get('entropy_coef', 'N/A')}")
        print(f"    Inaction Penalty: {adaptive_config.get('inaction_penalty', 'N/A')}")
        print(f"    Learning Rate: {adaptive_config.get('learning_rate', 'N/A')}")
        print(f"    Risk/Reward Ratio: {adaptive_config.get('risk_reward_ratio', 'N/A')}")
        print(f"    Min Quality Score: {adaptive_config.get('min_quality_score', 'N/A')}")
        print(f"    Stop Loss %: {adaptive_config.get('stop_loss_pct', 'N/A')}")
        
        # Load adjustment history
        adjustments_path = Path("logs/adaptive_training/config_adjustments.jsonl")
        adjustments = load_jsonl_file(adjustments_path)
        if adjustments:
            print(f"\n  Total Adjustments Made: {len(adjustments)}")
            if adjustments:
                last_adjustment = adjustments[-1]
                print(f"  Last Adjustment Timestep: {last_adjustment.get('timestep', 'N/A')}")
                if 'adjustments' in last_adjustment:
                    print(f"  Last Adjustment Type: {', '.join(last_adjustment['adjustments'].keys())}")
    else:
        print("  [WARNING] Could not load adaptive learning configuration")
    
    # 4. Load Recent Trades for Detailed Analysis
    print("\n[4] DETAILED TRADE ANALYSIS")
    print("-" * 80)
    
    # Try to get trades from journal database or API
    trades_data = fetch_api_data("/api/journal/trades?limit=100")
    
    if trades_data and trades_data.get("status") == "success":
        trades = trades_data.get("trades", [])
        if trades:
            df = pd.DataFrame(trades)
            
            winning_trades = df[df.get("is_win", False) == 1] if "is_win" in df.columns else df[df.get("net_pnl", 0) > 0]
            losing_trades = df[df.get("is_win", False) == 0] if "is_win" in df.columns else df[df.get("net_pnl", 0) <= 0]
            
            if len(winning_trades) > 0:
                avg_win = winning_trades['net_pnl'].mean()
                print(f"  Average Win: ${avg_win:.2f}")
            else:
                print(f"  Average Win: N/A (no winning trades)")
            
            if len(losing_trades) > 0:
                avg_loss = abs(losing_trades['net_pnl'].mean())
                print(f"  Average Loss: ${avg_loss:.2f}")
            else:
                print(f"  Average Loss: N/A (no losing trades)")
            
            if len(winning_trades) > 0 and len(losing_trades) > 0:
                risk_reward = avg_win / avg_loss
                print(f"  Risk/Reward Ratio: {risk_reward:.2f}")
                breakeven_win_rate = 1 / (1 + risk_reward)
                print(f"  Breakeven Win Rate: {breakeven_win_rate*100:.1f}%")
                
                # Compare to actual win rate
                actual_win_rate = len(winning_trades) / len(trades)
                if actual_win_rate < breakeven_win_rate:
                    gap = (breakeven_win_rate - actual_win_rate) * 100
                    print(f"  [WARNING] Win Rate Gap: {gap:.1f}% BELOW breakeven (unprofitable)")
                else:
                    gap = (actual_win_rate - breakeven_win_rate) * 100
                    print(f"  [OK] Win Rate Gap: {gap:.1f}% ABOVE breakeven (profitable)")
            
            # Analyze recent trends
            if len(trades) >= 10:
                recent_10 = trades[:10]  # Most recent 10
                recent_wins = sum(1 for t in recent_10 if t.get("is_win") or t.get("net_pnl", 0) > 0)
                recent_win_rate = recent_wins / 10
                print(f"\n  Recent 10 Trades Win Rate: {recent_win_rate*100:.0f}%")
                
                if len(trades) >= 20:
                    earlier_10 = trades[10:20]
                    earlier_wins = sum(1 for t in earlier_10 if t.get("is_win") or t.get("net_pnl", 0) > 0)
                    earlier_win_rate = earlier_wins / 10
                    trend = "improving" if recent_win_rate > earlier_win_rate else "declining"
                    print(f"  Previous 10 Trades Win Rate: {earlier_win_rate*100:.0f}%")
                    print(f"  Trend: {trend.upper()}")
        else:
            print("  [WARNING] No trades found in journal")
    else:
        print("  [WARNING] Could not fetch trade data from API")
    
    # 5. Critical Issues Analysis
    print("\n[5] CRITICAL ISSUES IDENTIFIED")
    print("-" * 80)
    issues = []
    
    # Issue 1: Negative Total P&L
    if total_pnl < 0:
        issues.append({
            "severity": "CRITICAL",
            "issue": f"Total P&L is negative: ${total_pnl:,.2f}",
            "impact": "System is losing money overall"
        })
    
    # Issue 2: Low Win Rate
    if win_rate < 0.35:
        issues.append({
            "severity": "CRITICAL",
            "issue": f"Win rate is too low: {win_rate*100:.1f}% (target: 55%+)",
            "impact": "System is not selecting profitable trades"
        })
    
    # Issue 3: Profit Factor below 1.0
    if profit_factor > 0 and profit_factor < 1.0:
        issues.append({
            "severity": "CRITICAL",
            "issue": f"Profit factor is below 1.0: {profit_factor:.2f}",
            "impact": "System is unprofitable even before commissions"
        })
    
    # Issue 4: Very low trade count
    trades_per_episode = total_trades / episode if episode > 0 else 0
    if trades_per_episode < 0.1 and episode > 50:
        issues.append({
            "severity": "HIGH",
            "issue": f"Trade count is extremely low: {trades_per_episode:.3f} trades/episode",
            "impact": "System is too conservative, not learning from enough trades"
        })
    
    # Issue 5: Large drawdown
    if max_drawdown > 5000:  # More than $5k drawdown
        issues.append({
            "severity": "HIGH",
            "issue": f"Large drawdown: ${max_drawdown:,.2f}",
            "impact": "Risk management may need adjustment"
        })
    
    # Issue 6: Negative recent rewards
    if training_status and mean_reward_10 < 0:
        issues.append({
            "severity": "HIGH",
            "issue": f"Recent rewards are negative: {mean_reward_10:.4f}",
            "impact": "Agent is not learning profitable strategies"
        })
    
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"\n  [{issue['severity']}] Issue {i}: {issue['issue']}")
            print(f"        Impact: {issue['impact']}")
    else:
        print("  [OK] No critical issues identified")
    
    # 6. Recommendations
    print("\n[6] REMEDIATION RECOMMENDATIONS")
    print("-" * 80)
    
    recommendations = []
    
    # Recommendation 1: Fix Win Rate
    if win_rate < 0.50:
        recommendations.append({
            "priority": "URGENT",
            "action": "Improve Trade Quality Filtering",
            "steps": [
                "1. Increase min_quality_score threshold (currently may be too low)",
                "2. Tighten confidence requirements for trade entries",
                "3. Improve quality scorer to better identify profitable setups",
                "4. Consider increasing min_combined_confidence in DecisionGate",
                "5. Review and improve entry signal quality criteria"
            ]
        })
    
    # Recommendation 2: Increase Trade Frequency (if too low)
    if trades_per_episode < 0.3 and episode > 50:
        recommendations.append({
            "priority": "HIGH",
            "action": "Increase Trade Frequency",
            "steps": [
                "1. Reduce action_threshold (currently 0.05, try 0.03-0.04)",
                "2. Reduce min_combined_confidence (currently 0.5, try 0.45)",
                "3. Reduce min_quality_score slightly (but not too much - quality first)",
                "4. Check DecisionGate filters - may be too restrictive",
                "5. Ensure adaptive learning is adjusting filters appropriately"
            ]
        })
    
    # Recommendation 3: Improve Risk/Reward
    if profit_factor > 0 and profit_factor < 1.2:
        recommendations.append({
            "priority": "HIGH",
            "action": "Improve Risk/Reward Ratio",
            "steps": [
                "1. Tighten stop-losses (use adaptive stop-loss based on volatility)",
                "2. Let winners run longer (improve exit strategy)",
                "3. Review position sizing - may need to reduce size on weak signals",
                "4. Improve entry timing to catch better setups",
                "5. Review adaptive learning stop-loss adjustments"
            ]
        })
    
    # Recommendation 4: Stop Losing Streak
    if total_pnl < 0:
        recommendations.append({
            "priority": "URGENT",
            "action": "Stop Current Losing Streak",
            "steps": [
                "1. PAUSE TRAINING temporarily to prevent further losses",
                "2. Review recent trades to identify patterns in losses",
                "3. Tighten all filters aggressively (confidence, quality, stop-loss)",
                "4. Reduce position sizes until profitability returns",
                "5. Consider rolling back to a previous checkpoint if available"
            ]
        })
    
    # Recommendation 5: Adaptive Learning Tuning
    if adaptive_config:
        recommendations.append({
            "priority": "MEDIUM",
            "action": "Optimize Adaptive Learning",
            "steps": [
                "1. Review adaptive learning adjustment history",
                "2. Ensure eval_frequency is appropriate (every 5000 timesteps)",
                "3. Verify adaptive learning is responding to poor performance",
                "4. Check if adjustments are being applied correctly",
                "5. Consider more aggressive adjustments if performance is poor"
            ]
        })
    
    # Recommendation 6: Reward Function Review
    if training_status and mean_reward_10 < 0:
        recommendations.append({
            "priority": "HIGH",
            "action": "Review Reward Function",
            "steps": [
                "1. Check if inaction penalty is too high",
                "2. Verify reward scaling is appropriate",
                "3. Ensure rewards properly incentivize profitable trades",
                "4. Review quality filter impact on rewards",
                "5. Check if commission costs are properly reflected"
            ]
        })
    
    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\n  [{rec['priority']}] Recommendation {i}: {rec['action']}")
        for step in rec['steps']:
            print(f"      {step}")
    
    # 7. Immediate Action Plan
    print("\n[7] IMMEDIATE ACTION PLAN")
    print("-" * 80)
    
    if total_pnl < 0 or win_rate < 0.40:
        print("\n  [CRITICAL] System is unprofitable. Take immediate action:")
        print("\n      STEP 1: Pause training (if not already paused)")
        print("              - Go to Training tab and click 'Stop Training'")
        print("              - This prevents further capital loss")
        print("\n      STEP 2: Review configuration files:")
        print("              - Check config/reward_config.yaml")
        print("              - Review adaptive learning adjustments")
        print("              - Verify quality filter settings")
        print("\n      STEP 3: Tighten filters aggressively:")
        print("              - Increase min_quality_score to 0.50-0.60")
        print("              - Increase min_combined_confidence to 0.60")
        print("              - Reduce action_threshold to 0.03 (to allow more trades)")
        print("              - Tighten stop-loss (adaptive should help)")
        print("\n      STEP 4: Restart training with tighter filters")
        print("              - Monitor first 10-20 trades closely")
        print("              - If still losing, pause and review again")
    else:
        print("\n  [MONITOR] System shows mixed performance. Continue monitoring:")
        print("\n      - Track win rate trend (is it improving?)")
        print("      - Monitor total P&L trajectory")
        print("      - Watch for consecutive losses")
        print("      - Review adaptive learning adjustments")
    
    # Summary
    print("\n[8] SUMMARY")
    print("-" * 80)
    pnl_status = "PROFITABLE" if total_pnl >= 0 else "LOSING MONEY"
    wr_status = "GOOD" if win_rate >= 0.55 else "NEEDS IMPROVEMENT" if win_rate >= 0.40 else "TOO LOW"
    pf_status = "GOOD" if profit_factor >= 1.5 else "MARGINAL" if profit_factor >= 1.0 else "POOR"
    print(f"  Total P&L: ${total_pnl:,.2f} [{pnl_status}]")
    print(f"  Win Rate: {win_rate*100:.1f}% [{wr_status}]")
    print(f"  Profit Factor: {profit_factor:.2f} [{pf_status}]")
    print(f"  Total Trades: {total_trades} ({trades_per_episode:.3f} per episode)")
    print(f"  Critical Issues: {len([i for i in issues if i['severity'] == 'CRITICAL'])}")
    print(f"  Recommendations: {len(recommendations)}")
    
    if total_pnl < 0:
        print("\n  [WARNING] System is currently unprofitable. Review recommendations above.")
    
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        analyze_training_performance()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

