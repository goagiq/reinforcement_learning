"""
Analyze Losing Trades - Phase 3 Task 3.1

Analyzes recent losing trades to identify common patterns and issues.
Compares winners vs losers to understand what differentiates them.

Questions to Answer:
- Are stop-losses too tight?
- Are take-profits too far?
- Are entries happening at bad times?
- Are quality filters letting bad trades through?
"""

import sys
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import json

# Add project root to path
try:
    project_root = Path(__file__).parent.parent
except NameError:
    project_root = Path.cwd()
sys.path.insert(0, str(project_root))

# Configure stdout for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def load_trades_from_journal(db_path: str = "logs/trading_journal.db", limit: int = 1000) -> pd.DataFrame:
    """Load recent trades from trading journal"""
    db_path_full = project_root / db_path
    
    if not db_path_full.exists():
        print(f"[ERROR] Trading journal not found at: {db_path_full}")
        return pd.DataFrame()
    
    conn = sqlite3.connect(str(db_path_full))
    
    try:
        query = """
            SELECT 
                trade_id,
                timestamp,
                episode,
                step,
                entry_price,
                exit_price,
                position_size,
                pnl,
                commission,
                net_pnl,
                strategy,
                strategy_confidence,
                is_win,
                duration_steps,
                entry_timestamp,
                exit_timestamp,
                market_conditions,
                decision_metadata
            FROM trades
            ORDER BY timestamp DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        
        if len(df) == 0:
            print("[WARN] No trades found in journal")
            return df
        
        # Convert is_win to boolean
        df['is_win'] = df['is_win'].astype(bool)
        
        # Calculate additional metrics
        df['pnl_pct'] = ((df['exit_price'] - df['entry_price']) / df['entry_price']) * np.sign(df['position_size'])
        df['abs_pnl'] = df['net_pnl'].abs()
        df['abs_pnl_pct'] = df['pnl_pct'].abs()
        
        # Parse market conditions if available
        if 'market_conditions' in df.columns:
            df['market_conditions_parsed'] = df['market_conditions'].apply(
                lambda x: json.loads(x) if x and isinstance(x, str) else {}
            )
        
        return df
        
    except Exception as e:
        print(f"[ERROR] Failed to load trades: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def analyze_entry_timing(trades: pd.DataFrame) -> Dict:
    """Analyze entry timing patterns"""
    if len(trades) == 0:
        return {}
    
    # Extract hour from timestamp if available
    if 'entry_timestamp' in trades.columns:
        trades['entry_hour'] = pd.to_datetime(trades['entry_timestamp'], errors='coerce').dt.hour
    elif 'timestamp' in trades.columns:
        trades['entry_hour'] = pd.to_datetime(trades['timestamp'], errors='coerce').dt.hour
    else:
        return {"error": "No timestamp data available"}
    
    # Analyze by hour
    hourly_stats = trades.groupby('entry_hour').agg({
        'is_win': ['count', 'sum', 'mean'],
        'net_pnl': 'mean',
        'pnl_pct': 'mean'
    }).round(4)
    
    hourly_stats.columns = ['total_trades', 'wins', 'win_rate', 'avg_pnl', 'avg_pnl_pct']
    
    # Find worst hours
    worst_hours = hourly_stats.sort_values('win_rate').head(5)
    best_hours = hourly_stats.sort_values('win_rate', ascending=False).head(5)
    
    return {
        "hourly_stats": hourly_stats.to_dict('index'),
        "worst_hours": worst_hours.to_dict('index'),
        "best_hours": best_hours.to_dict('index'),
        "summary": {
            "total_hours_analyzed": len(hourly_stats),
            "worst_hour": worst_hours.index[0] if len(worst_hours) > 0 else None,
            "best_hour": best_hours.index[0] if len(best_hours) > 0 else None
        }
    }


def analyze_stop_loss_patterns(trades: pd.DataFrame) -> Dict:
    """Analyze stop-loss patterns (if available)"""
    if len(trades) == 0:
        return {}
    
    losers = trades[~trades['is_win']].copy()
    
    if len(losers) == 0:
        return {"error": "No losing trades to analyze"}
    
    # Calculate loss percentage
    losers['loss_pct'] = losers['pnl_pct'].abs()
    
    # Analyze loss distribution
    loss_stats = {
        "avg_loss_pct": float(losers['loss_pct'].mean()),
        "median_loss_pct": float(losers['loss_pct'].median()),
        "max_loss_pct": float(losers['loss_pct'].max()),
        "min_loss_pct": float(losers['loss_pct'].min()),
        "std_loss_pct": float(losers['loss_pct'].std()),
        "loss_distribution": {
            "under_1pct": int((losers['loss_pct'] < 0.01).sum()),
            "1_to_1.5pct": int(((losers['loss_pct'] >= 0.01) & (losers['loss_pct'] < 0.015)).sum()),
            "1.5_to_2pct": int(((losers['loss_pct'] >= 0.015) & (losers['loss_pct'] < 0.02)).sum()),
            "over_2pct": int((losers['loss_pct'] >= 0.02).sum())
        }
    }
    
    # Check if losses are hitting stop-loss threshold (1.5%)
    stop_loss_threshold = 0.015
    losses_at_stop = (losers['loss_pct'] >= stop_loss_threshold * 0.95) & (losers['loss_pct'] <= stop_loss_threshold * 1.05)
    loss_stats["hitting_stop_loss"] = {
        "count": int(losses_at_stop.sum()),
        "percentage": float(losses_at_stop.sum() / len(losers) * 100) if len(losers) > 0 else 0.0
    }
    
    return loss_stats


def analyze_quality_filters(trades: pd.DataFrame) -> Dict:
    """Analyze quality filter effectiveness"""
    if len(trades) == 0:
        return {}
    
    winners = trades[trades['is_win']].copy()
    losers = trades[~trades['is_win']].copy()
    
    if len(winners) == 0 or len(losers) == 0:
        return {"error": "Insufficient data for comparison"}
    
    # Analyze confidence levels
    confidence_stats = {
        "winners": {
            "avg_confidence": float(winners['strategy_confidence'].mean()) if 'strategy_confidence' in winners.columns else None,
            "median_confidence": float(winners['strategy_confidence'].median()) if 'strategy_confidence' in winners.columns else None,
            "min_confidence": float(winners['strategy_confidence'].min()) if 'strategy_confidence' in winners.columns else None,
            "max_confidence": float(winners['strategy_confidence'].max()) if 'strategy_confidence' in winners.columns else None
        },
        "losers": {
            "avg_confidence": float(losers['strategy_confidence'].mean()) if 'strategy_confidence' in losers.columns else None,
            "median_confidence": float(losers['strategy_confidence'].median()) if 'strategy_confidence' in losers.columns else None,
            "min_confidence": float(losers['strategy_confidence'].min()) if 'strategy_confidence' in losers.columns else None,
            "max_confidence": float(losers['strategy_confidence'].max()) if 'strategy_confidence' in losers.columns else None
        }
    }
    
    # Check if low-confidence trades are losing more
    if 'strategy_confidence' in trades.columns:
        low_conf_threshold = 0.3
        high_conf_threshold = 0.7
        
        low_conf_trades = trades[trades['strategy_confidence'] < low_conf_threshold]
        high_conf_trades = trades[trades['strategy_confidence'] >= high_conf_threshold]
        
        confidence_stats["low_confidence_trades"] = {
            "count": len(low_conf_trades),
            "win_rate": float(low_conf_trades['is_win'].mean()) if len(low_conf_trades) > 0 else 0.0,
            "avg_pnl": float(low_conf_trades['net_pnl'].mean()) if len(low_conf_trades) > 0 else 0.0
        }
        
        confidence_stats["high_confidence_trades"] = {
            "count": len(high_conf_trades),
            "win_rate": float(high_conf_trades['is_win'].mean()) if len(high_conf_trades) > 0 else 0.0,
            "avg_pnl": float(high_conf_trades['net_pnl'].mean()) if len(high_conf_trades) > 0 else 0.0
        }
    
    return confidence_stats


def compare_winners_vs_losers(trades: pd.DataFrame) -> Dict:
    """Compare winners vs losers across multiple dimensions"""
    if len(trades) == 0:
        return {}
    
    winners = trades[trades['is_win']].copy()
    losers = trades[~trades['is_win']].copy()
    
    if len(winners) == 0 or len(losers) == 0:
        return {"error": "Insufficient data for comparison"}
    
    comparison = {
        "count": {
            "winners": len(winners),
            "losers": len(losers),
            "total": len(trades),
            "win_rate": float(len(winners) / len(trades))
        },
        "pnl": {
            "winners_avg": float(winners['net_pnl'].mean()),
            "losers_avg": float(losers['net_pnl'].mean()),
            "winners_total": float(winners['net_pnl'].sum()),
            "losers_total": float(losers['net_pnl'].sum()),
            "profit_factor": float(abs(winners['net_pnl'].sum()) / abs(losers['net_pnl'].sum())) if losers['net_pnl'].sum() != 0 else 0.0
        },
        "duration": {
            "winners_avg_steps": float(winners['duration_steps'].mean()) if 'duration_steps' in winners.columns else None,
            "losers_avg_steps": float(losers['duration_steps'].mean()) if 'duration_steps' in losers.columns else None
        },
        "position_size": {
            "winners_avg": float(winners['position_size'].abs().mean()),
            "losers_avg": float(losers['position_size'].abs().mean())
        },
        "strategy": {}
    }
    
    # Analyze by strategy
    if 'strategy' in trades.columns:
        strategy_stats = trades.groupby('strategy').agg({
            'is_win': ['count', 'sum', 'mean'],
            'net_pnl': 'mean'
        }).round(4)
        strategy_stats.columns = ['total_trades', 'wins', 'win_rate', 'avg_pnl']
        comparison['strategy'] = strategy_stats.to_dict('index')
    
    return comparison


def identify_common_patterns(trades: pd.DataFrame) -> Dict:
    """Identify common patterns in losing trades"""
    if len(trades) == 0:
        return {}
    
    losers = trades[~trades['is_win']].copy()
    
    if len(losers) == 0:
        return {"error": "No losing trades to analyze"}
    
    patterns = {
        "most_common_strategies": {},
        "confidence_distribution": {},
        "loss_magnitude_patterns": {}
    }
    
    # Most common strategies in losers
    if 'strategy' in losers.columns:
        strategy_counts = losers['strategy'].value_counts()
        patterns["most_common_strategies"] = strategy_counts.to_dict()
    
    # Confidence distribution in losers
    if 'strategy_confidence' in losers.columns:
        patterns["confidence_distribution"] = {
            "avg": float(losers['strategy_confidence'].mean()),
            "median": float(losers['strategy_confidence'].median()),
            "std": float(losers['strategy_confidence'].std()),
            "min": float(losers['strategy_confidence'].min()),
            "max": float(losers['strategy_confidence'].max())
        }
    
    # Loss magnitude patterns
    losers['loss_pct'] = losers['pnl_pct'].abs()
    patterns["loss_magnitude_patterns"] = {
        "small_losses_under_1pct": int((losers['loss_pct'] < 0.01).sum()),
        "medium_losses_1_to_1.5pct": int(((losers['loss_pct'] >= 0.01) & (losers['loss_pct'] < 0.015)).sum()),
        "large_losses_over_1.5pct": int((losers['loss_pct'] >= 0.015).sum())
    }
    
    return patterns


def print_analysis_report(trades: pd.DataFrame, analysis: Dict):
    """Print comprehensive analysis report"""
    print("=" * 80)
    print("LOSING TRADES ANALYSIS REPORT")
    print("=" * 80)
    print(f"\n📊 Dataset: {len(trades)} total trades")
    print(f"   Winners: {len(trades[trades['is_win']])} ({len(trades[trades['is_win']])/len(trades)*100:.1f}%)")
    print(f"   Losers: {len(trades[~trades['is_win']])} ({len(trades[~trades['is_win']])/len(trades)*100:.1f}%)")
    
    # Entry Timing Analysis
    if "entry_timing" in analysis and "error" not in analysis["entry_timing"]:
        print("\n" + "=" * 80)
        print("⏰ ENTRY TIMING ANALYSIS")
        print("=" * 80)
        timing = analysis["entry_timing"]
        if "worst_hours" in timing:
            print("\n⚠️  Worst Trading Hours (by win rate):")
            for hour, stats in list(timing["worst_hours"].items())[:5]:
                print(f"   Hour {int(hour):02d}:00 - Win Rate: {stats['win_rate']*100:.1f}% ({stats['wins']}/{stats['total_trades']} trades)")
        
        if "best_hours" in timing:
            print("\n✅ Best Trading Hours (by win rate):")
            for hour, stats in list(timing["best_hours"].items())[:5]:
                print(f"   Hour {int(hour):02d}:00 - Win Rate: {stats['win_rate']*100:.1f}% ({stats['wins']}/{stats['total_trades']} trades)")
    
    # Stop-Loss Analysis
    if "stop_loss" in analysis and "error" not in analysis["stop_loss"]:
        print("\n" + "=" * 80)
        print("🛑 STOP-LOSS ANALYSIS")
        print("=" * 80)
        sl = analysis["stop_loss"]
        print(f"\n   Average Loss: {sl.get('avg_loss_pct', 0)*100:.2f}%")
        print(f"   Median Loss: {sl.get('median_loss_pct', 0)*100:.2f}%")
        print(f"   Max Loss: {sl.get('max_loss_pct', 0)*100:.2f}%")
        
        if "loss_distribution" in sl:
            dist = sl["loss_distribution"]
            print(f"\n   Loss Distribution:")
            print(f"      < 1.0%: {dist.get('under_1pct', 0)} trades")
            print(f"      1.0-1.5%: {dist.get('1_to_1.5pct', 0)} trades")
            print(f"      1.5-2.0%: {dist.get('1.5_to_2pct', 0)} trades")
            print(f"      > 2.0%: {dist.get('over_2pct', 0)} trades")
        
        if "hitting_stop_loss" in sl:
            hit = sl["hitting_stop_loss"]
            print(f"\n   Stop-Loss Hits (1.5% threshold):")
            print(f"      {hit.get('count', 0)} trades ({hit.get('percentage', 0):.1f}% of losers)")
    
    # Quality Filters Analysis
    if "quality_filters" in analysis and "error" not in analysis["quality_filters"]:
        print("\n" + "=" * 80)
        print("🎯 QUALITY FILTERS ANALYSIS")
        print("=" * 80)
        qf = analysis["quality_filters"]
        
        if "winners" in qf and qf["winners"].get("avg_confidence") is not None:
            print(f"\n   Winners:")
            print(f"      Avg Confidence: {qf['winners']['avg_confidence']:.3f}")
            print(f"      Median Confidence: {qf['winners']['median_confidence']:.3f}")
        
        if "losers" in qf and qf["losers"].get("avg_confidence") is not None:
            print(f"\n   Losers:")
            print(f"      Avg Confidence: {qf['losers']['avg_confidence']:.3f}")
            print(f"      Median Confidence: {qf['losers']['median_confidence']:.3f}")
        
        if "low_confidence_trades" in qf:
            lc = qf["low_confidence_trades"]
            print(f"\n   Low Confidence Trades (<0.3):")
            print(f"      Count: {lc.get('count', 0)}")
            print(f"      Win Rate: {lc.get('win_rate', 0)*100:.1f}%")
            print(f"      Avg PnL: ${lc.get('avg_pnl', 0):.2f}")
        
        if "high_confidence_trades" in qf:
            hc = qf["high_confidence_trades"]
            print(f"\n   High Confidence Trades (≥0.7):")
            print(f"      Count: {hc.get('count', 0)}")
            print(f"      Win Rate: {hc.get('win_rate', 0)*100:.1f}%")
            print(f"      Avg PnL: ${hc.get('avg_pnl', 0):.2f}")
    
    # Winners vs Losers Comparison
    if "comparison" in analysis and "error" not in analysis["comparison"]:
        print("\n" + "=" * 80)
        print("📈 WINNERS VS LOSERS COMPARISON")
        print("=" * 80)
        comp = analysis["comparison"]
        
        if "count" in comp:
            print(f"\n   Win Rate: {comp['count']['win_rate']*100:.1f}%")
            print(f"   Winners: {comp['count']['winners']}, Losers: {comp['count']['losers']}")
        
        if "pnl" in comp:
            print(f"\n   PnL Analysis:")
            print(f"      Avg Winner: ${comp['pnl']['winners_avg']:.2f}")
            print(f"      Avg Loser: ${comp['pnl']['losers_avg']:.2f}")
            print(f"      Total Winners: ${comp['pnl']['winners_total']:.2f}")
            print(f"      Total Losers: ${comp['pnl']['losers_total']:.2f}")
            print(f"      Profit Factor: {comp['pnl']['profit_factor']:.2f}")
        
        if "strategy" in comp and comp["strategy"]:
            print(f"\n   Strategy Performance:")
            for strategy, stats in comp["strategy"].items():
                print(f"      {strategy}: {stats['wins']}/{stats['total_trades']} wins ({stats['win_rate']*100:.1f}% win rate, ${stats['avg_pnl']:.2f} avg)")
    
    # Common Patterns
    if "patterns" in analysis and "error" not in analysis["patterns"]:
        print("\n" + "=" * 80)
        print("🔍 COMMON PATTERNS IN LOSING TRADES")
        print("=" * 80)
        patterns = analysis["patterns"]
        
        if "most_common_strategies" in patterns:
            print(f"\n   Most Common Strategies in Losers:")
            for strategy, count in sorted(patterns["most_common_strategies"].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"      {strategy}: {count} trades")
        
        if "loss_magnitude_patterns" in patterns:
            lmp = patterns["loss_magnitude_patterns"]
            print(f"\n   Loss Magnitude Patterns:")
            print(f"      Small losses (<1%): {lmp.get('small_losses_under_1pct', 0)}")
            print(f"      Medium losses (1-1.5%): {lmp.get('medium_losses_1_to_1.5pct', 0)}")
            print(f"      Large losses (>1.5%): {lmp.get('large_losses_over_1.5pct', 0)}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    recommendations = []
    
    if "stop_loss" in analysis and "error" not in analysis["stop_loss"]:
        sl = analysis["stop_loss"]
        avg_loss = sl.get("avg_loss_pct", 0)
        if avg_loss > 0.02:
            recommendations.append("⚠️  Average loss is >2% - consider tightening stop-loss or improving entry timing")
        elif avg_loss < 0.01:
            recommendations.append("✅ Average loss is <1% - stop-loss appears effective")
    
    if "quality_filters" in analysis and "error" not in analysis["quality_filters"]:
        qf = analysis["quality_filters"]
        if "low_confidence_trades" in qf:
            lc = qf["low_confidence_trades"]
            if lc.get("win_rate", 0) < 0.4:
                recommendations.append("⚠️  Low confidence trades have poor win rate - consider increasing min_action_confidence threshold")
    
    if "comparison" in analysis and "error" not in analysis["comparison"]:
        comp = analysis["comparison"]
        if comp.get("pnl", {}).get("profit_factor", 0) < 1.2:
            recommendations.append("⚠️  Profit factor <1.2 - need to improve risk/reward ratio or win rate")
    
    if "entry_timing" in analysis and "error" not in analysis["entry_timing"]:
        timing = analysis["entry_timing"]
        if "worst_hours" in timing and len(timing["worst_hours"]) > 0:
            worst_hour = list(timing["worst_hours"].keys())[0]
            recommendations.append(f"⚠️  Consider avoiding trades during hour {int(worst_hour):02d}:00 (lowest win rate)")
    
    if not recommendations:
        recommendations.append("✅ No critical issues identified - continue monitoring")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n   {i}. {rec}")
    
    print("\n" + "=" * 80)


def main():
    """Main analysis function"""
    print("🔍 Analyzing Losing Trades...")
    print(f"📂 Project root: {project_root}")
    
    # Load trades
    trades = load_trades_from_journal()
    
    if len(trades) == 0:
        print("[ERROR] No trades found. Make sure training has been running and trades have been logged.")
        return
    
    print(f"✅ Loaded {len(trades)} trades from journal")
    
    # Perform analyses
    analysis = {
        "entry_timing": analyze_entry_timing(trades),
        "stop_loss": analyze_stop_loss_patterns(trades),
        "quality_filters": analyze_quality_filters(trades),
        "comparison": compare_winners_vs_losers(trades),
        "patterns": identify_common_patterns(trades)
    }
    
    # Print report
    print_analysis_report(trades, analysis)
    
    # Save to file
    output_file = project_root / "reports" / "losing_trades_analysis.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_analysis = convert_to_serializable(analysis)
    
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_trades": len(trades),
            "winners": int(trades['is_win'].sum()),
            "losers": int((~trades['is_win']).sum()),
            "win_rate": float(trades['is_win'].mean()),
            "analysis": serializable_analysis
        }, f, indent=2)
    
    print(f"\n💾 Analysis saved to: {output_file}")


if __name__ == "__main__":
    main()

