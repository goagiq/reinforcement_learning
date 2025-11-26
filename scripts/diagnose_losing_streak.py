"""
Diagnostic Script for Losing Streak Analysis

Checks:
1. Checkpoint loading status
2. Adaptive training parameters
3. Recent trade analysis
4. Quality filter status
5. System configuration
"""

import sys
from pathlib import Path
import json
import torch
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

# Add project root to path
try:
    project_root = Path(__file__).parent.parent
except NameError:
    # If __file__ not available (e.g., exec()), use current working directory
    project_root = Path.cwd()
sys.path.insert(0, str(project_root))

def check_checkpoints():
    """Check available checkpoints and find latest"""
    print("=" * 80)
    print("[CHECKPOINT] CHECKPOINT ANALYSIS")
    print("=" * 80)
    
    models_dir = project_root / "models"
    if not models_dir.exists():
        print("[ERROR] Models directory not found!")
        return None
    
    # Find all checkpoints
    checkpoints = sorted(models_dir.glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    
    if not checkpoints:
        print("[ERROR] No checkpoints found!")
        return None
    
    latest = checkpoints[-1]
    print(f"[OK] Found {len(checkpoints)} checkpoints")
    print(f"[LATEST] Latest checkpoint: {latest.name}")
    print(f"   Timestep: {latest.stem.split('_')[1]}")
    print(f"   Size: {latest.stat().st_size / (1024*1024):.2f} MB")
    print(f"   Modified: {datetime.fromtimestamp(latest.stat().st_mtime)}")
    
    # Check checkpoint contents
    try:
        checkpoint = torch.load(str(latest), map_location='cpu', weights_only=False)
        print(f"\n[INFO] Checkpoint Contents:")
        print(f"   Keys: {list(checkpoint.keys())}")
        
        if "timestep" in checkpoint:
            print(f"   Saved timestep: {checkpoint['timestep']:,}")
        if "episode" in checkpoint:
            print(f"   Saved episode: {checkpoint['episode']}")
        if "hidden_dims" in checkpoint:
            print(f"   Architecture: {checkpoint['hidden_dims']}")
        if "state_dim" in checkpoint:
            print(f"   State dim: {checkpoint['state_dim']}")
            
    except Exception as e:
        print(f"[WARN] Could not read checkpoint: {e}")
    
    return latest

def check_adaptive_config():
    """Check adaptive training configuration"""
    print("\n" + "=" * 80)
    print("[CONFIG] ADAPTIVE TRAINING CONFIG")
    print("=" * 80)
    
    config_path = project_root / "logs" / "adaptive_training" / "current_reward_config.json"
    
    if not config_path.exists():
        print("[ERROR] Adaptive config not found!")
        print(f"   Expected: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"[OK] Adaptive config found")
        print(f"   Modified: {datetime.fromtimestamp(config_path.stat().st_mtime)}")
        print(f"\n[INFO] Current Parameters:")
        
        # Quality filters
        quality = config.get("quality_filters", {})
        print(f"\n   Quality Filters:")
        print(f"      min_action_confidence: {quality.get('min_action_confidence', 'N/A')}")
        print(f"      min_quality_score: {quality.get('min_quality_score', 'N/A')}")
        print(f"      min_risk_reward_ratio: {quality.get('min_risk_reward_ratio', 'N/A')}")
        
        # Entropy
        entropy = config.get("entropy_coef", "N/A")
        print(f"\n   Exploration:")
        print(f"      entropy_coef: {entropy}")
        
        # Learning rate
        lr = config.get("learning_rate", "N/A")
        print(f"\n   Learning:")
        print(f"      learning_rate: {lr}")
        
        return config
        
    except Exception as e:
        print(f"⚠️  Could not read adaptive config: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_trading_journal():
    """Check recent trades from trading journal"""
    print("\n" + "=" * 80)
    print("[TRADES] RECENT TRADES ANALYSIS")
    print("=" * 80)
    
    journal_path = project_root / "logs" / "trading_journal.db"
    
    if not journal_path.exists():
        print("[ERROR] Trading journal not found!")
        print(f"   Expected: {journal_path}")
        return None
    
    try:
        import sqlite3
        conn = sqlite3.connect(str(journal_path))
        
        # Get recent trades
        query = """
        SELECT 
            timestamp, episode, strategy, 
            entry_price, exit_price, pnl, net_pnl,
            strategy_confidence, is_win
        FROM trades
        ORDER BY timestamp DESC
        LIMIT 50
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            print("[ERROR] No trades found in journal!")
            return None
        
        print(f"[OK] Found {len(df)} recent trades")
        
        # Analyze
        recent_trades = df.head(20)
        print(f"\n[STATS] Last 20 Trades:")
        print(f"   Wins: {len(recent_trades[recent_trades['pnl'] > 0])}")
        print(f"   Losses: {len(recent_trades[recent_trades['pnl'] < 0])}")
        print(f"   Win Rate: {len(recent_trades[recent_trades['pnl'] > 0]) / len(recent_trades) * 100:.1f}%")
        print(f"   Total PnL: ${recent_trades['pnl'].sum():.2f}")
        print(f"   Avg PnL: ${recent_trades['pnl'].mean():.2f}")
        
        # Confidence
        if 'strategy_confidence' in recent_trades.columns:
            avg_confidence = recent_trades['strategy_confidence'].mean()
            print(f"\n   Confidence Metrics:")
            print(f"      Avg Confidence: {avg_confidence:.3f}")
            print(f"      Min Confidence: {recent_trades['strategy_confidence'].min():.3f}")
            print(f"      Max Confidence: {recent_trades['strategy_confidence'].max():.3f}")
        
        # Check for patterns
        print(f"\n[PATTERNS]")
        losing_trades = recent_trades[recent_trades['pnl'] < 0]
        if len(losing_trades) > 0:
            print(f"   Losing trades:")
            print(f"      Avg Loss: ${losing_trades['pnl'].mean():.2f}")
            if 'strategy_confidence' in losing_trades.columns:
                print(f"      Avg Confidence (losers): {losing_trades['strategy_confidence'].mean():.3f}")
            else:
                print(f"      Avg Confidence (losers): N/A")
        
        return df
        
    except Exception as e:
        print(f"[WARN] Could not read trading journal: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_config_files():
    """Check training configuration"""
    print("\n" + "=" * 80)
    print("[CONFIG] TRAINING CONFIGURATION")
    print("=" * 80)
    
    config_paths = [
        project_root / "configs" / "train_config_full.yaml",
        project_root / "configs" / "train_config_adaptive.yaml",
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            print(f"[OK] Found: {config_path.name}")
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check decision gate
                dg = config.get("decision_gate", {})
                print(f"\n   Decision Gate:")
                print(f"      min_confluence_required: {dg.get('min_confluence_required', 'N/A')}")
                print(f"      min_combined_confidence: {dg.get('min_combined_confidence', 'N/A')}")
                
                # Check quality scorer
                qs = config.get("decision_gate", {}).get("quality_scorer", {})
                print(f"\n   Quality Scorer:")
                print(f"      enabled: {qs.get('enabled', 'N/A')}")
                
                break
            except Exception as e:
                print(f"⚠️  Could not read config: {e}")
        else:
            print(f"❌ Not found: {config_path.name}")

def check_training_logs():
    """Check recent training logs for errors"""
    print("\n" + "=" * 80)
    print("[LOGS] RECENT TRAINING LOGS")
    print("=" * 80)
    
    logs_dir = project_root / "logs"
    if not logs_dir.exists():
        print("[ERROR] Logs directory not found!")
        return
    
    # Find latest log file
    log_files = sorted(logs_dir.glob("training_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not log_files:
        print("[ERROR] No training logs found!")
        return
    
    latest_log = log_files[0]
    print(f"[OK] Latest log: {latest_log.name}")
    print(f"   Modified: {datetime.fromtimestamp(latest_log.stat().st_mtime)}")
    
    # Read last 50 lines
    try:
        with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            last_lines = lines[-50:] if len(lines) > 50 else lines
            
            print(f"\n[LOGS] Last 50 lines:")
            print("   " + "=" * 76)
            for line in last_lines:
                # Look for errors or warnings
                if "ERROR" in line or "WARN" in line or "CRITICAL" in line:
                    print(f"   [WARN] {line.strip()}")
                elif "Resuming from checkpoint" in line or "Resume:" in line:
                    print(f"   [CHECKPOINT] {line.strip()}")
                elif "Episode" in line and ("PnL" in line or "Reward" in line):
                    print(f"   [EPISODE] {line.strip()}")
            
    except Exception as e:
        print(f"⚠️  Could not read log: {e}")

def generate_report():
    """Generate diagnostic report"""
    print("\n" + "=" * 80)
    print("[SUMMARY] DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    results = {
        "checkpoint": check_checkpoints(),
        "adaptive_config": check_adaptive_config(),
        "recent_trades": check_trading_journal(),
    }
    
    check_config_files()
    check_training_logs()
    
    print("\n" + "=" * 80)
    print("[RECOMMENDATIONS]")
    print("=" * 80)
    
    # Generate recommendations based on findings
    recommendations = []
    
    if results["checkpoint"] is None:
        recommendations.append("[ERROR] No checkpoint found - training may have started fresh")
        recommendations.append("   -> Check if checkpoint_path was specified correctly")
    
    if results["adaptive_config"] is None:
        recommendations.append("[ERROR] Adaptive config not found - quality filters may be disabled")
        recommendations.append("   -> Adaptive training may not be active")
    
    if results["recent_trades"] is not None:
        df = results["recent_trades"]
        if len(df) > 0:
            recent = df.head(20)
            win_rate = len(recent[recent['pnl'] > 0]) / len(recent) if len(recent) > 0 else 0
            avg_pnl = recent['pnl'].mean()
            
            if win_rate < 0.5:
                recommendations.append(f"[WARN] Low win rate: {win_rate*100:.1f}% (last 20 trades)")
                recommendations.append("   -> Consider tightening quality filters")
            
            if avg_pnl < 0:
                recommendations.append(f"[WARN] Negative average PnL: ${avg_pnl:.2f}")
                recommendations.append("   -> Recent trades are losing money")
            
            if 'strategy_confidence' in recent.columns:
                avg_confidence = recent['strategy_confidence'].mean()
                if avg_confidence < 0.5:
                    recommendations.append(f"[WARN] Low average confidence: {avg_confidence:.3f}")
                    recommendations.append("   -> Trades may have low confidence")
    
    if not recommendations:
        recommendations.append("[OK] No obvious issues found")
        recommendations.append("   -> Check training logs for more details")
        recommendations.append("   -> Verify checkpoint was loaded correctly")
    
    for rec in recommendations:
        print(f"   {rec}")
    
    print("\n" + "=" * 80)
    print("[OK] Diagnostic Complete")
    print("=" * 80)

if __name__ == "__main__":
    generate_report()

