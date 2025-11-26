"""
Analyze Trade Performance and Training Progress

Comprehensive analysis of trading performance, training metrics, and progress.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from src.utils.colors import success, info, warn, error


def analyze_trade_performance():
    """Analyze trade performance from trading journal database"""
    
    print(info("\n" + "="*70))
    print(info("TRADE PERFORMANCE ANALYSIS"))
    print(info("="*70))
    
    db_path = Path("logs/trading_journal.db")
    
    if not db_path.exists():
        print(error("[ERROR] Trading journal database not found"))
        return
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        # Total trades
        cursor.execute("SELECT COUNT(*) FROM trades")
        total_trades = cursor.fetchone()[0]
        print(info(f"\n[TRADES] Total trades: {total_trades}"))
        
        if total_trades == 0:
            print(warn("[WARN] No trades found in database"))
            conn.close()
            return
        
        # Win rate
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN pnl = 0 THEN 1 ELSE 0 END) as breakeven,
                COUNT(*) as total
            FROM trades 
            WHERE pnl IS NOT NULL
        """)
        result = cursor.fetchone()
        wins, losses, breakeven, total = result
        
        if total > 0:
            win_rate = (wins / total) * 100
            loss_rate = (losses / total) * 100
            print(success(f"\n[WIN RATE]"))
            print(info(f"  Wins: {wins} ({win_rate:.1f}%)"))
            print(info(f"  Losses: {losses} ({loss_rate:.1f}%)"))
            print(info(f"  Breakeven: {breakeven}"))
            print(info(f"  Total: {total}"))
        
        # PnL Statistics
        cursor.execute("""
            SELECT 
                AVG(pnl) as avg_pnl,
                SUM(pnl) as total_pnl,
                MIN(pnl) as min_pnl,
                MAX(pnl) as max_pnl,
                AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss
            FROM trades 
            WHERE pnl IS NOT NULL
        """)
        result = cursor.fetchone()
        avg_pnl, total_pnl, min_pnl, max_pnl, avg_win, avg_loss = result
        
        print(success(f"\n[PnL STATISTICS]"))
        print(info(f"  Total PnL: ${total_pnl:.2f}" if total_pnl else "  Total PnL: $0.00"))
        print(info(f"  Average PnL per trade: ${avg_pnl:.2f}" if avg_pnl else "  Average PnL: $0.00"))
        print(info(f"  Average Win: ${avg_win:.2f}" if avg_win else "  Average Win: $0.00"))
        print(info(f"  Average Loss: ${avg_loss:.2f}" if avg_loss else "  Average Loss: $0.00"))
        print(info(f"  Best Trade: ${max_pnl:.2f}" if max_pnl else "  Best Trade: $0.00"))
        print(info(f"  Worst Trade: ${min_pnl:.2f}" if min_pnl else "  Worst Trade: $0.00"))
        
        # Risk/Reward Ratio
        if avg_win and avg_loss:
            risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            print(success(f"\n[RISK/REWARD]"))
            print(info(f"  Risk/Reward Ratio: {risk_reward:.2f}"))
            if risk_reward >= 1.5:
                print(success(f"  [OK] Good R:R ratio (>= 1.5)"))
            elif risk_reward >= 1.0:
                print(warn(f"  [WARN] Acceptable R:R ratio (>= 1.0)"))
            else:
                print(error(f"  [ERROR] Poor R:R ratio (< 1.0)"))
        
        # Recent trades (last 10)
        cursor.execute("""
            SELECT 
                timestamp,
                entry_price,
                exit_price,
                pnl,
                position_size,
                entry_reason,
                exit_reason
            FROM trades 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        recent_trades = cursor.fetchall()
        
        if recent_trades:
            print(success(f"\n[RECENT TRADES] (Last 10)"))
            for trade in recent_trades:
                timestamp, entry, exit, pnl, size, entry_reason, exit_reason = trade
                pnl_color = success if pnl > 0 else error if pnl < 0 else info
                print(pnl_color(f"  {timestamp}: Entry=${entry:.2f}, Exit=${exit:.2f}, PnL=${pnl:.2f}, Size={size:.2f}"))
        
        # Episode statistics
        cursor.execute("SELECT COUNT(*) FROM episodes")
        total_episodes = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT 
                AVG(total_pnl) as avg_episode_pnl,
                SUM(total_pnl) as total_episode_pnl,
                AVG(trades) as avg_trades_per_episode,
                AVG(win_rate) as avg_win_rate
            FROM episodes
            WHERE total_pnl IS NOT NULL
        """)
        episode_stats = cursor.fetchone()
        
        print(success(f"\n[EPISODE STATISTICS]"))
        print(info(f"  Total Episodes: {total_episodes}"))
        if episode_stats[0]:
            print(info(f"  Average Episode PnL: ${episode_stats[0]:.2f}"))
            print(info(f"  Total Episode PnL: ${episode_stats[1]:.2f}"))
            print(info(f"  Average Trades per Episode: {episode_stats[2]:.1f}"))
            print(info(f"  Average Win Rate: {episode_stats[3]*100:.1f}%"))
        
        # Training progress (from episodes)
        cursor.execute("""
            SELECT 
                episode_number,
                total_pnl,
                trades,
                win_rate,
                timestamp
            FROM episodes
            ORDER BY episode_number DESC
            LIMIT 20
        """)
        recent_episodes = cursor.fetchall()
        
        if recent_episodes:
            print(success(f"\n[RECENT EPISODES] (Last 20)"))
            print(info("  Episode | PnL      | Trades | Win Rate | Timestamp"))
            print(info("  " + "-"*60))
            for ep in recent_episodes:
                ep_num, pnl, trades, win_rate, timestamp = ep
                pnl_str = f"${pnl:.2f}" if pnl else "$0.00"
                win_rate_str = f"{win_rate*100:.1f}%" if win_rate else "0.0%"
                print(info(f"  {ep_num:7d} | {pnl_str:8s} | {trades:6d} | {win_rate_str:8s} | {timestamp}"))
        
    except Exception as e:
        print(error(f"[ERROR] Failed to analyze trade performance: {e}"))
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
    
    print(info("\n" + "="*70))


def analyze_training_progress():
    """Analyze training progress from checkpoints and logs"""
    
    print(info("\n" + "="*70))
    print(info("TRAINING PROGRESS ANALYSIS"))
    print(info("="*70))
    
    # Check for checkpoints
    model_dir = Path("models")
    if model_dir.exists():
        checkpoints = list(model_dir.glob("checkpoint_*.pt"))
        if checkpoints:
            print(success(f"\n[CHECKPOINTS] Found {len(checkpoints)} checkpoint(s)"))
            for cp in sorted(checkpoints, key=lambda x: int(x.stem.split('_')[1])):
                print(info(f"  {cp.name}"))
        else:
            print(warn("[WARN] No checkpoints found"))
    
    # Check for best model
    best_model = model_dir / "best_model.pt"
    if best_model.exists():
        print(success(f"\n[BEST MODEL] {best_model.name} exists"))
    
    # Check for pre-trained weights
    pretrained_model = model_dir / "pretrained_actor.pt"
    if pretrained_model.exists():
        print(success(f"\n[PRETRAINED] Pre-trained weights found: {pretrained_model.name}"))
    
    print(info("\n" + "="*70))


if __name__ == "__main__":
    analyze_trade_performance()
    analyze_training_progress()

