"""
Comprehensive Trade Performance and Training Progress Analysis
"""

import sqlite3
from pathlib import Path
from src.utils.colors import success, info, warn, error

def analyze_performance():
    """Analyze trade performance and training progress"""
    
    print(info("\n" + "="*70))
    print(info("COMPREHENSIVE PERFORMANCE ANALYSIS"))
    print(info("="*70))
    
    db_path = Path("logs/trading_journal.db")
    
    if not db_path.exists():
        print(error("[ERROR] Trading journal database not found"))
        return
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        # Get database schema
        cursor.execute("PRAGMA table_info(trades)")
        trade_cols = [col[1] for col in cursor.fetchall()]
        cursor.execute("PRAGMA table_info(episodes)")
        episode_cols = [col[1] for col in cursor.fetchall()]
        
        print(info(f"\n[SCHEMA] Trades columns: {', '.join(trade_cols)}"))
        print(info(f"[SCHEMA] Episodes columns: {', '.join(episode_cols)}"))
        
        # Total trades
        cursor.execute("SELECT COUNT(*) FROM trades")
        total_trades = cursor.fetchone()[0]
        print(success(f"\n[TRADES] Total trades: {total_trades:,}"))
        
        if total_trades == 0:
            print(warn("[WARN] No trades found"))
            conn.close()
            return
        
        # Win rate
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                COUNT(*) as total
            FROM trades 
            WHERE pnl IS NOT NULL
        """)
        result = cursor.fetchone()
        wins, losses, total = result
        
        if total > 0:
            win_rate = (wins / total) * 100
            print(success(f"\n[WIN RATE]"))
            print(info(f"  Wins: {wins:,} ({win_rate:.1f}%)"))
            print(info(f"  Losses: {losses:,} ({100-win_rate:.1f}%)"))
            print(info(f"  Total: {total:,}"))
        
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
        if total_pnl:
            pnl_color = success if total_pnl > 0 else error
            print(pnl_color(f"  Total PnL: ${total_pnl:,.2f}"))
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
        
        # Recent trades
        if 'timestamp' in trade_cols:
            cursor.execute("""
                SELECT timestamp, entry_price, exit_price, pnl, position_size
                FROM trades 
                ORDER BY timestamp DESC 
                LIMIT 10
            """)
            recent_trades = cursor.fetchall()
            
            if recent_trades:
                print(success(f"\n[RECENT TRADES] (Last 10)"))
                for trade in recent_trades:
                    timestamp, entry, exit, pnl, size = trade[:5]
                    pnl_str = f"${pnl:.2f}" if pnl else "$0.00"
                    pnl_color = success if (pnl and pnl > 0) else error if (pnl and pnl < 0) else info
                    print(pnl_color(f"  {timestamp}: Entry=${entry:.2f}, Exit=${exit:.2f}, PnL={pnl_str}, Size={size:.2f}"))
        
        # Episode statistics
        if 'episode_number' in episode_cols:
            cursor.execute("SELECT COUNT(*) FROM episodes")
            total_episodes = cursor.fetchone()[0]
            print(success(f"\n[EPISODES] Total episodes: {total_episodes:,}"))
            
            if 'total_pnl' in episode_cols:
                cursor.execute("""
                    SELECT 
                        AVG(total_pnl) as avg_episode_pnl,
                        SUM(total_pnl) as total_episode_pnl
                    FROM episodes
                    WHERE total_pnl IS NOT NULL
                """)
                episode_stats = cursor.fetchone()
                if episode_stats[0]:
                    print(info(f"  Average Episode PnL: ${episode_stats[0]:.2f}"))
                    print(info(f"  Total Episode PnL: ${episode_stats[1]:.2f}"))
        
        # Training progress from checkpoints
        model_dir = Path("models")
        if model_dir.exists():
            checkpoints = sorted([cp for cp in model_dir.glob("checkpoint_*.pt")], 
                               key=lambda x: int(x.stem.split('_')[1]))
            if checkpoints:
                latest_checkpoint = checkpoints[-1]
                latest_timestep = int(latest_checkpoint.stem.split('_')[1])
                print(success(f"\n[TRAINING PROGRESS]"))
                print(info(f"  Checkpoints: {len(checkpoints)}"))
                print(info(f"  Latest checkpoint: {latest_checkpoint.name}"))
                print(info(f"  Latest timestep: {latest_timestep:,}"))
                print(info(f"  Progress: {latest_timestep/20000000*100:.2f}% (of 20M target)"))
        
    except Exception as e:
        print(error(f"[ERROR] Failed to analyze: {e}"))
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
    
    print(info("\n" + "="*70))


if __name__ == "__main__":
    analyze_performance()

