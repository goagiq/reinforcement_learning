"""
Monitor Trading Journal Updates in Real-Time

This script watches the trading journal database and reports when new trades
are added, showing that the journal is being updated in real-time during training.
"""
import sqlite3
import time
import sys
from pathlib import Path

def monitor_journal(db_path, check_interval=2.0):
    """Monitor journal for new trades"""
    print("Monitoring Trading Journal for real-time updates...")
    print(f"Database: {db_path}")
    print(f"Check interval: {check_interval} seconds")
    print("=" * 60)
    
    if not db_path.exists():
        print(f"[ERROR] Journal database not found: {db_path}")
        print("   The journal will be created when the first trade is logged.")
        return
    
    last_trade_count = 0
    last_timestamp = None
    check_count = 0
    
    try:
        while True:
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Get current trade count and latest trade
                cursor.execute("SELECT COUNT(*) FROM trades")
                current_count = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT timestamp, episode, pnl, net_pnl, is_win 
                    FROM trades 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """)
                latest_trade = cursor.fetchone()
                
                conn.close()
                
                check_count += 1
                
                # Check if new trades were added
                if current_count > last_trade_count:
                    new_trades = current_count - last_trade_count
                    print(f"\n[UPDATE #{check_count}] {time.strftime('%H:%M:%S')}")
                    print(f"   New trades detected: +{new_trades} (Total: {current_count})")
                    
                    if latest_trade:
                        timestamp, episode, pnl, net_pnl, is_win = latest_trade
                        win_str = "WIN" if is_win else "LOSS"
                        print(f"   Latest trade: Episode {episode}, PnL: ${net_pnl:.2f} ({win_str})")
                        print(f"   Timestamp: {timestamp}")
                    
                    last_trade_count = current_count
                    last_timestamp = latest_trade[0] if latest_trade else None
                elif current_count == 0:
                    if check_count == 1:
                        print(f"[INFO] Journal exists but no trades yet (check #{check_count})")
                    elif check_count % 10 == 0:  # Print every 10 checks if no trades
                        print(f"[INFO] Still no trades (check #{check_count})")
                else:
                    # No new trades, but show status periodically
                    if check_count % 15 == 0:  # Print every 15 checks
                        print(f"[INFO] Check #{check_count}: {current_count} trades (no new trades)")
                
                time.sleep(check_interval)
                
            except sqlite3.OperationalError as e:
                if "no such table" in str(e).lower():
                    print(f"[INFO] Trades table doesn't exist yet - waiting for first trade...")
                    time.sleep(check_interval)
                else:
                    print(f"[ERROR] Database error: {e}")
                    time.sleep(check_interval)
            except KeyboardInterrupt:
                print(f"\n\n[STOPPED] Monitoring stopped by user")
                print(f"   Final trade count: {current_count}")
                break
            except Exception as e:
                print(f"[ERROR] Unexpected error: {e}")
                time.sleep(check_interval)
                
    except KeyboardInterrupt:
        print(f"\n\n[STOPPED] Monitoring stopped")
        sys.exit(0)

if __name__ == "__main__":
    # Get project root
    project_root = Path(__file__).parent.parent
    db_path = project_root / "logs" / "trading_journal.db"
    
    # Allow custom check interval
    check_interval = 2.0
    if len(sys.argv) > 1:
        try:
            check_interval = float(sys.argv[1])
        except ValueError:
            print(f"[WARN] Invalid interval '{sys.argv[1]}', using default 2.0 seconds")
    
    monitor_journal(db_path, check_interval)

