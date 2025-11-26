"""
Verify Performance Endpoint Updates

This script checks if the /api/monitoring/performance endpoint
is returning real-time data from the journal (not cached trainer data).
"""
import requests
import time
import sys
from datetime import datetime

def verify_endpoint(base_url="http://localhost:8200", check_interval=5.0, max_checks=12):
    """Verify performance endpoint is updating"""
    print("Verifying Performance Endpoint Real-Time Updates")
    print("=" * 60)
    print(f"Will check {max_checks} times (every {check_interval} seconds)")
    print("Press Ctrl+C to stop early\n")
    
    last_trade_count = None
    last_pnl = None
    last_timestamp = None
    check_count = 0
    
    try:
        while check_count < max_checks:
            try:
                response = requests.get(f"{base_url}/api/monitoring/performance", timeout=5)
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") != "success":
                    print(f"[ERROR] API returned error: {data.get('message', 'Unknown')}")
                    time.sleep(check_interval)
                    continue
                
                metrics = data.get("metrics", {})
                source = metrics.get("source", "unknown")
                trade_count = metrics.get("total_trades", 0)
                total_pnl = metrics.get("total_pnl", 0.0)
                current_time = datetime.now().strftime("%H:%M:%S")
                
                check_count += 1
                
                # Check if data is updating
                if last_trade_count is not None:
                    trades_changed = trade_count != last_trade_count
                    pnl_changed = abs(total_pnl - last_pnl) > 0.01
                    
                    if trades_changed or pnl_changed:
                        print(f"\n[UPDATE #{check_count}] {current_time}")
                        print(f"   Source: {source}")
                        if trades_changed:
                            print(f"   Trades: {last_trade_count} -> {trade_count} (+{trade_count - last_trade_count})")
                        if pnl_changed:
                            print(f"   PnL: ${last_pnl:.2f} -> ${total_pnl:.2f} (${total_pnl - last_pnl:+.2f})")
                        print(f"   Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
                        print(f"   Profit Factor: {metrics.get('profit_factor', 0):.2f}")
                    elif check_count % 6 == 0:  # Print status every 30 seconds (6 * 5s)
                        print(f"[INFO] Check #{check_count} ({current_time}): {trade_count} trades, PnL: ${total_pnl:.2f} (no changes)")
                else:
                    # First check
                    print(f"[INIT] Check #{check_count} ({current_time})")
                    print(f"   Source: {source}")
                    print(f"   Trades: {trade_count}")
                    print(f"   Total PnL: ${total_pnl:.2f}")
                    print(f"   Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
                    
                    if source == "journal":
                        print(f"   [OK] Endpoint is reading from journal (real-time)")
                    elif source == "training":
                        print(f"   [WARN] Endpoint is reading from trainer (may be stale)")
                        print(f"   [INFO] Backend needs restart to use journal-based endpoint")
                    else:
                        print(f"   [WARN] Unknown source: {source}")
                
                last_trade_count = trade_count
                last_pnl = total_pnl
                last_timestamp = current_time
                
                time.sleep(check_interval)
                
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Failed to connect to API: {e}")
                print(f"   Make sure backend server is running on {base_url}")
                time.sleep(check_interval)
            except KeyboardInterrupt:
                print(f"\n\n[STOPPED] Monitoring stopped by user")
                if last_trade_count is not None:
                    print(f"   Final state: {last_trade_count} trades, PnL: ${last_pnl:.2f}")
                break
            except Exception as e:
                print(f"[ERROR] Unexpected error: {e}")
                time.sleep(check_interval)
                
    except KeyboardInterrupt:
        print(f"\n\n[STOPPED] Monitoring stopped")
        sys.exit(0)

if __name__ == "__main__":
    # Allow custom check interval and max checks
    check_interval = 5.0
    max_checks = 12  # Default: 12 checks = 60 seconds
    
    if len(sys.argv) > 1:
        try:
            check_interval = float(sys.argv[1])
        except ValueError:
            print(f"[WARN] Invalid interval '{sys.argv[1]}', using default 5.0 seconds")
    
    if len(sys.argv) > 2:
        try:
            max_checks = int(sys.argv[2])
        except ValueError:
            print(f"[WARN] Invalid max_checks '{sys.argv[2]}', using default 12")
    
    verify_endpoint(check_interval=check_interval, max_checks=max_checks)

