"""
Test script to verify Monitoring tab timestamp filtering
"""
import requests
import json
from datetime import datetime

def test_monitoring_filter():
    base_url = "http://localhost:8200"
    
    print("=" * 80)
    print("TESTING MONITORING TAB TIMESTAMP FILTERING")
    print("=" * 80)
    print()
    
    # 1. Check training status for checkpoint resume timestamp
    print("[1] Checking training status for checkpoint resume timestamp...")
    try:
        response = requests.get(f"{base_url}/api/training/status")
        training_status = response.json()
        checkpoint_timestamp = training_status.get("checkpoint_resume_timestamp")
        
        if checkpoint_timestamp:
            print(f"   [OK] Found checkpoint resume timestamp: {checkpoint_timestamp}")
        else:
            print("   [INFO] No checkpoint resume timestamp found")
            print("   (This is OK if training started fresh, not from checkpoint)")
        print()
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        print()
    
    # 2. Test performance endpoint without filter
    print("[2] Testing /api/monitoring/performance WITHOUT timestamp filter...")
    try:
        response = requests.get(f"{base_url}/api/monitoring/performance")
        data = response.json()
        
        if data.get("status") == "success":
            metrics = data.get("metrics", {})
            total_trades = metrics.get("total_trades", 0)
            total_pnl = metrics.get("total_pnl", 0.0)
            filtered_since = metrics.get("filtered_since")
            
            print(f"   Total trades: {total_trades}")
            print(f"   Total PnL: ${total_pnl:.2f}")
            print(f"   Filtered since: {filtered_since if filtered_since else 'None (showing all trades)'}")
        else:
            print(f"   ✗ Error: {data.get('message', 'Unknown error')}")
        print()
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print()
    
    # 3. Test performance endpoint WITH timestamp filter (if checkpoint timestamp exists)
    if checkpoint_timestamp:
        print(f"[3] Testing /api/monitoring/performance WITH timestamp filter ({checkpoint_timestamp})...")
        try:
            response = requests.get(f"{base_url}/api/monitoring/performance", params={"since": checkpoint_timestamp})
            data = response.json()
            
            if data.get("status") == "success":
                metrics = data.get("metrics", {})
                total_trades = metrics.get("total_trades", 0)
                total_pnl = metrics.get("total_pnl", 0.0)
                filtered_since = metrics.get("filtered_since")
                
                print(f"   Total trades (filtered): {total_trades}")
                print(f"   Total PnL (filtered): ${total_pnl:.2f}")
                print(f"   Filtered since: {filtered_since}")
                
                # Compare with unfiltered
                if total_trades > 0:
                    print()
                    print("   Comparison:")
                    print(f"   - Filtered trades should be ≤ unfiltered trades")
                    print(f"   - If filtered trades < unfiltered trades, filtering is working!")
            else:
                print(f"   [ERROR] Error: {data.get('message', 'Unknown error')}")
            print()
        except Exception as e:
            print(f"   [ERROR] Error: {e}")
            print()
    
    # 4. Test with a specific timestamp (yesterday)
    print("[4] Testing with a specific timestamp (yesterday)...")
    try:
        from datetime import timedelta
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        response = requests.get(f"{base_url}/api/monitoring/performance", params={"since": yesterday})
        data = response.json()
        
        if data.get("status") == "success":
            metrics = data.get("metrics", {})
            total_trades = metrics.get("total_trades", 0)
            filtered_since = metrics.get("filtered_since")
            
            print(f"   Total trades (since yesterday): {total_trades}")
            print(f"   Filtered since: {filtered_since}")
        else:
            print(f"   ✗ Error: {data.get('message', 'Unknown error')}")
        print()
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print()
    
    # 5. Check frontend URL construction
    print("[5] Frontend URL construction check...")
    print("   The frontend should construct URLs like:")
    if checkpoint_timestamp:
        print(f"   /api/monitoring/performance?since={checkpoint_timestamp}")
    else:
        print("   /api/monitoring/performance (no filter)")
    print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("1. Check browser console (F12) for '[MonitoringPanel] Checkpoint resume timestamp' messages")
    print("2. Check if the 'Filtered since checkpoint resume' badge appears in the UI")
    print("3. Compare filtered vs unfiltered trade counts to verify filtering is working")
    print("=" * 80)

if __name__ == "__main__":
    test_monitoring_filter()

