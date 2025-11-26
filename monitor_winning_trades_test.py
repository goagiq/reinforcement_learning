"""
Monitor Training for Winning Trades Test

This script starts training (if not running) and monitors for winning trades
to verify that disabling stop loss allows profitable trades.
"""

import requests
import time
import json
from datetime import datetime
from pathlib import Path

API_BASE_URL = "http://localhost:8200"

def check_training_status():
    """Check if training is currently running"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/training/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"[ERROR] Failed to check training status: {e}")
        return None

def get_performance_metrics():
    """Get performance metrics including winning trades"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/monitoring/performance", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"[ERROR] Failed to get performance metrics: {e}")
        return None

def start_training(config_path="configs/train_config_adaptive.yaml", checkpoint_path=None):
    """Start training via API"""
    try:
        request_data = {
            "device": "cuda",  # Use GPU if available
            "config_path": config_path,
            "total_timesteps": 20000000
        }
        
        if checkpoint_path:
            request_data["checkpoint_path"] = checkpoint_path
        
        print(f"[INFO] Starting training with config: {config_path}")
        if checkpoint_path:
            print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")
        
        response = requests.post(
            f"{API_BASE_URL}/api/training/start",
            json=request_data,
            timeout=10
        )
        
        if response.status_code == 200:
            print("[SUCCESS] Training started successfully!")
            return True
        else:
            print(f"[ERROR] Failed to start training: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"[ERROR] Exception starting training: {e}")
        return False

def monitor_training(max_duration_minutes=15, check_interval_seconds=30):
    """
    Monitor training and check for winning trades
    
    Args:
        max_duration_minutes: Maximum time to monitor (default: 15 minutes)
        check_interval_seconds: How often to check status (default: 30 seconds)
    """
    print("="*80)
    print("MONITORING TRAINING FOR WINNING TRADES TEST")
    print("="*80)
    print()
    print(f"Monitor duration: {max_duration_minutes} minutes")
    print(f"Check interval: {check_interval_seconds} seconds")
    print()
    
    start_time = time.time()
    max_duration_seconds = max_duration_minutes * 60
    
    # Check if training is already running
    status = check_training_status()
    training_running = False
    
    if status and status.get("status") == "running":
        print("[INFO] Training is already running")
        training_running = True
    else:
        print("[INFO] Training is not running - will start it")
        # Check for latest checkpoint
        checkpoint_path = None
        models_dir = Path("models")
        if models_dir.exists():
            checkpoints = sorted(models_dir.glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if checkpoints:
                checkpoint_path = str(checkpoints[0])
                print(f"[INFO] Found checkpoint: {checkpoint_path}")
        
        # Start training
        if start_training(checkpoint_path=checkpoint_path):
            training_running = True
            # Wait a bit for training to initialize
            print("[INFO] Waiting 10 seconds for training to initialize...")
            time.sleep(10)
        else:
            print("[ERROR] Failed to start training. Please start manually.")
            return False
    
    # Initial metrics
    last_total_trades = 0
    last_winning_trades = 0
    
    print()
    print("="*80)
    print("MONITORING PROGRESS")
    print("="*80)
    print()
    
    check_count = 0
    
    while (time.time() - start_time) < max_duration_seconds:
        check_count += 1
        elapsed_minutes = (time.time() - start_time) / 60
        
        # Get training status
        status = check_training_status()
        if not status:
            print(f"[{elapsed_minutes:.1f}m] ERROR: Could not get training status")
            time.sleep(check_interval_seconds)
            continue
        
        # Get performance metrics
        perf = get_performance_metrics()
        
        # Extract key metrics
        timestep = status.get("timestep", 0)
        episode = status.get("episode", 0)
        progress = status.get("progress_percent", 0.0)
        
        if perf:
            total_trades = perf.get("total_trades", 0)
            winning_trades = perf.get("winning_trades", 0)
            losing_trades = perf.get("losing_trades", 0)
            win_rate = perf.get("win_rate", 0.0)
            total_pnl = perf.get("total_pnl", 0.0)
            
            # Check if we have new trades
            new_trades = total_trades - last_total_trades
            new_wins = winning_trades - last_winning_trades
            
            # Display status
            print(f"[{elapsed_minutes:.1f}m] Check #{check_count}")
            print(f"   Timestep: {timestep:,} ({progress:.1f}%) | Episode: {episode}")
            print(f"   Total Trades: {total_trades} | Winning: {winning_trades} | Losing: {losing_trades}")
            print(f"   Win Rate: {win_rate:.2f}% | Total P&L: ${total_pnl:,.2f}")
            
            if new_trades > 0:
                print(f"   🆕 NEW TRADES: {new_trades} (New wins: {new_wins})")
            
            # Success criteria
            if winning_trades > 0:
                print()
                print("="*80)
                print("✅ SUCCESS: WINNING TRADES DETECTED!")
                print("="*80)
                print(f"   Winning Trades: {winning_trades}")
                print(f"   Win Rate: {win_rate:.2f}%")
                print(f"   Total Trades: {total_trades}")
                print()
                print("Stop loss disable test is WORKING - trades can be profitable!")
                print("="*80)
                return True
            
            # Update last values
            last_total_trades = total_trades
            last_winning_trades = winning_trades
            
        else:
            print(f"[{elapsed_minutes:.1f}m] Timestep: {timestep:,} | Episode: {episode} | Progress: {progress:.1f}%")
            print(f"   (Performance metrics not available yet)")
        
        print()
        
        # Check if training stopped
        if status.get("status") != "running":
            print(f"[{elapsed_minutes:.1f}m] Training status: {status.get('status')}")
            print("   Training has stopped")
            break
        
        time.sleep(check_interval_seconds)
    
    # Final summary
    print()
    print("="*80)
    print("MONITORING COMPLETE")
    print("="*80)
    print(f"Duration: {elapsed_minutes:.1f} minutes")
    print(f"Checks performed: {check_count}")
    print()
    
    # Final metrics
    perf = get_performance_metrics()
    if perf:
        total_trades = perf.get("total_trades", 0)
        winning_trades = perf.get("winning_trades", 0)
        win_rate = perf.get("win_rate", 0.0)
        
        print("FINAL METRICS:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Winning Trades: {winning_trades}")
        print(f"   Win Rate: {win_rate:.2f}%")
        print()
        
        if winning_trades > 0:
            print("✅ SUCCESS: Found winning trades!")
            return True
        else:
            print("⚠️  WARNING: Still 0% win rate after {elapsed_minutes:.1f} minutes")
            print("   This suggests a deeper issue beyond stop loss")
            return False
    else:
        print("⚠️  Could not get final performance metrics")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor training for winning trades")
    parser.add_argument("--duration", type=int, default=15, help="Monitoring duration in minutes (default: 15)")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds (default: 30)")
    parser.add_argument("--config", type=str, default="configs/train_config_adaptive.yaml", help="Config file path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (optional)")
    
    args = parser.parse_args()
    
    # Run monitoring
    success = monitor_training(
        max_duration_minutes=args.duration,
        check_interval_seconds=args.interval
    )
    
    exit(0 if success else 1)

