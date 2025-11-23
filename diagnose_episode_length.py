"""
Diagnostic script to test episode length without impacting training.
This creates a separate environment instance to test episode termination.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def check_data_lengths():
    """Check if data files have enough bars"""
    print("\n[1] CHECKING DATA FILE LENGTHS")
    print("-" * 60)
    
    data_dir = project_root / "data" / "raw"
    if not data_dir.exists():
        print(f"  [ERROR] Data directory not found: {data_dir}")
        return False
    
    # Check for ES data files
    es_files = list(data_dir.glob("ES_*.csv"))
    if not es_files:
        print(f"  [WARN] No ES data files found in {data_dir}")
        return False
    
    print(f"  Found {len(es_files)} ES data file(s)")
    
    max_episode_steps = 10000
    lookback_bars = 20
    required_length = max_episode_steps + lookback_bars
    
    print(f"\n  Required data length: {required_length} bars")
    print(f"  (max_episode_steps: {max_episode_steps} + lookback_bars: {lookback_bars})")
    
    for file in es_files:
        try:
            df = pd.read_csv(file)
            actual_length = len(df)
            print(f"\n  File: {file.name}")
            print(f"    Actual length: {actual_length} bars")
            
            if actual_length < required_length:
                print(f"    [ERROR] Data is too short!")
                print(f"    Missing: {required_length - actual_length} bars")
                print(f"    This will cause episodes to terminate early!")
                return False
            else:
                print(f"    [OK] Data length is sufficient")
                print(f"    Can support {actual_length - lookback_bars} steps per episode")
        except Exception as e:
            print(f"    [ERROR] Failed to read file: {e}")
            return False
    
    return True

def test_episode_termination():
    """Test episode termination logic with a minimal environment"""
    print("\n[2] TESTING EPISODE TERMINATION LOGIC")
    print("-" * 60)
    
    try:
        from src.trading_env import TradingEnvironment
        
        # Create minimal test data
        num_bars = 15000  # More than enough
        test_data = {
            1: pd.DataFrame({
                'open': np.random.rand(num_bars) * 100 + 1000,
                'high': np.random.rand(num_bars) * 10 + 1050,
                'low': np.random.rand(num_bars) * 10 + 990,
                'close': np.random.rand(num_bars) * 100 + 1000,
                'volume': np.random.randint(100, 1000, num_bars)
            })
        }
        
        # Create environment with same config as training
        env = TradingEnvironment(
            data=test_data,
            timeframes=[1],
            max_episode_steps=10000,
            lookback_bars=20,
            reward_config={}
        )
        
        print("  [OK] Environment created successfully")
        
        # Test a few steps
        state, info = env.reset()
        print(f"  After reset: current_step={env.current_step}, max_steps={env.max_steps}")
        
        # Run a few steps to see if termination works correctly
        steps_taken = 0
        max_test_steps = 100  # Just test a few steps
        
        for i in range(max_test_steps):
            action = np.array([0.0], dtype=np.float32)  # Hold action
            next_state, reward, terminated, truncated, step_info = env.step(action)
            steps_taken += 1
            
            if terminated or truncated:
                print(f"  [FOUND] Episode terminated at step {steps_taken}")
                print(f"    terminated={terminated}, truncated={truncated}")
                print(f"    current_step={env.current_step}, max_steps={env.max_steps}")
                break
        
        if not (terminated or truncated):
            print(f"  [OK] Episode did not terminate early (tested {steps_taken} steps)")
            print(f"  [INFO] This suggests termination logic is working correctly")
            print(f"  [INFO] Actual training episodes might terminate due to data length or other issues")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Failed to test environment: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_training_logs():
    """Suggest checking training logs for patterns"""
    print("\n[3] CHECKING TRAINING LOGS")
    print("-" * 60)
    
    log_dir = project_root / "logs"
    if not log_dir.exists():
        print(f"  [INFO] Log directory not found: {log_dir}")
        print(f"  [INFO] Check console output for '[DEBUG] Episode completing' messages")
        return
    
    print(f"  Log directory: {log_dir}")
    print(f"  [INFO] Look for patterns in episode completion messages")
    print(f"  [INFO] Check for '[DEBUG] Episode completing' messages")
    print(f"  [INFO] These should show episode length and termination reason")

def main():
    """Run all diagnostic checks"""
    print("\n" + "=" * 60)
    print("EPISODE LENGTH DIAGNOSTIC (Non-Intrusive)")
    print("=" * 60)
    
    # Check data lengths
    data_ok = check_data_lengths()
    
    # Test episode termination
    if data_ok:
        test_episode_termination()
    
    # Suggest checking logs
    check_training_logs()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    
    print("\n[SUMMARY]")
    if not data_ok:
        print("  [CRITICAL] Data files are too short - this will cause early termination!")
        print("  [ACTION] Need to ensure data files have at least 10,020 bars")
    else:
        print("  [OK] Data files appear to have sufficient length")
        print("  [INFO] If episodes are still short, check:")
        print("    - Training logs for error messages")
        print("    - Exception handling in step() method")
        print("    - Boundary conditions in data access")

if __name__ == "__main__":
    main()

