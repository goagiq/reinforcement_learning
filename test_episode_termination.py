"""
Test script to verify episode termination fix.
Tests that episodes run for multiple steps instead of terminating in 1 step.
"""

import sys
import yaml
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_extraction import DataExtractor
from src.trading_env import TradingEnvironment
from src.trading_hours import TradingHoursManager

def test_episode_termination():
    """Test that episodes run for multiple steps"""
    print("=" * 80)
    print("TESTING EPISODE TERMINATION FIX")
    print("=" * 80)
    
    # Load config
    config_path = Path("configs/train_config_adaptive.yaml")
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    print("\n[1/5] Loading data...")
    data_extractor = DataExtractor()
    data_dir = Path("data/raw")
    
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return False
    
    # Get first available data file
    data_files = list(data_dir.glob("*.txt"))
    if not data_files:
        print(f"[ERROR] No data files found in {data_dir}")
        return False
    
    print(f"  Found {len(data_files)} data files, using: {data_files[0].name}")
    
    try:
        trading_hours = TradingHoursManager.from_dict(config.get("environment", {}).get("trading_hours", {}))
    except Exception as e:
        print(f"[WARN] Failed to create TradingHoursManager: {e}, using None")
        trading_hours = None
    
    timeframes = config.get("environment", {}).get("timeframes", [1, 5, 15])
    instrument = config.get("environment", {}).get("instrument", "ES")
    data = data_extractor.load_multi_timeframe_data(
        instrument=instrument,
        timeframes=timeframes,
        trading_hours=trading_hours
    )
    
    if not data:
        print("[ERROR] Failed to load data")
        return False
    
    print(f"  Loaded data: {[f'{tf}min: {len(df)} bars' for tf, df in data.items()]}")
    
    # Create environment
    print("\n[2/5] Creating environment...")
    env_config = config.get("environment", {})
    reward_config = env_config.get("reward", {})
    
    try:
        env = TradingEnvironment(
            data=data,
            timeframes=timeframes,
            initial_capital=100000.0,
            transaction_cost=reward_config.get("transaction_cost", 0.0001),
            lookback_bars=env_config.get("lookback_bars", 20),
            reward_config=reward_config,
            max_episode_steps=env_config.get("max_episode_steps", 10000),
            action_threshold=env_config.get("action_threshold", 0.01)
        )
        print("  Environment created successfully")
    except Exception as e:
        print(f"[ERROR] Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test episode reset
    print("\n[3/5] Testing episode reset...")
    try:
        state, info = env.reset()
        print(f"  Reset successful: state shape={state.shape}, step={info.get('step', 'N/A')}")
        print(f"  Current step: {env.current_step}")
        print(f"  Max steps: {env.max_steps}")
        
        # Check data length
        primary_data = env.data[min(env.timeframes)]
        data_len = len(primary_data)
        remaining = data_len - env.current_step
        print(f"  Data length: {data_len}")
        print(f"  Remaining data: {remaining}")
        print(f"  Lookback bars: {env.lookback_bars}")
        
        if remaining <= env.lookback_bars:
            print(f"  [WARN] Not enough data remaining! ({remaining} <= {env.lookback_bars})")
            return False
        else:
            print(f"  [OK] Enough data remaining ({remaining} > {env.lookback_bars})")
    except Exception as e:
        print(f"[ERROR] Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test episode execution (run for multiple steps)
    print("\n[4/5] Testing episode execution (10 steps)...")
    episode_length = 0
    terminated = False
    truncated = False
    
    try:
        for step in range(10):
            # Random action
            action = np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)
            
            next_state, reward, term, trunc, step_info = env.step(action)
            
            episode_length += 1
            terminated = term
            truncated = trunc
            
            if terminated or truncated:
                print(f"  Step {step + 1}: Episode terminated/truncated")
                break
            
            if step < 5:  # Print first 5 steps
                print(f"  Step {step + 1}: reward={reward:.6f}, current_step={env.current_step}, terminated={terminated}")
        
        print(f"  Episode ran for {episode_length} steps")
        
        if episode_length == 1:
            print("  [ERROR] Episode terminated in 1 step - FIX NOT WORKING!")
            return False
        elif episode_length < 5:
            print(f"  [WARN] Episode only ran for {episode_length} steps (expected more)")
            return False
        else:
            print(f"  [OK] Episode ran for {episode_length} steps (expected behavior)")
    except Exception as e:
        print(f"[ERROR] Episode execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test multiple episode resets
    print("\n[5/5] Testing multiple episode resets...")
    try:
        episode_lengths = []
        for episode in range(3):
            state, info = env.reset()
            episode_length = 0
            
            for step in range(100):  # Limit to 100 steps per episode for testing
                action = np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)
                next_state, reward, term, trunc, step_info = env.step(action)
                
                episode_length += 1
                if term or trunc:
                    break
            
            episode_lengths.append(episode_length)
            print(f"  Episode {episode + 1}: {episode_length} steps")
        
        avg_length = sum(episode_lengths) / len(episode_lengths)
        print(f"  Average episode length: {avg_length:.1f} steps")
        
        if all(length == 1 for length in episode_lengths):
            print("  [ERROR] All episodes terminated in 1 step - FIX NOT WORKING!")
            return False
        elif avg_length < 5:
            print(f"  [WARN] Average episode length too short ({avg_length:.1f} steps)")
            return False
        else:
            print(f"  [OK] Episodes running for reasonable length ({avg_length:.1f} steps average)")
    except Exception as e:
        print(f"[ERROR] Multiple episode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("[OK] ALL TESTS PASSED - Episode termination fix verified!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_episode_termination()
    sys.exit(0 if success else 1)
