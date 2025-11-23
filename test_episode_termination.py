"""
Test script to reproduce the 20-step episode termination issue
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
import numpy as np
from src.data_extraction import DataExtractor
from src.trading_env import TradingEnvironment

def test_episode_termination():
    """Test episode termination to reproduce 20-step issue"""
    print("=" * 80)
    print("EPISODE TERMINATION TEST")
    print("=" * 80)
    print()
    
    try:
        # Load config
        print("1. Loading configuration...")
        with open('configs/train_config_adaptive.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("   [OK] Config loaded")
        
        # Load data
        print("\n2. Loading data...")
        nt8_data_path = config.get("data", {}).get("nt8_data_path")
        extractor = DataExtractor(nt8_data_path=nt8_data_path)
        instrument = config["environment"]["instrument"]
        timeframes = config["environment"]["timeframes"]
        data = extractor.load_multi_timeframe_data(instrument, timeframes)
        primary_data = data[min(timeframes)]
        print(f"   [OK] Data loaded: {len(primary_data):,} bars")
        print(f"   Max episode steps: {config['environment']['max_episode_steps']:,}")
        print(f"   Lookback bars: {config['environment']['lookback_bars']}")
        
        # Create environment
        print("\n3. Creating environment...")
        env_config = config["environment"]
        reward_config = env_config.get("reward", {})
        env = TradingEnvironment(
            data=data,
            timeframes=timeframes,
            initial_capital=config.get("risk_management", {}).get("initial_capital", 100000.0),
            transaction_cost=reward_config.get("transaction_cost", 0.0003),
            lookback_bars=env_config.get("lookback_bars", 20),
            reward_config=reward_config,
            max_episode_steps=env_config.get("max_episode_steps", 10000),
            action_threshold=env_config.get("action_threshold", 0.01)
        )
        print("   [OK] Environment created")
        
        # Test episode
        print("\n4. Testing episode...")
        print("   Resetting environment...")
        state, info = env.reset()
        print(f"   [OK] Reset complete. Initial state shape: {state.shape}")
        print(f"   Current step: {env.current_step}")
        print(f"   Max steps: {env.max_steps}")
        
        # Run steps and monitor for early termination
        print("\n5. Running episode steps...")
        max_test_steps = env.max_steps  # Test full episode
        print(f"   Testing full episode: {max_test_steps} steps")
        terminated_early = False
        exception_occurred = False
        exception_step = None
        exception_type = None
        exception_message = None
        
        for step in range(max_test_steps):
            try:
                # Sample random action
                action = env.action_space.sample()
                
                # Take step
                next_state, reward, terminated, truncated, step_info = env.step(action)
                
                # Check for termination
                if terminated or truncated:
                    print(f"\n   [WARNING] Episode terminated at step {step + 1}")
                    print(f"   Terminated: {terminated}, Truncated: {truncated}")
                    
                    # Check for error in step_info
                    if 'error' in step_info:
                        print(f"   [ERROR] Error: {step_info['error']}")
                        exception_occurred = True
                    
                    # Check termination reason
                    if 'termination_reason' in step_info:
                        print(f"   Reason: {step_info['termination_reason']}")
                    
                    terminated_early = True
                    break
                
                # Log every 100 steps (or at key milestones)
                if (step + 1) % 100 == 0 or step + 1 <= 30:
                    print(f"   Step {step + 1}: current_step={env.current_step}, reward={reward:.4f}, terminated={terminated}")
                
                # Check for exceptions in step_info
                if step_info and 'error' in step_info:
                    print(f"\n   [ERROR] Exception at step {step + 1}: {step_info['error']}")
                    exception_occurred = True
                    break
                    
            except Exception as e:
                print(f"\n   [ERROR] EXCEPTION at step {step + 1}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                exception_occurred = True
                exception_step = step + 1
                exception_type = type(e).__name__
                exception_message = str(e)
                break
        
        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        if terminated_early:
            print(f"[FAIL] Episode terminated early at step {step + 1}!")
            if exception_occurred:
                print(f"   Root cause: Exception occurred at step {exception_step}")
                print(f"   Exception type: {exception_type}")
                print(f"   Exception message: {exception_message}")
            else:
                print("   Root cause: Normal termination (check termination logic)")
                print(f"   Terminated: {terminated}, Truncated: {truncated}")
        elif exception_occurred:
            print(f"[FAIL] Exception occurred during episode at step {exception_step}")
            print(f"   Exception type: {exception_type}")
            print(f"   Exception message: {exception_message}")
        else:
            print(f"[OK] Episode completed successfully: {step + 1} steps")
            if step + 1 < max_test_steps:
                print(f"   Note: Stopped at {step + 1} steps (test limit reached)")
            else:
                print(f"   Episode ran for full {max_test_steps} steps without issues")
        
        print(f"\nFinal state:")
        print(f"   Current step: {env.current_step}")
        print(f"   Max steps: {env.max_steps}")
        print(f"   Episode trades: {env.episode_trades}")
        print(f"   Episode length: {step + 1} steps")
        
        # Check if episode was suspiciously short
        if step + 1 < 100 and not exception_occurred:
            print(f"\n[WARNING] Episode was very short ({step + 1} steps) but no exception occurred")
            print("   This might indicate a data boundary issue or early termination logic")
        
        return not terminated_early and not exception_occurred
        
    except Exception as e:
        print(f"\n[ERROR] FATAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_episode_termination()
    sys.exit(0 if success else 1)

