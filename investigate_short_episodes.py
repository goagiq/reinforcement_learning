"""
Investigate Short Episode Issue

Check for:
1. Data boundary issues
2. Consecutive loss limit causing early termination
3. Trading paused state preventing progress
4. IndexError exceptions
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.trading_env import TradingEnvironment
from src.data_extraction import DataExtractor
import pandas as pd
import numpy as np

def test_episode_termination():
    """Test what causes episodes to terminate early"""
    
    print("="*60)
    print("INVESTIGATING SHORT EPISODE ISSUE")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    extractor = DataExtractor("data")
    data = extractor.load_multi_timeframe_data(
        instrument="ES",
        timeframes=[1, 5, 15]
    )
    
    if not data:
        print("[ERROR] Could not load data")
        return
    
    print(f"   Data loaded: {len(data[1])} bars for 1min timeframe")
    
    # Create environment
    print("\n2. Creating environment...")
    reward_config = {
        "max_consecutive_losses": 3,  # Default
        "inaction_penalty": -0.001,
        "quality_filters": {
            "enabled": True,
            "min_action_confidence": 0.15,
            "min_quality_score": 0.4
        }
    }
    
    env = TradingEnvironment(
        data=data,
        timeframes=[1, 5, 15],
        max_episode_steps=10000,
        reward_config=reward_config,
        action_threshold=0.02
    )
    
    print(f"   Max episode steps: {env.max_steps}")
    print(f"   Data length: {len(data[1])}")
    print(f"   Available steps: {len(data[1]) - env.lookback_bars}")
    
    # Test episode with consecutive losses
    print("\n3. Testing episode with consecutive losses...")
    state, info = env.reset()
    episode_length = 0
    consecutive_losses = 0
    
    for step in range(10000):
        # Take random action
        action = np.array([np.random.uniform(-1.0, 1.0)])
        
        try:
            next_state, reward, terminated, truncated, step_info = env.step(action)
            episode_length += 1
            
            # Check if trading is paused
            if hasattr(env.state, 'trading_paused') and env.state.trading_paused:
                print(f"   Step {step}: Trading PAUSED (consecutive_losses={env.state.consecutive_losses})")
                consecutive_losses = env.state.consecutive_losses
            
            # Check for early termination
            if terminated or truncated:
                print(f"\n   Episode terminated at step {episode_length}")
                print(f"   Reason: terminated={terminated}, truncated={truncated}")
                print(f"   Current step: {env.current_step}")
                print(f"   Max steps: {env.max_steps}")
                print(f"   Trading paused: {getattr(env.state, 'trading_paused', False)}")
                print(f"   Consecutive losses: {getattr(env.state, 'consecutive_losses', 0)}")
                break
            
            state = next_state
            
            # Check data boundaries
            if env.current_step >= len(data[1]) - env.lookback_bars:
                print(f"\n   [WARNING] Approaching data boundary at step {step}")
                print(f"   Current step: {env.current_step}")
                print(f"   Data length: {len(data[1])}")
                print(f"   Lookback bars: {env.lookback_bars}")
                print(f"   Available steps: {len(data[1]) - env.lookback_bars}")
                break
                
        except IndexError as e:
            print(f"\n   [ERROR] IndexError at step {step}: {e}")
            print(f"   Current step: {env.current_step}")
            print(f"   Data length: {len(data[1])}")
            break
        except Exception as e:
            print(f"\n   [ERROR] Exception at step {step}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print(f"\n4. Episode Summary:")
    print(f"   Final length: {episode_length}")
    print(f"   Expected length: {env.max_steps}")
    print(f"   Ratio: {episode_length / env.max_steps:.1%}")
    
    if episode_length < env.max_steps * 0.5:
        print(f"\n   [CRITICAL] Episode terminated early!")
        print(f"   Possible causes:")
        print(f"   - Data boundary reached")
        print(f"   - Trading paused for too long")
        print(f"   - IndexError exception")
        print(f"   - Early termination logic")
    
    # Test with trading paused
    print("\n5. Testing behavior when trading is paused...")
    state, info = env.reset()
    
    # Force trading to pause by taking losing trades
    for i in range(5):
        # Take action that will likely lose
        action = np.array([1.0])  # Long position
        next_state, reward, terminated, truncated, step_info = env.step(action)
        
        # Simulate a loss by closing position at lower price
        # (This is simplified - in reality loss depends on price movement)
        if hasattr(env.state, 'trading_paused') and env.state.trading_paused:
            print(f"   Trading paused after {i+1} trades")
            print(f"   Consecutive losses: {env.state.consecutive_losses}")
            break
    
    # Now try to take actions while paused
    print("\n   Testing actions while paused...")
    actions_taken = 0
    for step in range(100):
        action = np.array([np.random.uniform(-1.0, 1.0)])
        next_state, reward, terminated, truncated, step_info = env.step(action)
        
        # Check if position changed
        if abs(step_info.get('position_change', 0)) > 0.01:
            actions_taken += 1
        
        if terminated or truncated:
            print(f"   Episode terminated at step {step} while paused")
            break
    
    print(f"   Actions taken while paused: {actions_taken}")
    print(f"   (Should be 0 if pause is working correctly)")
    
    print("\n" + "="*60)
    print("INVESTIGATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    test_episode_termination()

