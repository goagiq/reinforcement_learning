"""
Investigate Consecutive Loss Limit Logic

Check if consecutive loss limit is causing too many pauses
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.trading_env import TradingEnvironment
from src.data_extraction import DataExtractor
import pandas as pd
import numpy as np

def test_consecutive_loss_behavior():
    """Test how consecutive loss limit affects trading"""
    
    print("="*60)
    print("INVESTIGATING CONSECUTIVE LOSS LIMIT")
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
    
    # Test with different consecutive loss limits
    test_configs = [
        {"max_consecutive_losses": 3, "name": "Current (3)"},
        {"max_consecutive_losses": 5, "name": "Updated (5)"},
        {"max_consecutive_losses": 10, "name": "Relaxed (10)"},
    ]
    
    for test_config in test_configs:
        print(f"\n2. Testing with {test_config['name']} consecutive losses limit...")
        
        reward_config = {
            "max_consecutive_losses": test_config["max_consecutive_losses"],
            "inaction_penalty": 0.00005,
            "quality_filters": {
                "enabled": True,
                "min_action_confidence": 0.08,
                "min_quality_score": 0.25
            }
        }
        
        env = TradingEnvironment(
            data=data,
            timeframes=[1, 5, 15],
            max_episode_steps=1000,  # Shorter for testing
            reward_config=reward_config,
            action_threshold=0.01
        )
        
        state, info = env.reset()
        
        # Simulate a scenario with consecutive losses
        consecutive_losses_count = 0
        trading_paused_count = 0
        total_steps = 0
        trades_taken = 0
        trades_rejected = 0
        
        for step in range(1000):
            # Take random action
            action = np.array([np.random.uniform(-0.5, 0.5)])
            
            try:
                next_state, reward, terminated, truncated, step_info = env.step(action)
                total_steps += 1
                
                # Check if trading is paused
                if hasattr(env.state, 'trading_paused') and env.state.trading_paused:
                    trading_paused_count += 1
                    # Check if we tried to trade but were rejected
                    if abs(action[0]) > 0.01:
                        trades_rejected += 1
                
                # Check if a trade was taken
                if step_info.get("episode_trades", 0) > trades_taken:
                    trades_taken = step_info.get("episode_trades", 0)
                
                # Track consecutive losses
                if hasattr(env.state, 'consecutive_losses'):
                    consecutive_losses_count = max(consecutive_losses_count, env.state.consecutive_losses)
                
                if terminated or truncated:
                    break
                    
                state = next_state
                
            except Exception as e:
                print(f"   [ERROR] Exception at step {step}: {e}")
                break
        
        print(f"   Results:")
        print(f"     Total steps: {total_steps}")
        print(f"     Trades taken: {trades_taken}")
        print(f"     Trades rejected (while paused): {trades_rejected}")
        print(f"     Steps with trading paused: {trading_paused_count} ({trading_paused_count/max(1,total_steps):.1%})")
        print(f"     Max consecutive losses: {consecutive_losses_count}")
        print(f"     Pause impact: {trades_rejected} trades rejected due to pause")
        
        if trading_paused_count > total_steps * 0.5:
            print(f"     [WARNING] Trading paused for >50% of steps!")
            print(f"     Consider increasing max_consecutive_losses or improving trade quality")
    
    print("\n3. RECOMMENDATIONS")
    print("-" * 60)
    print("If trading is paused too often:")
    print("  - Increase max_consecutive_losses (currently 5)")
    print("  - Improve stop loss enforcement to reduce loss size")
    print("  - Relax quality filters to allow more trades")
    print("  - Consider resetting consecutive losses after a certain number of steps")
    
    print("\n" + "="*60)
    print("INVESTIGATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    test_consecutive_loss_behavior()

