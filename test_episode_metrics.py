"""
Quick test to verify episode metrics are being captured correctly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_episode_trades_tracking():
    """Test that episode_trades is tracked correctly"""
    print("\n[TEST] Episode Trades Tracking")
    print("=" * 60)
    
    try:
        from src.trading_env import TradingEnvironment
        import pandas as pd
        
        # Create minimal data
        data = {
            1: pd.DataFrame({
                'open': [100.0] * 100,
                'high': [101.0] * 100,
                'low': [99.0] * 100,
                'close': [100.0] * 100,
                'volume': [1000] * 100
            })
        }
        
        env = TradingEnvironment(
            data=data,
            timeframes=[1],
            max_episode_steps=10,  # Short episode for testing
            reward_config={}  # Provide empty reward_config
        )
        
        state, info = env.reset()
        print(f"  After reset: episode_trades={env.episode_trades}")
        
        # Simulate a few steps with trades
        for i in range(5):
            action = [0.6] if i % 2 == 0 else [0.0]  # Alternate between trade and hold
            next_state, reward, terminated, truncated, step_info = env.step(action)
            print(f"  Step {i+1}: episode_trades={env.episode_trades}, step_info.episode_trades={step_info.get('episode_trades', 'N/A')}")
        
        print(f"  Final: episode_trades={env.episode_trades}")
        
        # Check if episode_trades is in step_info
        if "episode_trades" in step_info:
            print(f"  [OK] episode_trades is in step_info: {step_info['episode_trades']}")
        else:
            print(f"  [ERROR] episode_trades NOT in step_info!")
            print(f"  step_info keys: {list(step_info.keys())}")
            return False
        
        print("[OK] Episode trades tracking works")
        return True
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_episode_trades_tracking()

