"""
Test script to verify:
1. Adaptive learning no-trades fix (relaxes filters when no trades detected)
2. Trailing stop and take profit are re-enabled
3. Config loads correctly with all features
"""

import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_config_loading():
    """Test that config loads correctly with trailing stop and take profit enabled"""
    print("=" * 80)
    print("TEST 1: Config Loading")
    print("=" * 80)
    
    config_path = Path("configs/train_config_adaptive.yaml")
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check trailing stop
    trailing_stop = config.get("environment", {}).get("reward", {}).get("trailing_stop", {})
    trailing_enabled = trailing_stop.get("enabled", False)
    
    if not trailing_enabled:
        print("[ERROR] Trailing stop is NOT enabled!")
        return False
    else:
        print(f"[OK] Trailing stop: ENABLED")
        print(f"  - ATR multiplier: {trailing_stop.get('atr_multiplier', 'N/A')}")
        print(f"  - Activation: {trailing_stop.get('activation_pct', 'N/A')*100:.1f}%")
    
    # Check take profit
    take_profit = config.get("environment", {}).get("reward", {}).get("take_profit", {})
    take_profit_enabled = take_profit.get("enabled", False)
    
    if not take_profit_enabled:
        print("[ERROR] Take profit is NOT enabled!")
        return False
    else:
        print(f"[OK] Take profit: ENABLED")
        print(f"  - Mode: {take_profit.get('mode', 'N/A')}")
        print(f"  - R:R ratio: {take_profit.get('risk_reward_ratio', 'N/A')}")
    
    print("\n[OK] Config loading test PASSED")
    return True

def test_adaptive_learning_logic():
    """Test that adaptive learning relaxes filters when no trades detected"""
    print("\n" + "=" * 80)
    print("TEST 2: Adaptive Learning No-Trades Fix")
    print("=" * 80)
    
    try:
        from src.adaptive_trainer import AdaptiveTrainer, AdaptiveConfig
        
        # Create a minimal config for testing
        test_config_path = Path("configs/train_config_adaptive.yaml")
        if not test_config_path.exists():
            print(f"[ERROR] Config file not found: {test_config_path}")
            return False
        
        # Initialize adaptive trainer
        adaptive_trainer = AdaptiveTrainer(str(test_config_path))
        
        # Store initial values
        initial_confidence = adaptive_trainer.current_min_action_confidence
        initial_quality = adaptive_trainer.current_min_quality_score
        
        print(f"Initial values:")
        print(f"  - min_action_confidence: {initial_confidence:.3f}")
        print(f"  - min_quality_score: {initial_quality:.3f}")
        
        # Simulate no trades condition
        adaptive_trainer.consecutive_no_trade_episodes = 1
        
        # Call quick_adjust_for_negative_trend with no trades
        adjustments = adaptive_trainer.quick_adjust_for_negative_trend(
            recent_mean_pnl=-100.0,  # Negative PnL
            recent_win_rate=0.0,
            agent=None,  # Not needed for this test
            recent_trades_data=None,
            recent_total_trades=0  # KEY: No trades!
        )
        
        # Check if filters were relaxed
        new_confidence = adaptive_trainer.current_min_action_confidence
        new_quality = adaptive_trainer.current_min_quality_score
        
        print(f"\nAfter no-trades adjustment:")
        print(f"  - min_action_confidence: {new_confidence:.3f}")
        print(f"  - min_quality_score: {new_quality:.3f}")
        
        # Verify filters were relaxed (decreased)
        if new_confidence >= initial_confidence:
            print(f"[ERROR] Confidence filter was NOT relaxed! ({initial_confidence:.3f} -> {new_confidence:.3f})")
            return False
        
        if new_quality >= initial_quality:
            print(f"[ERROR] Quality filter was NOT relaxed! ({initial_quality:.3f} -> {new_quality:.3f})")
            return False
        
        print(f"\n[OK] Filters were relaxed correctly:")
        print(f"  - Confidence: {initial_confidence:.3f} -> {new_confidence:.3f} (decreased by {initial_confidence - new_confidence:.3f})")
        print(f"  - Quality: {initial_quality:.3f} -> {new_quality:.3f} (decreased by {initial_quality - new_quality:.3f})")
        
        # Verify adjustments dict was returned
        if adjustments and "quality_filters" in adjustments:
            print(f"\n[OK] Adjustments dict returned correctly")
            print(f"  - Reason: {adjustments['quality_filters']['min_action_confidence'].get('reason', 'N/A')}")
        else:
            print(f"[WARN] No adjustments dict returned (might be expected if already at floor)")
        
        print("\n[OK] Adaptive learning no-trades fix test PASSED")
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trading_env_initialization():
    """Test that trading environment can initialize with trailing stop and take profit"""
    print("\n" + "=" * 80)
    print("TEST 3: Trading Environment Initialization")
    print("=" * 80)
    
    try:
        import yaml
        from src.trading_env import TradingEnvironment
        from src.data_extraction import DataExtractor
        from src.trading_hours import TradingHoursManager
        import pandas as pd
        
        # Load config
        config_path = Path("configs/train_config_adaptive.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create minimal test data
        test_data = {}
        for tf in [1, 5, 15]:
            # Create minimal DataFrame
            dates = pd.date_range('2024-01-01', periods=100, freq=f'{tf}min')
            test_data[tf] = pd.DataFrame({
                'timestamp': dates,
                'open': [100.0] * 100,
                'high': [101.0] * 100,
                'low': [99.0] * 100,
                'close': [100.0] * 100,
                'volume': [1000] * 100
            })
        
        # Create environment
        env_config = config.get("environment", {})
        reward_config = env_config.get("reward", {})
        
        env = TradingEnvironment(
            data=test_data,
            timeframes=[1, 5, 15],
            initial_capital=100000.0,
            transaction_cost=reward_config.get("transaction_cost", 0.0001),
            lookback_bars=env_config.get("lookback_bars", 20),
            reward_config=reward_config,
            max_episode_steps=env_config.get("max_episode_steps", 10000),
            action_threshold=env_config.get("action_threshold", 0.01)
        )
        
        # Check trailing stop
        if hasattr(env, 'trailing_stop_enabled'):
            if env.trailing_stop_enabled:
                print(f"[OK] Trailing stop initialized: ENABLED")
                print(f"  - ATR multiplier: {env.trailing_stop_atr_multiplier}")
            else:
                print(f"[ERROR] Trailing stop initialized but DISABLED")
                return False
        else:
            print(f"[ERROR] Trailing stop attribute not found")
            return False
        
        # Check take profit
        if hasattr(env, 'take_profit_enabled'):
            if env.take_profit_enabled:
                print(f"[OK] Take profit initialized: ENABLED")
                print(f"  - Mode: {env.take_profit_mode}")
                print(f"  - R:R ratio: {env.take_profit_rr_ratio}")
            else:
                print(f"[ERROR] Take profit initialized but DISABLED")
                return False
        else:
            print(f"[ERROR] Take profit attribute not found")
            return False
        
        print("\n[OK] Trading environment initialization test PASSED")
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("TESTING ADAPTIVE LEARNING FIX AND RE-ENABLED FEATURES")
    print("=" * 80)
    
    results = []
    
    # Test 1: Config loading
    results.append(("Config Loading", test_config_loading()))
    
    # Test 2: Adaptive learning logic
    results.append(("Adaptive Learning Fix", test_adaptive_learning_logic()))
    
    # Test 3: Trading environment initialization
    results.append(("Trading Environment", test_trading_env_initialization()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n" + "=" * 80)
        print("[OK] ALL TESTS PASSED!")
        print("=" * 80)
        return True
    else:
        print("\n" + "=" * 80)
        print("[ERROR] SOME TESTS FAILED")
        print("=" * 80)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

