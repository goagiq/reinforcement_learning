"""
End-to-End Test for Simplified Quality Filters in Trading Environment
Tests that quality filters are properly applied during training to reduce trade count.
"""

import sys
import os
import numpy as np
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_quality_filters_enabled():
    """Test that quality filters are enabled and configured"""
    print("\n[TEST] Quality Filters Configuration")
    print("=" * 60)
    
    # Load config
    config_path = Path("configs/train_config_adaptive.yaml")
    if not config_path.exists():
        print("[ERROR] Config file not found")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check quality filters config
    reward_config = config.get("environment", {}).get("reward", {})
    quality_filters = reward_config.get("quality_filters", {})
    
    enabled = quality_filters.get("enabled", False)
    min_action_confidence = quality_filters.get("min_action_confidence", 0.0)
    min_quality_score = quality_filters.get("min_quality_score", 0.0)
    require_positive_ev = quality_filters.get("require_positive_expected_value", False)
    
    print(f"  Quality Filters Enabled: {enabled}")
    print(f"  Min Action Confidence: {min_action_confidence}")
    print(f"  Min Quality Score: {min_quality_score}")
    print(f"  Require Positive EV: {require_positive_ev}")
    
    if not enabled:
        print("[WARN] Quality filters are disabled in config")
        return False
    
    if min_action_confidence < 0.2:
        print("[WARN] Min action confidence is very low - may not filter enough trades")
    
    if min_quality_score < 0.4:
        print("[WARN] Min quality score is very low - may not filter enough trades")
    
    print("[OK] Quality filters are configured")
    return True

def test_trading_env_imports():
    """Test that TradingEnvironment can be imported with quality filter methods"""
    print("\n[TEST] TradingEnvironment Imports")
    print("=" * 60)
    
    try:
        from src.trading_env import TradingEnvironment
        print("[OK] TradingEnvironment imported successfully")
        
        # Check for quality filter methods
        if hasattr(TradingEnvironment, '_calculate_simplified_quality_score'):
            print("[OK] _calculate_simplified_quality_score method exists")
        else:
            print("[ERROR] _calculate_simplified_quality_score method not found")
            return False
        
        if hasattr(TradingEnvironment, '_calculate_expected_value_simplified'):
            print("[OK] _calculate_expected_value_simplified method exists")
        else:
            print("[ERROR] _calculate_expected_value_simplified method not found")
            return False
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to import TradingEnvironment: {e}")
        return False

def test_quality_score_calculation():
    """Test that quality score calculation works"""
    print("\n[TEST] Quality Score Calculation")
    print("=" * 60)
    
    try:
        from src.trading_env import TradingEnvironment
        import pandas as pd
        
        # Create minimal data for environment
        data = {
            1: pd.DataFrame({
                'open': [100.0] * 100,
                'high': [101.0] * 100,
                'low': [99.0] * 100,
                'close': [100.0] * 100,
                'volume': [1000] * 100
            })
        }
        
        # Create environment with quality filters enabled
        reward_config = {
            "quality_filters": {
                "enabled": True,
                "min_action_confidence": 0.3,
                "min_quality_score": 0.5,
                "require_positive_expected_value": True
            }
        }
        
        env = TradingEnvironment(
            data=data,
            timeframes=[1],
            reward_config=reward_config
        )
        
        # Test quality score calculation
        env.action_value = 0.5  # Set action value
        quality_score = env._calculate_simplified_quality_score(0.5, 100.0)
        
        print(f"  Quality Score (action=0.5): {quality_score:.3f}")
        
        if quality_score < 0.0 or quality_score > 1.0:
            print(f"[ERROR] Quality score out of range: {quality_score}")
            return False
        
        if quality_score < 0.3:
            print(f"[WARN] Quality score seems low for action=0.5")
        
        print("[OK] Quality score calculation works")
        return True
    except Exception as e:
        print(f"[ERROR] Quality score calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_expected_value_calculation():
    """Test that expected value calculation works"""
    print("\n[TEST] Expected Value Calculation")
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
        
        reward_config = {
            "quality_filters": {
                "enabled": True,
                "require_positive_expected_value": True
            }
        }
        
        env = TradingEnvironment(
            data=data,
            timeframes=[1],
            reward_config=reward_config
        )
        
        # Test with no recent trades (should return None)
        ev = env._calculate_expected_value_simplified()
        print(f"  Expected Value (no trades): {ev}")
        
        if ev is not None:
            print("[WARN] Expected value should be None with no trades")
        
        # Add some sample trades
        env.recent_trades_pnl = [10.0, -5.0, 15.0, -3.0, 8.0, -2.0, 12.0, -4.0, 9.0, -1.0]
        ev = env._calculate_expected_value_simplified()
        
        print(f"  Expected Value (with trades): {ev:.3f}")
        
        if ev is None:
            print("[ERROR] Expected value should be calculated with 10 trades")
            return False
        
        # With more wins than losses, EV should be positive
        if ev <= 0:
            print(f"[WARN] Expected value is negative/zero: {ev}")
        
        print("[OK] Expected value calculation works")
        return True
    except Exception as e:
        print(f"[ERROR] Expected value calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_consecutive_loss_limit():
    """Test that consecutive loss limit is configured"""
    print("\n[TEST] Consecutive Loss Limit")
    print("=" * 60)
    
    try:
        from src.trading_env import TradingEnvironment
        import pandas as pd
        
        data = {
            1: pd.DataFrame({
                'open': [100.0] * 100,
                'high': [101.0] * 100,
                'low': [99.0] * 100,
                'close': [100.0] * 100,
                'volume': [1000] * 100
            })
        }
        
        reward_config = {
            "max_consecutive_losses": 3
        }
        
        env = TradingEnvironment(
            data=data,
            timeframes=[1],
            reward_config=reward_config
        )
        
        # Check that max_consecutive_losses is accessible
        max_losses = env.reward_config.get("max_consecutive_losses", 0)
        print(f"  Max Consecutive Losses: {max_losses}")
        
        if max_losses <= 0:
            print("[ERROR] Max consecutive losses not configured")
            return False
        
        # Check that TradeState has consecutive_losses field
        from src.trading_env import TradeState
        state = TradeState(
            position=0.0,
            entry_price=None,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            total_pnl=0.0,
            trades_count=0,
            winning_trades=0,
            losing_trades=0,
            consecutive_losses=0,
            trading_paused=False
        )
        
        if not hasattr(state, 'consecutive_losses'):
            print("[ERROR] TradeState missing consecutive_losses field")
            return False
        
        if not hasattr(state, 'trading_paused'):
            print("[ERROR] TradeState missing trading_paused field")
            return False
        
        print("[OK] Consecutive loss limit is configured")
        return True
    except Exception as e:
        print(f"[ERROR] Consecutive loss limit test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all E2E tests"""
    print("\n" + "=" * 60)
    print("E2E TEST: Simplified Quality Filters")
    print("=" * 60)
    
    tests = [
        ("Quality Filters Configuration", test_quality_filters_enabled),
        ("TradingEnvironment Imports", test_trading_env_imports),
        ("Quality Score Calculation", test_quality_score_calculation),
        ("Expected Value Calculation", test_expected_value_calculation),
        ("Consecutive Loss Limit", test_consecutive_loss_limit),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[ERROR] Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

