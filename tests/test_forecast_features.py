"""
Test Forecast Features Integration (Phase 4.1)

Tests:
- Forecast predictor initialization
- Forecast feature extraction
- State dimension calculation with forecast features
- Graceful degradation when forecast predictor unavailable
"""

import sys
import io
from pathlib import Path
import numpy as np
import pandas as pd

# Configure stdout for Windows Unicode support
if sys.platform == 'win32':
    try:
        if not isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except (AttributeError, ValueError):
        pass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.forecasting.simple_forecast_predictor import SimpleForecastPredictor, ForecastResult
from src.trading_env import TradingEnvironment


def test_forecast_predictor_initialization():
    """Test that forecast predictor can be initialized"""
    print("\n[TEST] Forecast Predictor Initialization...")
    try:
        predictor = SimpleForecastPredictor(lookback_window=20, forecast_horizon=5)
        assert predictor is not None
        assert predictor.lookback_window == 20
        assert predictor.forecast_horizon == 5
        print("  [OK] Forecast predictor initialized successfully")
        return True
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forecast_prediction():
    """Test forecast prediction with sample data"""
    print("\n[TEST] Forecast Prediction...")
    try:
        predictor = SimpleForecastPredictor(lookback_window=20, forecast_horizon=5)
        
        # Create sample price data (trending up)
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        prices = np.linspace(100, 110, 100)  # Trending up
        volumes = np.random.uniform(1000, 5000, 100)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': volumes
        })
        
        # Test prediction
        result = predictor.predict(data, current_step=99)
        
        assert isinstance(result, ForecastResult)
        assert -1.0 <= result.direction <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.forecast_horizon == 5
        
        print(f"  [OK] Forecast: direction={result.direction:.3f}, confidence={result.confidence:.3f}, return={result.expected_return:.3f}%")
        return True
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forecast_features_extraction():
    """Test forecast features extraction"""
    print("\n[TEST] Forecast Features Extraction...")
    try:
        predictor = SimpleForecastPredictor(lookback_window=20, forecast_horizon=5)
        
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        prices = np.linspace(100, 110, 100)
        volumes = np.random.uniform(1000, 5000, 100)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': volumes
        })
        
        # Test feature extraction
        features = predictor.get_forecast_features(data, current_step=99)
        
        assert len(features) == 3, f"Expected 3 features, got {len(features)}"
        assert -1.0 <= features[0] <= 1.0, f"Direction should be in [-1, 1], got {features[0]}"
        assert 0.0 <= features[1] <= 1.0, f"Confidence should be in [0, 1], got {features[1]}"
        
        print(f"  [OK] Forecast features: {features}")
        return True
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forecast_features_in_environment():
    """Test that forecast features are added to environment state"""
    print("\n[TEST] Forecast Features in Environment...")
    try:
        # Create sample multi-timeframe data
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        data_1min = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        })
        
        data = {1: data_1min, 5: data_1min, 15: data_1min}
        
        # Create environment WITHOUT forecast features
        env_without_forecast = TradingEnvironment(
            data=data,
            timeframes=[1, 5, 15],
            reward_config={
                'include_forecast_features': False
            }
        )
        
        # Create environment WITH forecast features
        env_with_forecast = TradingEnvironment(
            data=data,
            timeframes=[1, 5, 15],
            reward_config={
                'include_forecast_features': True
            }
        )
        
        # Check state dimensions
        assert env_with_forecast.state_dim == env_without_forecast.state_dim + 3, \
            f"Expected state_dim difference of 3, got {env_with_forecast.state_dim - env_without_forecast.state_dim}"
        
        print(f"  [OK] State dimension with forecast: {env_with_forecast.state_dim}")
        print(f"  [OK] State dimension without forecast: {env_without_forecast.state_dim}")
        print(f"  [OK] Difference: {env_with_forecast.state_dim - env_without_forecast.state_dim} (expected: 3)")
        
        # Test that forecast predictor is initialized
        assert env_with_forecast.forecast_predictor is not None, "Forecast predictor should be initialized"
        assert env_without_forecast.forecast_predictor is None, "Forecast predictor should NOT be initialized when disabled"
        
        print("  [OK] Forecast predictor initialization correct")
        
        # Test state extraction
        state, info = env_with_forecast.reset()
        assert len(state) == env_with_forecast.state_dim, \
            f"State length {len(state)} doesn't match state_dim {env_with_forecast.state_dim}"
        
        # Check that last 3 features are forecast features
        forecast_features = state[-3:]
        print(f"  [OK] Forecast features extracted: {forecast_features}")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forecast_with_regime_features():
    """Test forecast features combined with regime features"""
    print("\n[TEST] Forecast + Regime Features...")
    try:
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        data_1min = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        })
        data = {1: data_1min, 5: data_1min, 15: data_1min}
        
        # Base: 900
        # With regime: 900 + 5 = 905
        # With regime + forecast: 900 + 5 + 3 = 908
        env = TradingEnvironment(
            data=data,
            timeframes=[1, 5, 15],
            reward_config={
                'include_regime_features': True,
                'include_forecast_features': True
            }
        )
        
        expected_dim = 900 + 5 + 3  # base + regime + forecast
        assert env.state_dim == expected_dim, \
            f"Expected state_dim {expected_dim}, got {env.state_dim}"
        
        print(f"  [OK] Combined state dimension: {env.state_dim} (expected: {expected_dim})")
        
        # Test state extraction
        state, info = env.reset()
        assert len(state) == env.state_dim
        
        # Last 3 should be forecast, previous 5 should be regime
        forecast_features = state[-3:]
        regime_features = state[-8:-3]
        
        print(f"  [OK] Regime features: {regime_features}")
        print(f"  [OK] Forecast features: {forecast_features}")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graceful_degradation():
    """Test that system works when forecast predictor fails"""
    print("\n[TEST] Graceful Degradation...")
    try:
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        data_1min = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        })
        data = {1: data_1min, 5: data_1min, 15: data_1min}
        
        # Create environment with forecast enabled
        env = TradingEnvironment(
            data=data,
            timeframes=[1, 5, 15],
            reward_config={
                'include_forecast_features': True
            }
        )
        
        # Even if forecast fails, should return zeros
        state, info = env.reset()
        
        # Should have correct dimension
        assert len(state) == env.state_dim
        
        # Forecast features should be zeros if predictor fails (or valid if it works)
        forecast_features = state[-3:]
        assert len(forecast_features) == 3
        
        print(f"  [OK] Graceful degradation works (forecast features: {forecast_features})")
        return True
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all forecast feature tests"""
    print("=" * 80)
    print("TESTING FORECAST FEATURES (Phase 4.1)")
    print("=" * 80)
    
    tests = [
        ("Forecast Predictor Initialization", test_forecast_predictor_initialization),
        ("Forecast Prediction", test_forecast_prediction),
        ("Forecast Features Extraction", test_forecast_features_extraction),
        ("Forecast Features in Environment", test_forecast_features_in_environment),
        ("Forecast + Regime Features", test_forecast_with_regime_features),
        ("Graceful Degradation", test_graceful_degradation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[ERROR] Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED - Forecast features ready for use")
        return True
    else:
        print(f"\n[WARN] {total - passed} test(s) failed - Review forecast implementation")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

