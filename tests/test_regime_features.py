"""
Test Regime Features Integration (Phase 1.4)

Tests:
- Regime detector initialization
- Regime feature extraction
- State dimension calculation
- Transfer learning compatibility
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Configure stdout for Windows Unicode support (only if not already wrapped)
if sys.platform == 'win32':
    import io
    try:
        if not isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except (AttributeError, ValueError):
        pass  # Already wrapped or can't wrap

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.regime_detector import RealTimeRegimeDetector
from src.trading_env import TradingEnvironment


def test_regime_detector_initialization():
    """Test that regime detector can be initialized"""
    print("\n[TEST] Regime Detector Initialization...")
    try:
        detector = RealTimeRegimeDetector(lookback_window=50)
        assert detector is not None
        assert detector.lookback_window == 50
        print("  [OK] Regime detector initialized successfully")
        return True
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        return False


def test_regime_detection():
    """Test regime detection with sample data"""
    print("\n[TEST] Regime Detection...")
    try:
        detector = RealTimeRegimeDetector(lookback_window=50)
        
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
        
        # Test detection
        result = detector.detect_regime(data, current_step=99)
        
        assert 'regime' in result
        assert 'confidence' in result
        assert 'duration' in result
        assert result['regime'] in ['trending', 'ranging', 'volatile']
        assert 0.0 <= result['confidence'] <= 1.0
        
        print(f"  [OK] Regime detected: {result['regime']} (confidence: {result['confidence']:.2f})")
        return True
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_regime_features_in_environment():
    """Test that regime features are added to environment state"""
    print("\n[TEST] Regime Features in Environment...")
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
        
        # Create environment WITH regime features
        env_with_regime = TradingEnvironment(
            data=data,
            timeframes=[1, 5, 15],
            reward_config={
                'include_regime_features': True
            }
        )
        
        # Create environment WITHOUT regime features
        env_without_regime = TradingEnvironment(
            data=data,
            timeframes=[1, 5, 15],
            reward_config={
                'include_regime_features': False
            }
        )
        
        # Check state dimensions
        assert env_with_regime.state_dim == env_without_regime.state_dim + 5, \
            f"Expected state_dim difference of 5, got {env_with_regime.state_dim - env_without_regime.state_dim}"
        
        print(f"  [OK] State dimension with regime: {env_with_regime.state_dim}")
        print(f"  [OK] State dimension without regime: {env_without_regime.state_dim}")
        print(f"  [OK] Difference: {env_with_regime.state_dim - env_without_regime.state_dim} (expected: 5)")
        
        # Test that regime detector is initialized
        assert env_with_regime.regime_detector is not None, "Regime detector should be initialized"
        assert env_without_regime.regime_detector is None, "Regime detector should NOT be initialized when disabled"
        
        print("  [OK] Regime detector initialization correct")
        
        # Test state extraction
        state, info = env_with_regime.reset()
        assert len(state) == env_with_regime.state_dim, \
            f"State length {len(state)} doesn't match state_dim {env_with_regime.state_dim}"
        
        # Check that last 5 features are regime features (should be non-zero if enough data)
        regime_features = state[-5:]
        print(f"  [OK] Regime features extracted: {regime_features}")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_dimension_calculation():
    """Test state dimension calculation with regime features"""
    print("\n[TEST] State Dimension Calculation...")
    try:
        # Base calculation: 15 features * 3 timeframes * 20 lookback = 900
        base_dim = 15 * 3 * 20
        
        # With regime features: 900 + 5 = 905
        expected_with_regime = base_dim + 5
        
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
        
        env = TradingEnvironment(
            data=data,
            timeframes=[1, 5, 15],
            reward_config={'include_regime_features': True}
        )
        
        assert env.state_dim == expected_with_regime, \
            f"Expected state_dim {expected_with_regime}, got {env.state_dim}"
        
        print(f"  [OK] State dimension: {env.state_dim} (expected: {expected_with_regime})")
        return True
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transfer_learning_compatibility():
    """Test that transfer learning can handle state dimension change (900 → 905)"""
    print("\n[TEST] Transfer Learning Compatibility...")
    try:
        from src.weight_transfer import transfer_checkpoint_weights
        from src.models import ActorNetwork, CriticNetwork
        
        # Create old architecture (state_dim=900)
        old_actor = ActorNetwork(state_dim=900, hidden_dims=[256, 256, 128])
        old_critic = CriticNetwork(state_dim=900, hidden_dims=[256, 256, 128])
        
        # Create new architecture (state_dim=905)
        new_actor = ActorNetwork(state_dim=905, hidden_dims=[256, 256, 128])
        new_critic = CriticNetwork(state_dim=905, hidden_dims=[256, 256, 128])
        
        # Create dummy checkpoint
        import tempfile
        checkpoint_path = tempfile.NamedTemporaryFile(suffix='.pt', delete=False).name
        
        checkpoint = {
            'actor_state_dict': old_actor.state_dict(),
            'critic_state_dict': old_critic.state_dict(),
            'state_dim': 900,
            'hidden_dims': [256, 256, 128],
            'timestep': 1000000
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Test transfer
        # Note: weight_transfer.py has emoji in print statements which may cause encoding issues
        # We'll redirect stdout to handle that
        import io
        import contextlib
        import os
        
        try:
            # Capture output to avoid emoji encoding issues
            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                new_actor_state, new_critic_state = transfer_checkpoint_weights(
                    checkpoint_path,
                    new_actor,
                    new_critic,
                    transfer_strategy='copy_and_extend'
                )
            
            # Verify weights were transferred
            assert new_actor_state is not None
            assert new_critic_state is not None
            
            # Verify first layer has correct input dimension (905)
            first_layer_weight = new_actor_state['feature_layers.0.weight']
            assert first_layer_weight.shape[1] == 905, \
                f"Expected input dim 905, got {first_layer_weight.shape[1]}"
            
            print("  [OK] Transfer learning works: 900 -> 905")
            print(f"  [OK] First layer input dim: {first_layer_weight.shape[1]}")
            
            # Cleanup
            os.unlink(checkpoint_path)
            
            return True
        except Exception as e:
            # Cleanup on error
            if os.path.exists(checkpoint_path):
                os.unlink(checkpoint_path)
            raise e
            
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all regime feature tests"""
    print("=" * 80)
    print("TESTING REGIME FEATURES (Phase 1.4)")
    print("=" * 80)
    
    tests = [
        ("Regime Detector Initialization", test_regime_detector_initialization),
        ("Regime Detection", test_regime_detection),
        ("Regime Features in Environment", test_regime_features_in_environment),
        ("State Dimension Calculation", test_state_dimension_calculation),
        ("Transfer Learning Compatibility", test_transfer_learning_compatibility),
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
        print("\n[SUCCESS] ALL TESTS PASSED - Regime features ready for training")
        return True
    else:
        print(f"\n[WARN] {total - passed} test(s) failed - Review before training")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

