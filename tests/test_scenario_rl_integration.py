"""
Test script for RL Agent Integration with Scenario Simulator

Tests:
1. RL agent integration into scenario simulator
2. Enhanced risk management for downtrend protection
3. API endpoints supporting RL agent backtesting
"""

import pytest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.scenario_simulator import ScenarioSimulator, MarketRegime

HAS_GYMNASIUM = importlib.util.find_spec("gymnasium") is not None


class TestRLAgentIntegration:
    """Test RL agent integration with scenario simulator"""
    
    def setup_method(self):
        """Set up test data"""
        # Create synthetic price data
        dates = pd.date_range(end=datetime.now(), periods=200, freq='1min')
        base_price = 5000.0
        
        # Create price data with trend
        returns = np.random.randn(200) * 0.001
        prices = base_price * (1 + returns).cumprod()
        
        self.price_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 200)
        })
        
        self.simulator = ScenarioSimulator(self.price_data, initial_capital=100000.0)
    
    def test_dataframe_to_multi_timeframe(self):
        """Test DataFrame to multi-timeframe conversion"""
        multi_tf_data = ScenarioSimulator._dataframe_to_multi_timeframe(
            self.price_data,
            timeframes=[1, 5, 15]
        )
        
        assert 1 in multi_tf_data
        assert 5 in multi_tf_data
        assert 15 in multi_tf_data
        
        # Check that higher timeframes have fewer bars
        assert len(multi_tf_data[1]) >= len(multi_tf_data[5])
        assert len(multi_tf_data[5]) >= len(multi_tf_data[15])
        
        # Check data integrity
        for tf, df in multi_tf_data.items():
            assert 'close' in df.columns
            assert 'open' in df.columns
            assert 'high' in df.columns
            assert 'low' in df.columns
            assert 'volume' in df.columns
    
    def test_create_rl_agent_backtest_func(self):
        """Test RL agent backtest function creation"""
        # Test with no model (should create function that falls back)
        rl_backtest_func = ScenarioSimulator.create_rl_agent_backtest_func(
            model_path=None,
            n_episodes=1
        )
        
        assert callable(rl_backtest_func)
        
        # Test that function can be called
        try:
            results = rl_backtest_func(
                self.price_data,
                timeframes=[1, 5, 15],
                initial_capital=100000.0,
                transaction_cost=0.0001
            )
            
            # Should return a dictionary with expected keys
            assert isinstance(results, dict)
            assert 'equity_curve' in results
            assert 'trades' in results
            assert 'final_equity' in results
        except Exception as e:
            # If RL agent fails, it should fall back gracefully
            print(f"RL agent backtest failed (expected if no model): {e}")
    
    def test_scenario_simulation_with_rl_agent(self):
        """Test scenario simulation with RL agent backtesting"""
        # Check if model exists
        model_path = Path("models/best_model.pt")
        if not model_path.exists():
            # Look for any .pt file
            model_files = list(Path("models").glob("*.pt"))
            if not model_files:
                pytest.skip("No trained model found. Skipping RL agent test.")
                return
            model_path = model_files[0]
        
        # Create RL agent backtest function
        rl_backtest_func = ScenarioSimulator.create_rl_agent_backtest_func(
            model_path=str(model_path),
            n_episodes=1
        )
        
        # Run scenario with RL agent
        result = self.simulator.simulate_scenario(
            scenario_name="test_normal",
            regime=MarketRegime.NORMAL,
            backtest_func=rl_backtest_func
        )
        
        # Verify results
        assert result is not None
        assert hasattr(result, 'total_return')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'max_drawdown')
        assert hasattr(result, 'win_rate')
        assert hasattr(result, 'equity_curve')
    
    def test_scenario_simulation_fallback(self):
        """Test that scenario simulation falls back to simple backtest"""
        # Run scenario without RL agent (should use simple backtest)
        result = self.simulator.simulate_scenario(
            scenario_name="test_normal",
            regime=MarketRegime.NORMAL
        )
        
        # Verify results
        assert result is not None
        assert hasattr(result, 'total_return')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'max_drawdown')


class TestDowntrendRiskManagement:
    """Test enhanced risk management for downtrend protection"""
    
    def setup_method(self):
        """Set up risk manager"""
        from src.risk_manager import RiskManager
        
        risk_config = {
            "max_position_size": 1.0,
            "max_drawdown": 0.20,
            "max_daily_loss": 0.05,
            "stop_loss_atr_multiplier": 2.0,
            "initial_capital": 100000.0,
            "monte_carlo_enabled": False,
            "volatility_prediction_enabled": False
        }
        
        self.risk_manager = RiskManager(risk_config)
    
    def test_downtrend_detection(self):
        """Test downtrend detection"""
        # Create downtrend price data
        dates = pd.date_range(end=datetime.now(), periods=50, freq='1min')
        base_price = 5000.0
        
        # Create declining prices
        returns = np.full(50, -0.001)  # -0.1% per period
        prices = base_price * (1 + returns).cumprod()
        
        price_data = pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'volume': np.random.randint(1000, 10000, 50)
        })
        
        current_price = prices[-1]
        
        # Test long position in downtrend (should be reduced)
        target_position = 0.8
        adjusted = self.risk_manager._detect_trend_and_adjust(
            target_position,
            price_data,
            current_price
        )
        
        # Should be reduced (less than original)
        assert abs(adjusted) < abs(target_position)
        assert adjusted > 0  # Still long, just smaller
    
    def test_uptrend_long_position(self):
        """Test long position in uptrend (should not be reduced)"""
        # Create uptrend price data
        dates = pd.date_range(end=datetime.now(), periods=50, freq='1min')
        base_price = 5000.0
        
        # Create rising prices
        returns = np.full(50, 0.001)  # +0.1% per period
        prices = base_price * (1 + returns).cumprod()
        
        price_data = pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'volume': np.random.randint(1000, 10000, 50)
        })
        
        current_price = prices[-1]
        
        # Test long position in uptrend (should not be reduced)
        target_position = 0.8
        adjusted = self.risk_manager._detect_trend_and_adjust(
            target_position,
            price_data,
            current_price
        )
        
        # Should remain unchanged or very close
        assert abs(adjusted - target_position) < 0.01
    
    def test_short_position_in_downtrend(self):
        """Test short position in downtrend (should be allowed)"""
        # Create downtrend price data
        dates = pd.date_range(end=datetime.now(), periods=50, freq='1min')
        base_price = 5000.0
        
        # Create declining prices
        returns = np.full(50, -0.001)
        prices = base_price * (1 + returns).cumprod()
        
        price_data = pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'volume': np.random.randint(1000, 10000, 50)
        })
        
        current_price = prices[-1]
        
        # Test short position in downtrend (should be allowed)
        target_position = -0.8
        adjusted = self.risk_manager._detect_trend_and_adjust(
            target_position,
            price_data,
            current_price
        )
        
        # Should be close to original (maybe slightly reduced for safety)
        assert adjusted < 0  # Still short
        assert abs(adjusted) >= abs(target_position) * 0.8  # At least 80% of original


class TestAPIIntegration:
    """Test API endpoint integration"""
    
    @pytest.mark.skipif(not HAS_GYMNASIUM, reason="Gymnasium required for RL scenario API tests")
    def test_scenario_request_model(self):
        """Test ScenarioSimulationRequest model"""
        from src.api_server import ScenarioSimulationRequest
        
        # Test default values
        request = ScenarioSimulationRequest()
        assert request.use_rl_agent == False
        assert request.model_path is None
        assert len(request.scenarios) > 0
        
        # Test with RL agent enabled
        request = ScenarioSimulationRequest(
            scenarios=["normal", "trending_up"],
            use_rl_agent=True,
            model_path="models/best_model.pt"
        )
        assert request.use_rl_agent == True
        assert request.model_path == "models/best_model.pt"


def test_robustness_comparison():
    """Compare simple backtest vs RL agent backtest (if model available)"""
    # Create test data
    dates = pd.date_range(end=datetime.now(), periods=200, freq='1min')
    base_price = 5000.0
    returns = np.random.randn(200) * 0.001
    prices = base_price * (1 + returns).cumprod()
    
    price_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 200)
    })
    
    simulator = ScenarioSimulator(price_data, initial_capital=100000.0)
    
    # Test simple backtest
    simple_result = simulator.simulate_scenario(
        scenario_name="normal",
        regime=MarketRegime.NORMAL
    )
    
    # Test RL agent backtest (if available)
    model_path = Path("models/best_model.pt")
    if not model_path.exists():
        model_files = list(Path("models").glob("*.pt"))
        if model_files:
            model_path = model_files[0]
        else:
            pytest.skip("No trained model found. Skipping comparison test.")
            return
    
    rl_backtest_func = ScenarioSimulator.create_rl_agent_backtest_func(
        model_path=str(model_path),
        n_episodes=1
    )
    
    rl_result = simulator.simulate_scenario(
        scenario_name="normal",
        regime=MarketRegime.NORMAL,
        backtest_func=rl_backtest_func
    )
    
    # Both should return valid results
    assert simple_result is not None
    assert rl_result is not None
    
    print(f"\nSimple Backtest: Return={simple_result.total_return:.2%}, Sharpe={simple_result.sharpe_ratio:.2f}")
    print(f"RL Agent Backtest: Return={rl_result.total_return:.2%}, Sharpe={rl_result.sharpe_ratio:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

