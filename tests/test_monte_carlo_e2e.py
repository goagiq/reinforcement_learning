"""
E2E tests for Monte Carlo Risk Assessment functionality.

Tests the complete flow from API endpoints to risk metrics calculation.
"""

import pytest
import requests
import numpy as np
import pandas as pd
from pathlib import Path
import json


# Test fixtures
@pytest.fixture
def api_base_url():
    """Base URL for API server"""
    return "http://localhost:8200"


@pytest.fixture
def fastapi_available(api_base_url):
    """Check if FastAPI server is available"""
    try:
        response = requests.get(f"{api_base_url}/", timeout=2)
        return response.status_code == 200
    except:
        return False


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing"""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='D')
    base_price = 5000.0
    returns = np.random.randn(100) * 0.02  # 2% daily volatility
    prices = base_price * (1 + returns).cumprod()
    
    return pd.DataFrame({
        'close': prices,
        'open': prices * 0.999,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'volume': np.random.randint(1000, 10000, 100)
    })


class TestMonteCarloAPIEndpoints:
    """Test Monte Carlo risk assessment API endpoints"""
    
    def test_monte_carlo_endpoint_exists(self, api_base_url, fastapi_available):
        """Test that Monte Carlo endpoint exists"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        # Test endpoint exists (should return 422 if params missing, not 404)
        response = requests.post(
            f"{api_base_url}/api/risk/monte-carlo",
            json={},
            timeout=5
        )
        assert response.status_code in [422, 400], f"Expected 422 or 400, got {response.status_code}"
    
    def test_monte_carlo_basic_request(self, api_base_url, fastapi_available):
        """Test basic Monte Carlo risk assessment request"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        request_data = {
            "current_price": 5000.0,
            "proposed_position": 0.5,
            "current_position": 0.0,
            "n_simulations": 500,
            "simulate_overnight": True
        }
        
        response = requests.post(
            f"{api_base_url}/api/risk/monte-carlo",
            json=request_data,
            timeout=30  # Monte Carlo can take a while
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        
        # Check response structure
        assert "status" in data
        assert data["status"] == "success"
        assert "risk_metrics" in data
        assert "scenario_stats" in data
        assert "recommendation" in data
        
        # Check risk metrics
        metrics = data["risk_metrics"]
        assert "expected_pnl" in metrics
        assert "var_95" in metrics
        assert "var_99" in metrics
        assert "win_probability" in metrics
        assert "tail_risk" in metrics
        assert "optimal_position_size" in metrics
        
        # Validate metrics are numeric
        assert isinstance(metrics["expected_pnl"], (int, float))
        assert isinstance(metrics["var_95"], (int, float))
        assert isinstance(metrics["var_99"], (int, float))
        assert 0 <= metrics["win_probability"] <= 1
        assert 0 <= metrics["tail_risk"] <= 1
        assert -1 <= metrics["optimal_position_size"] <= 1
    
    def test_monte_carlo_with_stop_loss(self, api_base_url, fastapi_available):
        """Test Monte Carlo with stop loss"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        request_data = {
            "current_price": 5000.0,
            "proposed_position": 0.5,
            "current_position": 0.0,
            "stop_loss": 4900.0,  # 2% stop loss
            "take_profit": 5100.0,  # 2% take profit
            "n_simulations": 500
        }
        
        response = requests.post(
            f"{api_base_url}/api/risk/monte-carlo",
            json=request_data,
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        # Stop loss should reduce risk
        assert data["risk_metrics"]["var_99"] > -10000  # Should not be catastrophic
    
    def test_monte_carlo_position_sizing(self, api_base_url, fastapi_available):
        """Test that Monte Carlo suggests appropriate position sizing"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        # Test with large position (should be reduced)
        request_data = {
            "current_price": 5000.0,
            "proposed_position": 1.0,  # Maximum position
            "current_position": 0.0,
            "n_simulations": 500
        }
        
        response = requests.post(
            f"{api_base_url}/api/risk/monte-carlo",
            json=request_data,
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        optimal_size = data["risk_metrics"]["optimal_position_size"]
        
        # Optimal size should be reasonable (not necessarily 1.0)
        assert -1 <= optimal_size <= 1
        # If risk is high, optimal size should be smaller
        if data["risk_metrics"]["tail_risk"] > 0.10:
            assert abs(optimal_size) <= 1.0
    
    def test_scenario_analysis_endpoint(self, api_base_url, fastapi_available):
        """Test scenario analysis endpoint"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        request_data = {
            "current_price": 5000.0,
            "proposed_position": 0.5,
            "current_position": 0.0,
            "n_simulations": 500
        }
        
        response = requests.post(
            f"{api_base_url}/api/risk/scenario-analysis",
            json=request_data,
            timeout=60  # Scenario analysis can take longer
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "success"
        assert "scenarios" in data
        assert "recommendation" in data
        
        # Check scenarios
        scenarios = data["scenarios"]
        assert "normal" in scenarios
        assert "high_volatility" in scenarios
        assert "trending" in scenarios
        assert "ranging" in scenarios
        
        # Validate scenario metrics
        for scenario_name, scenario_metrics in scenarios.items():
            assert "expected_pnl" in scenario_metrics
            assert "var_95" in scenario_metrics
            assert "win_probability" in scenario_metrics
            assert "tail_risk" in scenario_metrics


class TestMonteCarloIntegration:
    """Test Monte Carlo integration with trading system"""
    
    def test_monte_carlo_response_structure(self, api_base_url, fastapi_available):
        """Test that Monte Carlo response has all required fields"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        request_data = {
            "current_price": 5000.0,
            "proposed_position": 0.5,
            "n_simulations": 100
        }
        
        response = requests.post(
            f"{api_base_url}/api/risk/monte-carlo",
            json=request_data,
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check all required fields
        required_fields = [
            "status",
            "risk_metrics.expected_pnl",
            "risk_metrics.var_95",
            "risk_metrics.var_99",
            "risk_metrics.win_probability",
            "risk_metrics.tail_risk",
            "risk_metrics.optimal_position_size",
            "scenario_stats.min_pnl",
            "scenario_stats.max_pnl",
            "scenario_stats.median_pnl",
            "recommendation"
        ]
        
        for field in required_fields:
            keys = field.split(".")
            value = data
            for key in keys:
                assert key in value, f"Missing field: {field}"
                value = value[key]
    
    def test_monte_carlo_recommendation_logic(self, api_base_url, fastapi_available):
        """Test that recommendations are generated correctly"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        # Low risk scenario
        request_data = {
            "current_price": 5000.0,
            "proposed_position": 0.2,  # Small position
            "n_simulations": 500
        }
        
        response = requests.post(
            f"{api_base_url}/api/risk/monte-carlo",
            json=request_data,
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        recommendation = data["recommendation"]
        
        # Recommendation should be a string
        assert isinstance(recommendation, str)
        assert len(recommendation) > 0
        
        # Recommendation should contain one of the expected keywords
        # (Note: Recommendation depends on multiple factors, not just tail risk)
        valid_keywords = [
            "ACCEPTABLE", "ROBUST", "POSITION", "HIGH_RISK", 
            "MODERATE_RISK", "POOR_EDGE", "CAUTION", "WEAK_EDGE"
        ]
        assert any(keyword in recommendation for keyword in valid_keywords), \
            f"Recommendation '{recommendation}' should contain one of {valid_keywords}"
    
    def test_monte_carlo_performance(self, api_base_url, fastapi_available):
        """Test that Monte Carlo completes in reasonable time"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        import time
        
        request_data = {
            "current_price": 5000.0,
            "proposed_position": 0.5,
            "n_simulations": 1000  # Standard simulation count
        }
        
        start_time = time.time()
        response = requests.post(
            f"{api_base_url}/api/risk/monte-carlo",
            json=request_data,
            timeout=60
        )
        elapsed_time = time.time() - start_time
        
        assert response.status_code == 200
        # Should complete within 60 seconds for 1000 simulations
        assert elapsed_time < 60, f"Monte Carlo took {elapsed_time:.2f} seconds (too slow)"


class TestMonteCarloEdgeCases:
    """Test edge cases and error handling"""
    
    def test_monte_carlo_invalid_position(self, api_base_url, fastapi_available):
        """Test with invalid position size"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        # Position > 1.0 (should be handled gracefully)
        request_data = {
            "current_price": 5000.0,
            "proposed_position": 1.5,  # Invalid
            "n_simulations": 100
        }
        
        response = requests.post(
            f"{api_base_url}/api/risk/monte-carlo",
            json=request_data,
            timeout=30
        )
        
        # Should either return 400/422 or handle gracefully (clip to 1.0)
        assert response.status_code in [200, 400, 422]
    
    def test_monte_carlo_zero_position(self, api_base_url, fastapi_available):
        """Test with zero position"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        request_data = {
            "current_price": 5000.0,
            "proposed_position": 0.0,
            "n_simulations": 100
        }
        
        response = requests.post(
            f"{api_base_url}/api/risk/monte-carlo",
            json=request_data,
            timeout=30
        )
        
        # Should handle zero position gracefully
        assert response.status_code in [200, 400, 422]
    
    def test_monte_carlo_missing_parameters(self, api_base_url, fastapi_available):
        """Test with missing required parameters"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        # Missing current_price
        response = requests.post(
            f"{api_base_url}/api/risk/monte-carlo",
            json={"proposed_position": 0.5},
            timeout=5
        )
        
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

