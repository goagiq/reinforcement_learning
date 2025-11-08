"""
E2E tests for Scenario Simulation functionality.

Tests robustness testing, stress testing, and parameter sensitivity analysis.
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


class TestRobustnessTestAPI:
    """Test robustness testing API endpoints"""
    
    def test_robustness_test_endpoint_exists(self, api_base_url, fastapi_available):
        """Test that robustness test endpoint exists"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        # Test endpoint exists
        response = requests.post(
            f"{api_base_url}/api/scenario/robustness-test",
            json={"scenarios": ["normal"]},
            timeout=5
        )
        assert response.status_code in [200, 422, 400], f"Expected 200, 422, or 400, got {response.status_code}"
    
    def test_robustness_test_basic_request(self, api_base_url, fastapi_available):
        """Test basic robustness test request"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        request_data = {
            "scenarios": ["normal", "trending_up", "high_volatility"],
            "intensity": 1.0
        }
        
        response = requests.post(
            f"{api_base_url}/api/scenario/robustness-test",
            json=request_data,
            timeout=60
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        
        # Check response structure
        assert "status" in data
        assert data["status"] == "success"
        assert "scenarios" in data
        assert "summary" in data
        
        # Check scenarios
        assert len(data["scenarios"]) > 0
        for scenario in data["scenarios"]:
            assert "scenario_name" in scenario
            assert "total_return" in scenario
            assert "sharpe_ratio" in scenario
            assert "max_drawdown" in scenario
            assert "win_rate" in scenario
        
        # Check summary
        summary = data["summary"]
        assert "total_scenarios" in summary
        assert "average_return" in summary
        assert "worst_drawdown" in summary
    
    def test_robustness_test_multiple_scenarios(self, api_base_url, fastapi_available):
        """Test robustness test with multiple scenarios"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        scenarios = [
            "normal", "trending_up", "trending_down", "ranging",
            "high_volatility", "low_volatility"
        ]
        
        request_data = {
            "scenarios": scenarios,
            "intensity": 1.0
        }
        
        response = requests.post(
            f"{api_base_url}/api/scenario/robustness-test",
            json=request_data,
            timeout=120
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["scenarios"]) == len(scenarios)


class TestStressTestAPI:
    """Test stress testing API endpoints"""
    
    def test_stress_test_endpoint_exists(self, api_base_url, fastapi_available):
        """Test that stress test endpoint exists"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        response = requests.post(
            f"{api_base_url}/api/scenario/stress-test",
            json={"scenarios": ["crash"]},
            timeout=5
        )
        assert response.status_code in [200, 422, 400]
    
    def test_stress_test_basic_request(self, api_base_url, fastapi_available):
        """Test basic stress test request"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        request_data = {
            "scenarios": ["crash", "flash_crash"],
            "intensity": 2.0
        }
        
        response = requests.post(
            f"{api_base_url}/api/scenario/stress-test",
            json=request_data,
            timeout=60
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "success"
        assert "stress_tests" in data
        assert "summary" in data
        
        # Check stress test results
        assert len(data["stress_tests"]) > 0
        for test in data["stress_tests"]:
            assert "scenario_name" in test
            assert "max_drawdown" in test
            assert "survived" in test
            assert "recovery_time" in test
        
        # Check summary
        summary = data["summary"]
        assert "total_tests" in summary
        assert "survived_count" in summary
        assert "worst_drawdown" in summary
    
    def test_stress_test_survival_logic(self, api_base_url, fastapi_available):
        """Test that stress test correctly identifies survival"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        request_data = {
            "scenarios": ["crash"],
            "intensity": 2.0
        }
        
        response = requests.post(
            f"{api_base_url}/api/scenario/stress-test",
            json=request_data,
            timeout=60
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check that survived is a boolean
        for test in data["stress_tests"]:
            assert isinstance(test["survived"], bool)


class TestParameterSensitivityAPI:
    """Test parameter sensitivity analysis API endpoints"""
    
    def test_parameter_sensitivity_endpoint_exists(self, api_base_url, fastapi_available):
        """Test that parameter sensitivity endpoint exists"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        response = requests.post(
            f"{api_base_url}/api/scenario/parameter-sensitivity",
            json={
                "parameter_name": "test_param",
                "parameter_values": [0.1, 0.2, 0.3]
            },
            timeout=5
        )
        assert response.status_code in [200, 422, 400]
    
    def test_parameter_sensitivity_basic_request(self, api_base_url, fastapi_available):
        """Test basic parameter sensitivity request"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        request_data = {
            "parameter_name": "position_size",
            "parameter_values": [0.2, 0.4, 0.6, 0.8, 1.0],
            "base_parameters": {},
            "regime": "normal"
        }
        
        response = requests.post(
            f"{api_base_url}/api/scenario/parameter-sensitivity",
            json=request_data,
            timeout=60
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "success"
        assert "parameter_name" in data
        assert "parameter_values" in data
        assert "performance_metrics" in data
        assert "optimal_value" in data
        assert "sensitivity_score" in data
        assert "recommendations" in data
        
        # Validate structure
        assert len(data["parameter_values"]) == 5
        assert "total_return" in data["performance_metrics"]
        assert "sharpe_ratio" in data["performance_metrics"]
        assert 0 <= data["sensitivity_score"] <= 1
        assert isinstance(data["optimal_value"], (int, float))
    
    def test_parameter_sensitivity_different_regimes(self, api_base_url, fastapi_available):
        """Test parameter sensitivity with different regimes"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        regimes = ["normal", "trending_up", "high_volatility"]
        
        for regime in regimes:
            request_data = {
                "parameter_name": "test_param",
                "parameter_values": [0.5, 1.0],
                "regime": regime
            }
            
            response = requests.post(
                f"{api_base_url}/api/scenario/parameter-sensitivity",
                json=request_data,
                timeout=60
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"


class TestScenarioIntegration:
    """Test scenario simulation integration"""
    
    def test_robustness_test_response_structure(self, api_base_url, fastapi_available):
        """Test that robustness test response has all required fields"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        request_data = {
            "scenarios": ["normal", "high_volatility"],
            "intensity": 1.0
        }
        
        response = requests.post(
            f"{api_base_url}/api/scenario/robustness-test",
            json=request_data,
            timeout=60
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check all required fields in scenarios
        for scenario in data["scenarios"]:
            required_fields = [
                "scenario_name", "market_regime", "total_return",
                "sharpe_ratio", "max_drawdown", "win_rate",
                "profit_factor", "total_trades", "volatility"
            ]
            for field in required_fields:
                assert field in scenario, f"Missing field: {field}"
    
    def test_scenario_performance(self, api_base_url, fastapi_available):
        """Test that scenario simulation completes in reasonable time"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        import time
        
        request_data = {
            "scenarios": ["normal"],
            "intensity": 1.0
        }
        
        start_time = time.time()
        response = requests.post(
            f"{api_base_url}/api/scenario/robustness-test",
            json=request_data,
            timeout=120
        )
        elapsed_time = time.time() - start_time
        
        assert response.status_code == 200
        # Should complete within 60 seconds for single scenario
        assert elapsed_time < 60, f"Scenario test took {elapsed_time:.2f} seconds (too slow)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

