"""
E2E tests for Volatility Prediction functionality.

Tests the complete flow from API endpoints to volatility forecasting and adaptive position sizing.
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


class TestVolatilityPredictionAPI:
    """Test volatility prediction API endpoints"""
    
    def test_volatility_prediction_endpoint_exists(self, api_base_url, fastapi_available):
        """Test that volatility prediction endpoint exists"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        # Test endpoint exists (should return 422 if params missing, not 404)
        response = requests.post(
            f"{api_base_url}/api/volatility/predict",
            json={},
            timeout=5
        )
        assert response.status_code in [200, 422, 400], f"Expected 200, 422, or 400, got {response.status_code}"
    
    def test_volatility_prediction_basic_request(self, api_base_url, fastapi_available):
        """Test basic volatility prediction request"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        request_data = {
            "method": "adaptive",
            "lookback_periods": 252,
            "prediction_horizon": 1
        }
        
        response = requests.post(
            f"{api_base_url}/api/volatility/predict",
            json=request_data,
            timeout=30
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        
        # Check response structure
        assert "status" in data
        assert data["status"] == "success"
        assert "current_volatility" in data
        assert "predicted_volatility" in data
        assert "volatility_trend" in data
        assert "confidence" in data
        assert "volatility_percentile" in data
        assert "gap_risk_probability" in data
        assert "recommendations" in data
        
        # Validate metrics are numeric and in reasonable ranges
        assert isinstance(data["current_volatility"], (int, float))
        assert data["current_volatility"] >= 0
        assert isinstance(data["predicted_volatility"], (int, float))
        assert data["predicted_volatility"] >= 0
        assert data["volatility_trend"] in ["increasing", "decreasing", "stable"]
        assert 0 <= data["confidence"] <= 1
        assert 0 <= data["volatility_percentile"] <= 100
        assert 0 <= data["gap_risk_probability"] <= 1
    
    def test_volatility_prediction_different_methods(self, api_base_url, fastapi_available):
        """Test volatility prediction with different methods"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        methods = ["adaptive", "ewma", "historical_mean"]
        
        for method in methods:
            request_data = {
                "method": method,
                "lookback_periods": 100,
                "prediction_horizon": 1
            }
            
            response = requests.post(
                f"{api_base_url}/api/volatility/predict",
                json=request_data,
                timeout=30
            )
            
            assert response.status_code == 200, f"Method {method} failed: {response.status_code}"
            data = response.json()
            assert data["status"] == "success"
            assert data["prediction_method"] == method
    
    def test_volatility_prediction_multiple_horizons(self, api_base_url, fastapi_available):
        """Test that volatility prediction returns multiple horizon forecasts"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        request_data = {
            "method": "adaptive",
            "lookback_periods": 252,
            "prediction_horizon": 1
        }
        
        response = requests.post(
            f"{api_base_url}/api/volatility/predict",
            json=request_data,
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check all horizon forecasts are present
        assert "predicted_volatility" in data  # 1 period
        assert "predicted_volatility_5period" in data
        assert "predicted_volatility_20period" in data
        
        # All should be numeric and non-negative
        assert isinstance(data["predicted_volatility"], (int, float))
        assert isinstance(data["predicted_volatility_5period"], (int, float))
        assert isinstance(data["predicted_volatility_20period"], (int, float))
        assert data["predicted_volatility"] >= 0
        assert data["predicted_volatility_5period"] >= 0
        assert data["predicted_volatility_20period"] >= 0


class TestAdaptivePositionSizing:
    """Test adaptive position sizing based on volatility"""
    
    def test_adaptive_sizing_endpoint_exists(self, api_base_url, fastapi_available):
        """Test that adaptive sizing endpoint exists"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        # Test endpoint exists
        response = requests.post(
            f"{api_base_url}/api/volatility/adaptive-sizing",
            json={"base_position": 0.5},
            timeout=5
        )
        assert response.status_code in [200, 422, 400], f"Expected 200, 422, or 400, got {response.status_code}"
    
    def test_adaptive_sizing_basic_request(self, api_base_url, fastapi_available):
        """Test basic adaptive position sizing request"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        request_data = {
            "base_position": 0.5,
            "current_price": None
        }
        
        response = requests.post(
            f"{api_base_url}/api/volatility/adaptive-sizing",
            json=request_data,
            timeout=30
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        
        # Check response structure
        assert "status" in data
        assert data["status"] == "success"
        assert "base_position" in data
        assert "adjusted_position" in data
        assert "position_multiplier" in data
        assert "stop_loss_multiplier" in data
        assert "volatility_percentile" in data
        assert "volatility_trend" in data
        assert "recommendations" in data
        
        # Validate values
        assert abs(data["base_position"] - 0.5) < 0.01
        assert -1 <= data["adjusted_position"] <= 1
        assert 0.3 <= data["position_multiplier"] <= 1.2  # Clamped range
        assert 0.7 <= data["stop_loss_multiplier"] <= 2.0  # Clamped range
        assert 0 <= data["volatility_percentile"] <= 100
        assert data["volatility_trend"] in ["increasing", "decreasing", "stable"]
    
    def test_adaptive_sizing_reduces_high_volatility(self, api_base_url, fastapi_available):
        """Test that adaptive sizing reduces position in high volatility"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        request_data = {
            "base_position": 1.0,  # Maximum position
            "current_price": None
        }
        
        response = requests.post(
            f"{api_base_url}/api/volatility/adaptive-sizing",
            json=request_data,
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # If volatility is high (percentile > 80), position should be reduced
        if data["volatility_percentile"] > 80:
            assert abs(data["adjusted_position"]) < abs(data["base_position"])
            assert data["position_multiplier"] < 1.0
    
    def test_adaptive_sizing_different_positions(self, api_base_url, fastapi_available):
        """Test adaptive sizing with different base positions"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        positions = [0.2, 0.5, 0.8, 1.0]
        
        for base_pos in positions:
            request_data = {
                "base_position": base_pos,
                "current_price": None
            }
            
            response = requests.post(
                f"{api_base_url}/api/volatility/adaptive-sizing",
                json=request_data,
                timeout=30
            )
            
            assert response.status_code == 200
            data = response.json()
            assert abs(data["base_position"] - base_pos) < 0.01
            assert -1 <= data["adjusted_position"] <= 1


class TestVolatilityRecommendations:
    """Test volatility-based recommendations"""
    
    def test_recommendations_structure(self, api_base_url, fastapi_available):
        """Test that recommendations are properly structured"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        request_data = {
            "method": "adaptive",
            "lookback_periods": 252
        }
        
        response = requests.post(
            f"{api_base_url}/api/volatility/predict",
            json=request_data,
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        
        recommendations = data["recommendations"]
        assert isinstance(recommendations, dict)
        
        # Check for expected recommendation types
        expected_keys = ["position_sizing", "stop_loss", "trading_frequency", "risk_management"]
        for key in expected_keys:
            assert key in recommendations
            assert isinstance(recommendations[key], str)
            assert len(recommendations[key]) > 0


class TestVolatilityIntegration:
    """Test volatility prediction integration"""
    
    def test_volatility_prediction_response_structure(self, api_base_url, fastapi_available):
        """Test that volatility prediction response has all required fields"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        request_data = {
            "method": "adaptive",
            "lookback_periods": 100
        }
        
        response = requests.post(
            f"{api_base_url}/api/volatility/predict",
            json=request_data,
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check all required fields
        required_fields = [
            "status",
            "current_volatility",
            "predicted_volatility",
            "predicted_volatility_5period",
            "predicted_volatility_20period",
            "volatility_trend",
            "confidence",
            "volatility_percentile",
            "gap_risk_probability",
            "recommendations",
            "prediction_method"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
    
    def test_volatility_prediction_performance(self, api_base_url, fastapi_available):
        """Test that volatility prediction completes in reasonable time"""
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        import time
        
        request_data = {
            "method": "adaptive",
            "lookback_periods": 252
        }
        
        start_time = time.time()
        response = requests.post(
            f"{api_base_url}/api/volatility/predict",
            json=request_data,
            timeout=60
        )
        elapsed_time = time.time() - start_time
        
        assert response.status_code == 200
        # Should complete within 10 seconds
        assert elapsed_time < 10, f"Volatility prediction took {elapsed_time:.2f} seconds (too slow)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

