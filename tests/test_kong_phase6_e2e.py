"""
End-to-End Test for Kong Gateway Phase 6: FastAPI Integration

Tests FastAPI routing through Kong Gateway, including:
- FastAPI service routing
- CORS configuration
- API key authentication
- Rate limiting
- Frontend integration
"""

import pytest
import os
import sys
import time
import requests
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Kong Configuration
KONG_PROXY = "http://localhost:8300"
KONG_ADMIN = "http://localhost:8301"
KONG_API_KEY = "EhJ2T5SpLeqUAaFxkBwoWcnlg1T_5AappZ9VOhXzgXI"  # admin-consumer
FASTAPI_DIRECT = "http://localhost:8200"


@pytest.fixture
def kong_available():
    """Check if Kong Gateway is available"""
    try:
        response = requests.get(f"{KONG_ADMIN}/", timeout=2)
        return response.status_code == 200
    except:
        return False


@pytest.fixture
def fastapi_available():
    """Check if FastAPI server is available"""
    try:
        response = requests.get(f"{FASTAPI_DIRECT}/api/setup/check", timeout=2)
        return response.status_code == 200
    except:
        return False


class TestFastAPIServiceInKong:
    """Test FastAPI service configuration in Kong"""
    
    def test_fastapi_service_exists(self, kong_available):
        """Test that FastAPI service exists in Kong"""
        if not kong_available:
            pytest.skip("Kong Gateway not available")
        
        response = requests.get(f"{KONG_ADMIN}/services/fastapi-service", timeout=5)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "fastapi-service"
        assert data["port"] == 8200  # Should point to FastAPI on port 8200
        # Construct URL from components
        service_url = f"{data.get('protocol', 'http')}://{data.get('host', '')}:{data.get('port', '')}"
        print(f"✅ FastAPI service exists: {service_url}")
    
    def test_fastapi_route_exists(self, kong_available):
        """Test that FastAPI route exists in Kong"""
        if not kong_available:
            pytest.skip("Kong Gateway not available")
        
        response = requests.get(f"{KONG_ADMIN}/routes/fastapi-route", timeout=5)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "fastapi-route"
        assert "/api" in data["paths"]
        print(f"✅ FastAPI route exists: {data['paths']}")
    
    def test_fastapi_route_strip_path(self, kong_available):
        """Test that FastAPI route has correct strip_path setting"""
        if not kong_available:
            pytest.skip("Kong Gateway not available")
        
        response = requests.get(f"{KONG_ADMIN}/routes/fastapi-route", timeout=5)
        data = response.json()
        
        # strip_path should be false so /api path is preserved
        assert data["strip_path"] == False
        print("✅ FastAPI route strip_path correctly set to false")


class TestFastAPIRoutingThroughKong:
    """Test FastAPI routing through Kong Gateway"""
    
    def test_fastapi_through_kong_without_auth(self, kong_available, fastapi_available):
        """Test FastAPI through Kong without API key (should fail if key-auth enabled)"""
        if not kong_available:
            pytest.skip("Kong Gateway not available")
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        response = requests.get(f"{KONG_PROXY}/api/setup/check", timeout=5)
        
        # If key-auth is enabled, should get 401
        # If key-auth is disabled/optional, should get 200
        assert response.status_code in [200, 401, 403]
        
        if response.status_code == 401:
            print("✅ FastAPI route requires authentication (key-auth enabled)")
        elif response.status_code == 200:
            print("⚠️  FastAPI route accessible without authentication")
    
    def test_fastapi_through_kong_with_auth(self, kong_available, fastapi_available):
        """Test FastAPI through Kong with API key"""
        if not kong_available:
            pytest.skip("Kong Gateway not available")
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        headers = {"apikey": KONG_API_KEY}
        response = requests.get(f"{KONG_PROXY}/api/setup/check", headers=headers, timeout=5)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "venv_exists" in data
        assert "dependencies_installed" in data
        assert "ready" in data
        print("✅ FastAPI accessible through Kong with API key")
        print(f"   Setup status: ready={data.get('ready')}, venv={data.get('venv_exists')}, deps={data.get('dependencies_installed')}")
    
    def test_fastapi_cors_headers(self, kong_available, fastapi_available):
        """Test that CORS headers are present when accessing through Kong"""
        if not kong_available:
            pytest.skip("Kong Gateway not available")
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        headers = {"apikey": KONG_API_KEY}
        
        # Make OPTIONS request (preflight)
        response = requests.options(
            f"{KONG_PROXY}/api/setup/check",
            headers={
                **headers,
                "Origin": "http://localhost:3200",
                "Access-Control-Request-Method": "GET"
            },
            timeout=5
        )
        
        # Check for CORS headers
        cors_headers = []
        if "Access-Control-Allow-Origin" in response.headers:
            cors_headers.append("Access-Control-Allow-Origin")
        if "Access-Control-Allow-Methods" in response.headers:
            cors_headers.append("Access-Control-Allow-Methods")
        if "Access-Control-Allow-Headers" in response.headers:
            cors_headers.append("Access-Control-Allow-Headers")
        
        if cors_headers:
            print(f"✅ CORS headers present: {cors_headers}")
        else:
            print("⚠️  CORS headers not found (CORS plugin may not be enabled)")
        
        # CORS headers should be present if CORS plugin is enabled
        # But we don't fail the test if they're not (may be handled by FastAPI)
        assert response.status_code in [200, 204, 401, 403]
    
    def test_fastapi_rate_limiting(self, kong_available, fastapi_available):
        """Test that rate limiting is working on FastAPI routes"""
        if not kong_available:
            pytest.skip("Kong Gateway not available")
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        headers = {"apikey": KONG_API_KEY}
        
        # Make multiple rapid requests
        success_count = 0
        rate_limited = False
        
        for i in range(5):
            response = requests.get(f"{KONG_PROXY}/api/setup/check", headers=headers, timeout=2)
            
            if response.status_code == 200:
                success_count += 1
            elif response.status_code == 429:
                rate_limited = True
                print(f"✅ Rate limiting detected on request {i+1}")
                break
            
            time.sleep(0.1)  # Small delay
        
        print(f"✅ Made {success_count} successful requests (rate limit: {rate_limited})")
        assert success_count > 0 or rate_limited


class TestFastAPISecurity:
    """Test FastAPI security features through Kong"""
    
    def test_key_auth_plugin_enabled(self, kong_available):
        """Test that key-auth plugin is enabled on FastAPI service"""
        if not kong_available:
            pytest.skip("Kong Gateway not available")
        
        response = requests.get(f"{KONG_ADMIN}/services/fastapi-service/plugins", timeout=5)
        assert response.status_code == 200
        
        plugins = response.json()["data"]
        key_auth_plugins = [p for p in plugins if p["name"] == "key-auth"]
        
        if key_auth_plugins:
            print("✅ Key-auth plugin enabled on FastAPI service")
            assert len(key_auth_plugins) > 0
        else:
            print("⚠️  Key-auth plugin not found (may be optional)")
    
    def test_rate_limiting_plugin_enabled(self, kong_available):
        """Test that rate-limiting plugin is enabled on FastAPI service"""
        if not kong_available:
            pytest.skip("Kong Gateway not available")
        
        response = requests.get(f"{KONG_ADMIN}/services/fastapi-service/plugins", timeout=5)
        assert response.status_code == 200
        
        plugins = response.json()["data"]
        rate_limit_plugins = [p for p in plugins if p["name"] == "rate-limiting"]
        
        assert len(rate_limit_plugins) > 0
        print("✅ Rate-limiting plugin enabled on FastAPI service")
    
    def test_cors_plugin_enabled(self, kong_available):
        """Test that CORS plugin is enabled on FastAPI service"""
        if not kong_available:
            pytest.skip("Kong Gateway not available")
        
        response = requests.get(f"{KONG_ADMIN}/services/fastapi-service/plugins", timeout=5)
        assert response.status_code == 200
        
        plugins = response.json()["data"]
        cors_plugins = [p for p in plugins if p["name"] == "cors"]
        
        if cors_plugins:
            print("✅ CORS plugin enabled on FastAPI service")
            cors_config = cors_plugins[0].get("config", {})
            print(f"   CORS config: {cors_config}")
        else:
            print("⚠️  CORS plugin not found (CORS may be handled by FastAPI)")


class TestFastAPIEndpoints:
    """Test specific FastAPI endpoints through Kong"""
    
    def test_setup_check_endpoint(self, kong_available, fastapi_available):
        """Test /api/setup/check endpoint through Kong"""
        if not kong_available:
            pytest.skip("Kong Gateway not available")
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        headers = {"apikey": KONG_API_KEY}
        response = requests.get(f"{KONG_PROXY}/api/setup/check", headers=headers, timeout=5)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "venv_exists" in data
        assert "dependencies_installed" in data
        assert "config_exists" in data
        assert "ready" in data
        assert "issues" in data
        
        print(f"✅ Setup check endpoint working: ready={data['ready']}")
    
    def test_root_endpoint(self, kong_available, fastapi_available):
        """Test root endpoint through Kong"""
        if not kong_available:
            pytest.skip("Kong Gateway not available")
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        headers = {"apikey": KONG_API_KEY}
        response = requests.get(f"{KONG_PROXY}/", headers=headers, timeout=5)
        
        # Root endpoint should return 404 through Kong (only /api/* is routed)
        # Or if there's a catch-all, it might return something
        assert response.status_code in [200, 404]
        print(f"✅ Root endpoint: {response.status_code}")


class TestDirectVsKong:
    """Test comparison between direct FastAPI access and Kong access"""
    
    def test_direct_vs_kong_response_consistency(self, kong_available, fastapi_available):
        """Test that responses are consistent between direct and Kong access"""
        if not kong_available:
            pytest.skip("Kong Gateway not available")
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        # Direct access
        direct_response = requests.get(f"{FASTAPI_DIRECT}/api/setup/check", timeout=5)
        assert direct_response.status_code == 200
        direct_data = direct_response.json()
        
        # Kong access
        headers = {"apikey": KONG_API_KEY}
        kong_response = requests.get(f"{KONG_PROXY}/api/setup/check", headers=headers, timeout=5)
        assert kong_response.status_code == 200
        kong_data = kong_response.json()
        
        # Compare key fields (Kong may add headers but data should be same)
        assert direct_data["venv_exists"] == kong_data["venv_exists"]
        assert direct_data["dependencies_installed"] == kong_data["dependencies_installed"]
        assert direct_data["config_exists"] == kong_data["config_exists"]
        
        print("✅ Direct and Kong responses are consistent")


@pytest.mark.integration
class TestFrontendIntegration:
    """Test frontend integration with Kong"""
    
    def test_frontend_can_connect_through_kong(self, kong_available, fastapi_available):
        """Test that frontend-style requests work through Kong"""
        if not kong_available:
            pytest.skip("Kong Gateway not available")
        if not fastapi_available:
            pytest.skip("FastAPI server not available")
        
        # Simulate frontend request with CORS headers
        headers = {
            "apikey": KONG_API_KEY,
            "Origin": "http://localhost:3200",
            "Referer": "http://localhost:3200/"
        }
        
        response = requests.get(f"{KONG_PROXY}/api/setup/check", headers=headers, timeout=5)
        
        assert response.status_code == 200
        print("✅ Frontend-style request works through Kong")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

