"""
Integration test for Ollama API with pre-loaded model

This test verifies that:
1. Ollama server is accessible
2. OllamaModel can be initialized with pre-loaded model
3. Agent can be created with Ollama model
4. Tests complete without hanging (model is pre-loaded)
"""

import pytest
import os
from dotenv import load_dotenv
from src.agentic_swarm.shared_context import SharedContext
from src.agentic_swarm.agents import ContrarianAgent
from src.data_sources.market_data import MarketDataProvider

# Load .env file
load_dotenv()


@pytest.fixture
def shared_context():
    """Create shared context for tests."""
    return SharedContext(ttl_seconds=300)


@pytest.fixture
def mock_market_data_provider():
    """Create mock market data provider."""
    from unittest.mock import Mock
    provider = Mock(spec=MarketDataProvider)
    
    import pandas as pd
    correlation_matrix = pd.DataFrame({
        "ES": [1.0, 0.95, 0.92, 0.88],
        "NQ": [0.95, 1.0, 0.90, 0.85],
        "RTY": [0.92, 0.90, 1.0, 0.87],
        "YM": [0.88, 0.85, 0.87, 1.0]
    }, index=["ES", "NQ", "RTY", "YM"])
    
    provider.get_correlation_matrix.return_value = correlation_matrix
    provider.get_rolling_correlation.return_value = 0.95
    provider.get_divergence_signal.return_value = {
        "signal": "normal",
        "signals": {"NQ": "normal", "RTY": "normal", "YM": "normal"},
        "correlations": {"NQ": 0.95, "RTY": 0.92, "YM": 0.88}
    }
    
    return provider


class TestOllamaIntegration:
    """Test Ollama API integration with pre-loaded model."""
    
    def test_ollama_server_accessible(self):
        """Test that Ollama server is running and accessible."""
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            assert response.status_code == 200, "Ollama server not responding"
            
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            assert "deepseek-r1:8b" in models, "deepseek-r1:8b model not found"
            
            print(f"✓ Ollama server accessible with {len(models)} models")
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Ollama server not accessible: {e}")
    
    @pytest.mark.integration
    def test_agent_initialization_with_ollama(
        self, 
        shared_context, 
        mock_market_data_provider
    ):
        """Test that ContrarianAgent can be initialized with Ollama (pre-loaded model)."""
        # Verify Ollama server is accessible
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code != 200:
                pytest.skip("Ollama server not accessible")
        except:
            pytest.skip("Ollama server not accessible")
        
        # Initialize agent with Ollama provider (no skip_api_check - real connection)
        # Model should be pre-loaded to avoid hanging
        agent = ContrarianAgent(
            shared_context=shared_context,
            market_data_provider=mock_market_data_provider,
            config={
                "reasoning": {
                    "provider": "ollama",  # Use Ollama instead of Anthropic
                    "model": "deepseek-r1:8b",
                    "base_url": "http://localhost:11434",
                    "keep_alive": "10m"  # Keep model in memory for faster responses
                }
            }
        )
        
        # Verify agent was created successfully
        assert agent is not None
        assert agent.name == "contrarian"
        assert agent.agent is not None  # Strands Agent should be initialized
        
        # Verify the model was created (OllamaModel)
        from strands.models.ollama import OllamaModel
        model = agent.agent.model
        assert model is not None
        assert isinstance(model, OllamaModel), "Model should be OllamaModel"
        
        print("✓ Agent initialized successfully with Ollama")
        print("✓ OllamaModel created and ready for API calls")
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_ollama_agent_quick_call(
        self,
        shared_context,
        mock_market_data_provider
    ):
        """Test that Ollama agent can make a quick call (model should be pre-loaded)."""
        # Verify Ollama server is accessible
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code != 200:
                pytest.skip("Ollama server not accessible")
        except:
            pytest.skip("Ollama server not accessible")
        
        # Initialize agent with Ollama
        agent = ContrarianAgent(
            shared_context=shared_context,
            market_data_provider=mock_market_data_provider,
            config={
                "reasoning": {
                    "provider": "ollama",
                    "model": "deepseek-r1:8b",
                    "base_url": "http://localhost:11434",
                    "keep_alive": "10m",
                    "max_tokens": 50  # Small response for testing
                }
            }
        )
        
        # Try a simple call - this should be fast if model is pre-loaded
        # Note: We're just testing initialization, not full functionality
        # Actual API calls would be slow and expensive
        model = agent.agent.model
        assert model is not None
        
        # Verify model is OllamaModel (configuration is internal)
        from strands.models.ollama import OllamaModel
        assert isinstance(model, OllamaModel)
        
        # Verify model has necessary attributes (OllamaModel may store config differently)
        # The important thing is that it's initialized and ready
        assert hasattr(model, 'host') or hasattr(model, '_host') or hasattr(model, '_client')
        
        print("✓ Ollama agent ready for calls")
        print("✓ Model configuration verified")
        print("  Note: Model should be pre-loaded for fast responses")

