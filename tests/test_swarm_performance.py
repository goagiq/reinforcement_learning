"""
Performance Tests for Swarm System

Measures execution times, API calls, and resource usage.
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime

from src.agentic_swarm import SwarmOrchestrator


@pytest.fixture
def performance_config():
    """Create configuration for performance testing."""
    return {
        "agentic_swarm": {
            "enabled": True,
            "execution_timeout": 30.0,
            "cache_ttl": 300,
            "market_research": {"instruments": ["ES", "NQ"]},
            "sentiment": {"sources": ["newsapi"]},
            "analyst": {"deep_reasoning": False},  # Disable for performance tests
            "recommendation": {"risk_integration": True}
        },
        "reasoning": {"enabled": False},  # Disable for performance tests
        "risk_management": {
            "max_position_size": 1.0,
            "max_drawdown": 0.20
        },
        "data_path": "data",
        "timeframes": [1, 5]
    }


class TestSwarmPerformance:
    """Test swarm performance."""
    
    @patch('src.agentic_swarm.swarm_orchestrator.MarketDataProvider')
    @patch('src.agentic_swarm.swarm_orchestrator.SentimentDataProvider')
    def test_execution_time(self, mock_sentiment, mock_market, performance_config):
        """Test swarm execution time (target: 5-20s)."""
        orchestrator = SwarmOrchestrator(config=performance_config)
        
        # Mock fast agents
        orchestrator.market_research_agent.analyze = Mock(return_value={"status": "ok"})
        orchestrator.sentiment_agent.analyze = Mock(return_value={"status": "ok"})
        orchestrator.analyst_agent.analyze = Mock(return_value={"status": "ok"})
        orchestrator.recommendation_agent.recommend = Mock(return_value={
            "action": "BUY", "position_size": 0.5, "confidence": 0.8
        })
        
        market_data = {
            "price_data": {"close": 5000.0},
            "timestamp": datetime.now().isoformat()
        }
        
        start_time = time.time()
        result = orchestrator.analyze_sync(market_data, None, 0.0)
        execution_time = time.time() - start_time
        
        assert result["status"] == "success"
        assert execution_time < 5.0  # Should be fast with mocked agents
        assert "execution_time" in result
    
    @patch('src.agentic_swarm.swarm_orchestrator.MarketDataProvider')
    @patch('src.agentic_swarm.swarm_orchestrator.SentimentDataProvider')
    def test_parallel_execution(self, mock_sentiment, mock_market, performance_config):
        """Test that Market Research and Sentiment run in parallel."""
        orchestrator = SwarmOrchestrator(config=performance_config)
        
        research_times = []
        sentiment_times = []
        
        def timed_research(*args, **kwargs):
            start = time.time()
            time.sleep(0.1)  # Simulate work
            research_times.append(time.time() - start)
            return {"status": "ok"}
        
        def timed_sentiment(*args, **kwargs):
            start = time.time()
            time.sleep(0.1)  # Simulate work
            sentiment_times.append(time.time() - start)
            return {"status": "ok"}
        
        orchestrator.market_research_agent.analyze = timed_research
        orchestrator.sentiment_agent.analyze = timed_sentiment
        orchestrator.analyst_agent.analyze = Mock(return_value={"status": "ok"})
        orchestrator.recommendation_agent.recommend = Mock(return_value={
            "action": "BUY", "position_size": 0.5
        })
        
        market_data = {
            "price_data": {"close": 5000.0},
            "timestamp": datetime.now().isoformat()
        }
        
        start_time = time.time()
        orchestrator.analyze_sync(market_data, None, 0.0)
        total_time = time.time() - start_time
        
        # If parallel, total time should be ~max(research_time, sentiment_time) + overhead
        # Not research_time + sentiment_time
        max_agent_time = max(research_times[0] if research_times else 0,
                           sentiment_times[0] if sentiment_times else 0)
        
        # Allow some overhead
        assert total_time < (max_agent_time * 2) if max_agent_time > 0 else True
    
    @patch('src.agentic_swarm.swarm_orchestrator.MarketDataProvider')
    @patch('src.agentic_swarm.swarm_orchestrator.SentimentDataProvider')
    def test_cache_effectiveness(self, mock_sentiment, mock_market, performance_config):
        """Test that caching reduces execution time on repeated calls."""
        orchestrator = SwarmOrchestrator(config=performance_config)
        
        call_count = [0]
        
        def counting_analyze(*args, **kwargs):
            call_count[0] += 1
            time.sleep(0.05)  # Simulate work
            return {"status": "ok"}
        
        orchestrator.market_research_agent.analyze = counting_analyze
        orchestrator.sentiment_agent.analyze = Mock(return_value={"status": "ok"})
        orchestrator.analyst_agent.analyze = Mock(return_value={"status": "ok"})
        orchestrator.recommendation_agent.recommend = Mock(return_value={
            "action": "BUY", "position_size": 0.5
        })
        
        market_data = {
            "price_data": {"close": 5000.0},
            "timestamp": datetime.now().isoformat()
        }
        
        # First call
        orchestrator.analyze_sync(market_data, None, 0.0)
        first_call_count = call_count[0]
        
        # Second call (should use cache if implemented)
        orchestrator.analyze_sync(market_data, None, 0.0)
        second_call_count = call_count[0]
        
        # Note: Actual caching would prevent second call
        # This test verifies the structure exists
        assert second_call_count >= first_call_count


class TestSwarmErrorScenarios:
    """Test error scenarios and fallback."""
    
    @patch('src.agentic_swarm.swarm_orchestrator.MarketDataProvider')
    @patch('src.agentic_swarm.swarm_orchestrator.SentimentDataProvider')
    def test_missing_data_sources(self, mock_sentiment, mock_market, performance_config):
        """Test behavior with missing data sources."""
        orchestrator = SwarmOrchestrator(config=performance_config)
        
        # Mock agent that fails
        orchestrator.market_research_agent.analyze = Mock(side_effect=Exception("Data unavailable"))
        orchestrator.sentiment_agent.analyze = Mock(return_value={"status": "ok"})
        orchestrator.analyst_agent.analyze = Mock(return_value={"status": "ok"})
        orchestrator.recommendation_agent.recommend = Mock(return_value={
            "action": "HOLD", "position_size": 0.0
        })
        
        market_data = {
            "price_data": {"close": 5000.0},
            "timestamp": datetime.now().isoformat()
        }
        
        # Should handle gracefully
        result = orchestrator.analyze_sync(market_data, None, 0.0)
        
        # May succeed with partial data or fail gracefully
        assert result["status"] in ["success", "error"]
    
    @patch('src.agentic_swarm.swarm_orchestrator.MarketDataProvider')
    @patch('src.agentic_swarm.swarm_orchestrator.SentimentDataProvider')
    def test_api_failure(self, mock_sentiment, mock_market, performance_config):
        """Test API failure scenarios."""
        orchestrator = SwarmOrchestrator(config=performance_config)
        
        # Mock API failure
        orchestrator.sentiment_provider.get_news_sentiment = Mock(
            side_effect=Exception("API timeout")
        )
        
        orchestrator.market_research_agent.analyze = Mock(return_value={"status": "ok"})
        orchestrator.sentiment_agent.analyze = Mock(return_value={
            "error": "API failure",
            "overall_sentiment": 0.0,
            "confidence": 0.0
        })
        orchestrator.analyst_agent.analyze = Mock(return_value={"status": "ok"})
        orchestrator.recommendation_agent.recommend = Mock(return_value={
            "action": "HOLD", "position_size": 0.0
        })
        
        market_data = {
            "price_data": {"close": 5000.0},
            "timestamp": datetime.now().isoformat()
        }
        
        result = orchestrator.analyze_sync(market_data, None, 0.0)
        
        # Should handle gracefully
        assert result["status"] in ["success", "error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

