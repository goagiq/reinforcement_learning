"""
Integration Tests for Swarm Orchestrator

Tests swarm orchestration, handoff logic, and integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime
import yaml

from src.agentic_swarm import SwarmOrchestrator
from src.reasoning_engine import ReasoningEngine
from src.risk_manager import RiskManager


@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        "agentic_swarm": {
            "enabled": True,
            "max_handoffs": 10,
            "max_iterations": 15,
            "execution_timeout": 20.0,
            "node_timeout": 5.0,
            "cache_ttl": 300,
            "market_research": {
                "instruments": ["ES", "NQ"],
                "correlation_window": 20
            },
            "sentiment": {
                "sources": ["newsapi"]
            },
            "analyst": {
                "deep_reasoning": False,  # Disable for faster tests
                "conflict_detection": True
            },
            "recommendation": {
                "risk_integration": True,
                "position_sizing": True
            }
        },
        "reasoning": {
            "enabled": True,
            "provider": "ollama",
            "model": "deepseek-r1:8b",
            "timeout": 2.0
        },
        "risk_management": {
            "max_position_size": 1.0,
            "max_drawdown": 0.20,
            "max_daily_loss": 0.05,
            "stop_loss_atr_multiplier": 2.0
        },
        "data_path": "data",
        "timeframes": [1, 5, 15]
    }


@pytest.fixture
def mock_reasoning_engine():
    """Create mock reasoning engine."""
    engine = Mock(spec=ReasoningEngine)
    return engine


@pytest.fixture
def mock_risk_manager():
    """Create mock risk manager."""
    manager = Mock(spec=RiskManager)
    manager.validate_action.return_value = 0.5
    manager.calculate_stop_loss.return_value = 4950.0
    return manager


class TestSwarmOrchestrator:
    """Test Swarm Orchestrator."""
    
    @patch('src.agentic_swarm.swarm_orchestrator.MarketDataProvider')
    @patch('src.agentic_swarm.swarm_orchestrator.SentimentDataProvider')
    def test_initialization(self, mock_sentiment, mock_market, test_config, mock_reasoning_engine, mock_risk_manager):
        """Test orchestrator initialization."""
        orchestrator = SwarmOrchestrator(
            config=test_config,
            reasoning_engine=mock_reasoning_engine,
            risk_manager=mock_risk_manager
        )
        
        assert orchestrator.is_enabled() is True
        assert len(orchestrator.agents) == 4
        assert orchestrator.shared_context is not None
    
    @patch('src.agentic_swarm.swarm_orchestrator.MarketDataProvider')
    @patch('src.agentic_swarm.swarm_orchestrator.SentimentDataProvider')
    def test_analyze_success(self, mock_sentiment, mock_market, test_config, mock_reasoning_engine, mock_risk_manager):
        """Test successful swarm analysis."""
        orchestrator = SwarmOrchestrator(
            config=test_config,
            reasoning_engine=mock_reasoning_engine,
            risk_manager=mock_risk_manager
        )
        
        # Mock agent analyze methods
        orchestrator.market_research_agent.analyze = Mock(return_value={
            "correlation_matrix": {"ES": {"NQ": 0.95}},
            "divergence_signal": {"signal": "normal"}
        })
        
        orchestrator.sentiment_agent.analyze = Mock(return_value={
            "overall_sentiment": 0.7,
            "confidence": 0.8
        })
        
        orchestrator.analyst_agent.analyze = Mock(return_value={
            "synthesis": {"alignment": "aligned"},
            "conflicts": [],
            "confidence": 0.8
        })
        
        orchestrator.recommendation_agent.recommend = Mock(return_value={
            "action": "BUY",
            "position_size": 0.6,
            "confidence": 0.75,
            "reasoning": "Strong signals"
        })
        
        market_data = {
            "price_data": {"close": 5000.0},
            "volume_data": {"volume": 1000000},
            "timestamp": datetime.now().isoformat()
        }
        
        result = orchestrator.analyze_sync(market_data, None, 0.0)
        
        assert result["status"] == "success"
        assert "recommendation" in result
        assert result["recommendation"]["action"] == "BUY"
    
    @patch('src.agentic_swarm.swarm_orchestrator.MarketDataProvider')
    @patch('src.agentic_swarm.swarm_orchestrator.SentimentDataProvider')
    def test_timeout(self, mock_sentiment, mock_market, test_config, mock_reasoning_engine, mock_risk_manager):
        """Test swarm timeout handling."""
        orchestrator = SwarmOrchestrator(
            config=test_config,
            reasoning_engine=mock_reasoning_engine,
            risk_manager=mock_risk_manager
        )
        
        # Mock slow agent - use synchronous function that blocks
        # Since analyze is called via asyncio.to_thread(), it needs to be sync
        import time
        def slow_analyze(*args, **kwargs):
            time.sleep(10)  # Longer than timeout (5.0s)
            return {}
        
        orchestrator.market_research_agent.analyze = slow_analyze
        
        market_data = {
            "price_data": {"close": 5000.0},
            "timestamp": datetime.now().isoformat()
        }
        
        result = orchestrator.analyze_sync(market_data, None, 0.0, timeout=5.0)
        
        assert result["status"] == "timeout"
    
    @patch('src.agentic_swarm.swarm_orchestrator.MarketDataProvider')
    @patch('src.agentic_swarm.swarm_orchestrator.SentimentDataProvider')
    def test_error_handling(self, mock_sentiment, mock_market, test_config, mock_reasoning_engine, mock_risk_manager):
        """Test error handling."""
        orchestrator = SwarmOrchestrator(
            config=test_config,
            reasoning_engine=mock_reasoning_engine,
            risk_manager=mock_risk_manager
        )
        
        # Mock agent that raises error
        orchestrator.market_research_agent.analyze = Mock(side_effect=Exception("Agent error"))
        
        market_data = {
            "price_data": {"close": 5000.0},
            "timestamp": datetime.now().isoformat()
        }
        
        result = orchestrator.analyze_sync(market_data, None, 0.0)
        
        # Should handle error gracefully
        assert result["status"] in ["error", "success"]  # May still succeed if other agents work
    
    @patch('src.agentic_swarm.swarm_orchestrator.MarketDataProvider')
    @patch('src.agentic_swarm.swarm_orchestrator.SentimentDataProvider')
    def test_status(self, mock_sentiment, mock_market, test_config, mock_reasoning_engine, mock_risk_manager):
        """Test status reporting."""
        orchestrator = SwarmOrchestrator(
            config=test_config,
            reasoning_engine=mock_reasoning_engine,
            risk_manager=mock_risk_manager
        )
        
        status = orchestrator.get_status()
        
        assert "enabled" in status
        assert "agents_initialized" in status
        assert status["agent_count"] == 4
        assert status["execution_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

