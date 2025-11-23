"""
Unit Tests for Agentic Swarm Agents

Tests individual agents:
- Market Research Agent
- Sentiment Agent
- Contrarian Agent
- Analyst Agent
- Recommendation Agent
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.agentic_swarm.shared_context import SharedContext
from src.agentic_swarm.agents import (
    MarketResearchAgent,
    SentimentAgent,
    ContrarianAgent,
    AnalystAgent,
    RecommendationAgent
)
from src.data_sources.market_data import MarketDataProvider
from src.data_sources.sentiment_sources import SentimentDataProvider
from src.risk_manager import RiskManager


@pytest.fixture
def shared_context():
    """Create shared context for tests."""
    return SharedContext(ttl_seconds=300)


@pytest.fixture
def mock_market_data_provider():
    """Create mock market data provider."""
    provider = Mock(spec=MarketDataProvider)
    
    # Mock correlation matrix
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


@pytest.fixture
def mock_sentiment_provider():
    """Create mock sentiment provider."""
    provider = Mock(spec=SentimentDataProvider)
    
    provider.get_news_sentiment.return_value = {
        "sentiment_score": 0.6,
        "confidence": 0.8,
        "articles_count": 25,
        "source": "newsapi"
    }
    
    provider.get_market_sentiment.return_value = {
        "overall_sentiment": 0.65,
        "confidence": 0.75,
        "instrument_sentiments": {
            "ES": {"sentiment_score": 0.7, "confidence": 0.8},
            "NQ": {"sentiment_score": 0.6, "confidence": 0.7}
        }
    }
    
    provider.get_economic_sentiment.return_value = {
        "sentiment_score": 0.5,
        "confidence": 0.6,
        "source": "economic_calendar"
    }
    
    return provider


@pytest.fixture
def mock_risk_manager():
    """Create mock risk manager."""
    manager = Mock(spec=RiskManager)
    manager.validate_action.return_value = 0.5
    manager.calculate_stop_loss.return_value = 4950.0
    manager.get_risk_status.return_value = {
        "current_capital": 100000.0,
        "current_drawdown": 0.05,
        "can_trade": True
    }
    return manager


@pytest.fixture(autouse=True)
def mock_strands_agents():
    """Auto-mock Strands Agent and AnthropicModel to prevent API calls."""
    with patch('strands.models.anthropic.AnthropicModel') as mock_anthropic, \
         patch('strands.Agent') as mock_agent:
        mock_anthropic_instance = MagicMock()
        mock_anthropic.return_value = mock_anthropic_instance
        
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance
        
        yield {
            'anthropic_model': mock_anthropic_instance,
            'agent': mock_agent_instance
        }


class TestMarketResearchAgent:
    """Test Market Research Agent."""
    
    def test_initialization(self, shared_context, mock_market_data_provider):
        """Test agent initialization."""
        agent = MarketResearchAgent(
            shared_context=shared_context,
            market_data_provider=mock_market_data_provider,
            config={"instruments": ["ES", "NQ"], "correlation_window": 20}
        )
        
        assert agent.name == "market_research"
        assert agent.instruments == ["ES", "NQ"]
        assert agent.correlation_window == 20
    
    def test_analyze(self, shared_context, mock_market_data_provider):
        """Test market research analysis."""
        agent = MarketResearchAgent(
            shared_context=shared_context,
            market_data_provider=mock_market_data_provider
        )
        
        market_state = {
            "price_data": {"close": 5000.0},
            "timestamp": datetime.now().isoformat()
        }
        
        result = agent.analyze(market_state)
        
        assert "correlation_matrix" in result
        assert "divergence_signal" in result
        assert result["divergence_signal"]["signal"] == "normal"
    
    def test_tools(self, shared_context, mock_market_data_provider):
        """Test agent tools."""
        agent = MarketResearchAgent(
            shared_context=shared_context,
            market_data_provider=mock_market_data_provider
        )
        
        # Test correlation calculation
        corr_result = agent.tools["calculate_correlation"]("ES", "NQ", 5)
        assert "correlation" in corr_result
        assert corr_result["correlation"] == 0.95
        
        # Test divergence detection
        div_result = agent.tools["detect_divergence"]()
        assert "signal" in div_result


class TestSentimentAgent:
    """Test Sentiment Agent."""
    
    def test_initialization(self, shared_context, mock_sentiment_provider):
        """Test agent initialization."""
        agent = SentimentAgent(
            shared_context=shared_context,
            sentiment_provider=mock_sentiment_provider
        )
        
        assert agent.name == "sentiment"
        assert agent.sentiment_provider is not None
    
    def test_analyze(self, shared_context, mock_sentiment_provider):
        """Test sentiment analysis."""
        agent = SentimentAgent(
            shared_context=shared_context,
            sentiment_provider=mock_sentiment_provider
        )
        
        market_state = {
            "price_data": {"close": 5000.0},
            "timestamp": datetime.now().isoformat()
        }
        
        result = agent.analyze(market_state)
        
        assert "overall_sentiment" in result
        assert "confidence" in result
        assert result["overall_sentiment"] == pytest.approx(0.605, rel=1e-6)
        assert result["confidence"] == 0.75
    
    def test_tools(self, shared_context, mock_sentiment_provider):
        """Test agent tools."""
        agent = SentimentAgent(
            shared_context=shared_context,
            sentiment_provider=mock_sentiment_provider
        )
        
        # Test news sentiment
        news_result = agent.tools["get_news_sentiment"]("futures", 24)
        assert "sentiment_score" in news_result
        assert news_result["sentiment_score"] == 0.6
        
        # Test market sentiment
        market_result = agent.tools["get_market_sentiment"]()
        assert "overall_sentiment" in market_result


class TestContrarianAgent:
    """Test Contrarian Agent."""
    
    def test_initialization(self, shared_context, mock_market_data_provider):
        """Test agent initialization."""
        agent = ContrarianAgent(
            shared_context=shared_context,
            market_data_provider=mock_market_data_provider,
            config={
                "lookback_periods": 50,
                "greed_threshold_percentile": 90,
                "fear_threshold_percentile": 10,
                "reasoning": {"skip_api_check": True}
            }
        )
        
        assert agent.name == "contrarian"
        assert agent.market_data_provider is not None
        assert agent.lookback_periods == 50
        assert agent.greed_threshold_percentile == 90
        assert agent.fear_threshold_percentile == 10
    
    def test_analyze_greedy_condition(self, shared_context, mock_market_data_provider):
        """Test detection of greedy conditions (should return SELL)."""
        agent = ContrarianAgent(
            shared_context=shared_context,
            market_data_provider=mock_market_data_provider,
            config={"reasoning": {"skip_api_check": True}}
        )
        
        # Set up extreme positive sentiment (greedy)
        # First build sentiment history, then calculate threshold, then set sentiment above threshold
        # Build sentiment history with lower values so 0.95 is clearly above 90th percentile
        agent.sentiment_history = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85] * 5  # 50 values, max 0.85
        
        # Calculate what the 90th percentile would be
        import numpy as np
        greed_threshold, _ = agent._calculate_dynamic_thresholds()
        
        # Set sentiment above the greed threshold
        sentiment_value = max(0.95, greed_threshold + 0.1)  # Ensure it's above threshold
        
        shared_context.set("sentiment_findings", {
            "overall_sentiment": sentiment_value,  # Very high sentiment, above 90th percentile
            "confidence": 0.9
        }, "sentiment_scores")
        
        market_state = {
            "price_data": {
                "close": 5000.0,
                "high": 5100.0,  # High volatility (0.04 = 2%)
                "low": 4900.0
            },
            "volume_data": {
                "volume": 1500000,
                "volume_ratio": 1.5  # High volume
            },
            "indicators": {},
            "timestamp": datetime.now().isoformat()
        }
        
        result = agent.analyze(market_state)
        
        assert "market_condition" in result
        assert "contrarian_signal" in result
        assert "contrarian_confidence" in result
        assert result["market_condition"] == "GREEDY"
        assert result["contrarian_signal"] == "SELL"
        assert result["contrarian_confidence"] > 0.0
    
    def test_analyze_fearful_condition(self, shared_context, mock_market_data_provider):
        """Test detection of fearful conditions (should return BUY)."""
        agent = ContrarianAgent(
            shared_context=shared_context,
            market_data_provider=mock_market_data_provider,
            config={"reasoning": {"skip_api_check": True}}
        )
        
        # Set up extreme negative sentiment (fearful)
        # Need sentiment that's lower than the 10th percentile
        shared_context.set("sentiment_findings", {
            "overall_sentiment": -0.95,  # Very negative sentiment (extreme)
            "confidence": 0.9
        }, "sentiment_scores")
        
        # Build sentiment history for percentile calculation
        # Mix of positive and negative values, so -0.95 is clearly in bottom 10%
        agent.sentiment_history = list(range(-50, 51))
        agent.sentiment_history = [x / 50.0 for x in agent.sentiment_history]  # -1.0 to 1.0
        
        market_state = {
            "price_data": {
                "close": 4500.0,
                "high": 4590.0,  # High volatility (0.04 = 2%)
                "low": 4410.0
            },
            "volume_data": {
                "volume": 1500000,
                "volume_ratio": 1.5  # High volume
            },
            "indicators": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Ensure we have enough history for percentile calculation
        assert len(agent.sentiment_history) >= 10, "Need at least 10 sentiment history values"
        
        result = agent.analyze(market_state)
        
        # Debug output if needed
        if result["market_condition"] != "FEARFUL":
            print(f"DEBUG: sentiment={result.get('sentiment_score')}, "
                  f"greed_threshold={result.get('greed_threshold')}, "
                  f"fear_threshold={result.get('fear_threshold')}, "
                  f"volatility={result.get('volatility')}, "
                  f"volume_ratio={result.get('volume_metrics', {}).get('volume_ratio')}")
        
        assert "market_condition" in result
        assert "contrarian_signal" in result
        assert result["market_condition"] == "FEARFUL", f"Expected FEARFUL, got {result['market_condition']}"
        assert result["contrarian_signal"] == "BUY"
        assert result["contrarian_confidence"] > 0.0
    
    def test_analyze_neutral_condition(self, shared_context, mock_market_data_provider):
        """Test detection of neutral conditions (should return HOLD)."""
        agent = ContrarianAgent(
            shared_context=shared_context,
            market_data_provider=mock_market_data_provider,
            config={"reasoning": {"skip_api_check": True}}
        )
        
        # Set up neutral sentiment
        shared_context.set("sentiment_findings", {
            "overall_sentiment": 0.2,  # Moderate sentiment
            "confidence": 0.6
        }, "sentiment_scores")
        
        market_state = {
            "price_data": {
                "close": 5000.0,
                "high": 5010.0,  # Low volatility
                "low": 4990.0
            },
            "volume_data": {
                "volume": 1000000,
                "volume_ratio": 1.0  # Normal volume
            },
            "indicators": {},
            "timestamp": datetime.now().isoformat()
        }
        
        result = agent.analyze(market_state)
        
        assert "market_condition" in result
        assert "contrarian_signal" in result
        assert result["market_condition"] == "NEUTRAL"
        assert result["contrarian_signal"] == "HOLD"
        assert result["contrarian_confidence"] == 0.0
    
    def test_dynamic_thresholds(self, shared_context, mock_market_data_provider):
        """Test dynamic threshold calculation based on sentiment history."""
        agent = ContrarianAgent(
            shared_context=shared_context,
            market_data_provider=mock_market_data_provider,
            config={"reasoning": {"skip_api_check": True}}
        )
        
        # Build sentiment history with known distribution
        agent.sentiment_history = list(range(-50, 51))  # -50 to 50, normalized to -1.0 to 1.0
        agent.sentiment_history = [x / 50.0 for x in agent.sentiment_history]  # Normalize to -1.0 to 1.0
        
        greed_threshold, fear_threshold = agent._calculate_dynamic_thresholds()
        
        # 90th percentile should be around 0.8, 10th percentile around -0.8
        assert greed_threshold > 0.5  # Should be positive
        assert fear_threshold < -0.5  # Should be negative
        assert greed_threshold > fear_threshold  # Greed should be higher than fear
    
    def test_volatility_calculation(self, shared_context, mock_market_data_provider):
        """Test volatility calculation."""
        agent = ContrarianAgent(
            shared_context=shared_context,
            market_data_provider=mock_market_data_provider,
            config={"reasoning": {"skip_api_check": True}}
        )
        
        price_data = {
            "close": 5000.0,
            "high": 5100.0,
            "low": 4900.0
        }
        
        market_state = {
            "price_data": price_data,
            "indicators": {}
        }
        
        volatility = agent._calculate_volatility(price_data, market_state)
        
        # Volatility should be (5100 - 4900) / 5000 = 0.04
        assert volatility > 0.0
        assert volatility < 1.0
        assert abs(volatility - 0.04) < 0.01
    
    def test_volume_metrics(self, shared_context, mock_market_data_provider):
        """Test volume metrics calculation."""
        agent = ContrarianAgent(
            shared_context=shared_context,
            market_data_provider=mock_market_data_provider,
            config={"reasoning": {"skip_api_check": True}}
        )
        
        volume_data = {
            "volume": 1500000,
            "volume_ratio": 1.5
        }
        
        market_state = {
            "volume_data": volume_data
        }
        
        volume_metrics = agent._calculate_volume_metrics(volume_data, market_state)
        
        assert "current_volume" in volume_metrics
        assert "volume_ratio" in volume_metrics
        assert "volume_percentile" in volume_metrics
        assert volume_metrics["current_volume"] == 1500000.0
        assert volume_metrics["volume_ratio"] == 1.5
    
    def test_contrarian_influence_in_recommendation(self, shared_context, mock_risk_manager):
        """Test that RecommendationAgent is swayed by contrarian signals."""
        
        # Ensure risk manager returns valid values
        mock_risk_manager.validate_action.return_value = 0.5  # Return non-zero
        mock_risk_manager.calculate_stop_loss.return_value = 4950.0
        mock_risk_manager.get_risk_status.return_value = {
            "current_capital": 100000.0,
            "current_drawdown": 0.05,
            "can_trade": True
        }
        
        # Disable risk integration and position sizing to avoid issues
        agent = RecommendationAgent(
            shared_context=shared_context,
            risk_manager=mock_risk_manager,
            config={
                "risk_integration": False,
                "position_sizing": False,
                "reasoning": {
                    "enabled": False,  # Disable reasoning to avoid API calls
                    "skip_api_check": True  # Skip API key check for mocked tests
                }
            }
        )
        
        # Set up normal recommendation that would normally be BUY
        # Need to ensure alignment + sentiment triggers BUY
        shared_context.set("analyst_analysis", {
            "synthesis": {"alignment": "aligned"},
            "confidence": 0.8  # High confidence
        }, "analysis_results")
        
        shared_context.set("sentiment_findings", {
            "overall_sentiment": 0.7,  # Strong positive sentiment (above 0.5 threshold)
            "confidence": 0.8  # High confidence (above 0.6 threshold)
        }, "sentiment_scores")
        
        shared_context.set("market_research_findings", {
            "divergence_signal": {"signal": "normal"}
        }, "research_findings")
        
        # Set up GREEDY contrarian signal (should override to SELL)
        shared_context.set("contrarian_analysis", {
            "market_condition": "GREEDY",
            "contrarian_signal": "SELL",
            "contrarian_confidence": 0.8,  # High confidence triggers override
            "reasoning": "Extreme greed detected"
        }, "contrarian_signals")
        
        market_state = {
            "price_data": {"close": 5000.0, "high": 5010.0, "low": 4990.0},
            "volume_data": {"volume": 1000000},
            "timestamp": datetime.now().isoformat()
        }
        
        # Test the contrarian influence method directly to avoid full agent initialization
        # Create a base recommendation
        base_recommendation = {
            "action": "BUY",
            "position_size": 0.6,
            "confidence": 0.7,
            "reasoning": "Base recommendation"
        }
        
        # Store original position size before modification
        original_position_size = base_recommendation["position_size"]
        
        contrarian_analysis = shared_context.get("contrarian_analysis", "contrarian_signals")
        
        # Apply contrarian influence (modifies in place)
        result = agent._apply_contrarian_influence(base_recommendation, contrarian_analysis)
        
        # Should be overridden to SELL due to contrarian signal (confidence >= 0.6)
        assert result["action"] == "SELL", f"Expected SELL, got {result.get('action')}"
        assert "contrarian_confidence" in result
        assert result["contrarian_confidence"] == 0.8
        assert result["market_condition"] == "GREEDY"
        # Position size should increase: 0.6 * 1.2 = 0.72
        assert result["position_size"] > original_position_size, \
            f"Expected position_size > {original_position_size}, got {result['position_size']}"
        assert result["position_size"] == 0.72, \
            f"Expected position_size == 0.72 (0.6 * 1.2), got {result['position_size']}"


class TestAnalystAgent:
    """Test Analyst Agent."""
    
    def test_initialization(self, shared_context):
        """Test agent initialization."""
        agent = AnalystAgent(shared_context=shared_context)
        
        assert agent.name == "analyst"
        assert agent.deep_reasoning is True
    
    def test_analyze(self, shared_context):
        """Test analyst synthesis."""
        agent = AnalystAgent(shared_context=shared_context)
        
        # Set up research and sentiment findings
        shared_context.set("market_research_findings", {
            "divergence_signal": {"signal": "normal"},
            "correlation_matrix": {"analysis": {"average_correlation": 0.90}}
        }, "research_findings")
        
        shared_context.set("sentiment_findings", {
            "overall_sentiment": 0.7,
            "confidence": 0.8
        }, "sentiment_scores")
        
        market_state = {
            "price_data": {"close": 5000.0},
            "timestamp": datetime.now().isoformat()
        }
        
        result = agent.analyze(market_state, None)
        
        assert "synthesis" in result
        assert "conflicts" in result
        assert result["synthesis"]["alignment"] in ["aligned", "conflict", "neutral"]
    
    def test_conflict_detection(self, shared_context):
        """Test conflict detection."""
        agent = AnalystAgent(shared_context=shared_context)
        
        # Set up conflicting signals
        shared_context.set("market_research_findings", {
            "divergence_signal": {"signal": "divergence"}
        }, "research_findings")
        
        shared_context.set("sentiment_findings", {
            "overall_sentiment": 0.8,  # Strong positive sentiment
            "confidence": 0.9
        }, "sentiment_scores")
        
        market_state = {"timestamp": datetime.now().isoformat()}
        result = agent.analyze(market_state, None)
        
        # Should detect conflict
        assert len(result["conflicts"]) > 0


class TestRecommendationAgent:
    """Test Recommendation Agent."""
    
    def test_initialization(self, shared_context, mock_risk_manager):
        """Test agent initialization."""
        agent = RecommendationAgent(
            shared_context=shared_context,
            risk_manager=mock_risk_manager
        )
        
        assert agent.name == "recommendation"
        assert agent.risk_manager is not None
    
    def test_recommend(self, shared_context, mock_risk_manager):
        """Test recommendation generation."""
        agent = RecommendationAgent(
            shared_context=shared_context,
            risk_manager=mock_risk_manager
        )
        
        # Set up analyst analysis
        shared_context.set("analyst_analysis", {
            "synthesis": {"alignment": "aligned"},
            "conflicts": [],
            "confidence": 0.8
        }, "analysis_results")
        
        shared_context.set("market_research_findings", {
            "divergence_signal": {"signal": "normal"}
        }, "research_findings")
        
        shared_context.set("sentiment_findings", {
            "overall_sentiment": 0.7,
            "confidence": 0.8
        }, "sentiment_scores")
        
        market_state = {
            "price_data": {"close": 5000.0, "high": 5010.0, "low": 4990.0},
            "volume_data": {"volume": 1000000},
            "timestamp": datetime.now().isoformat()
        }
        
        result = agent.recommend(market_state, None, 0.0)
        
        assert "action" in result
        assert "position_size" in result
        assert "confidence" in result
        assert result["action"] in ["BUY", "SELL", "HOLD"]
    
    def test_risk_integration(self, shared_context, mock_risk_manager):
        """Test risk management integration."""
        agent = RecommendationAgent(
            shared_context=shared_context,
            risk_manager=mock_risk_manager,
            config={"risk_integration": True}
        )
        
        # Set up analysis
        shared_context.set("analyst_analysis", {
            "synthesis": {"alignment": "aligned"},
            "confidence": 0.9
        }, "analysis_results")
        shared_context.set("sentiment_findings", {
            "overall_sentiment": 0.7,
            "confidence": 0.8
        }, "sentiment_scores")
        
        market_state = {
            "price_data": {"close": 5000.0, "high": 5010.0, "low": 4990.0},
            "volume_data": {"volume": 1000000},
            "timestamp": datetime.now().isoformat()
        }
        
        result = agent.recommend(market_state, None, 0.0)
        
        # Risk manager should be called
        mock_risk_manager.validate_action.assert_called()
        mock_risk_manager.calculate_stop_loss.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

