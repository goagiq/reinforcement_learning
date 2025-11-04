"""
Sentiment Agent

Gathers and analyzes market and economic sentiment from free sources
to provide sentiment-based insights for trading decisions.
"""

from typing import Dict, Any, Optional
from strands import Agent
from src.agentic_swarm.base_agent import BaseSwarmAgent
from src.agentic_swarm.shared_context import SharedContext
from src.data_sources.sentiment_sources import SentimentDataProvider
from src.reasoning_engine import ReasoningEngine


class SentimentAgent(BaseSwarmAgent):
    """
    Sentiment Agent for market sentiment analysis.
    
    Responsibilities:
    - Aggregate news sentiment (free sources)
    - Analyze economic calendar events
    - Provide sentiment scores and confidence
    - Identify key sentiment drivers
    """
    
    def __init__(
        self,
        shared_context: SharedContext,
        sentiment_provider: SentimentDataProvider,
        reasoning_engine: Optional[ReasoningEngine] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Sentiment Agent.
        
        Args:
            shared_context: Shared context instance
            sentiment_provider: Sentiment data provider instance
            reasoning_engine: Optional reasoning engine
            config: Optional configuration
        """
        config = config or {}
        system_prompt = """You are a market sentiment analyst specializing in gathering and interpreting market sentiment from news, social media, and economic indicators.

Your role is to:
1. Aggregate sentiment from multiple free sources (NewsAPI, etc.)
2. Analyze sentiment trends and patterns
3. Identify key sentiment drivers
4. Provide sentiment scores with confidence levels
5. Assess risk-adjusted sentiment

You have access to:
- News sentiment data (NewsAPI)
- Economic calendar data
- Market volatility indicators

When analyzing sentiment:
- Consider multiple sources for reliability
- Weight recent news more heavily
- Identify sentiment shifts and trends
- Correlate sentiment with market movements
- Provide clear sentiment scores (-1.0 to +1.0)

Format your analysis with:
- Overall sentiment score
- Confidence level
- Key sentiment drivers
- Risk assessment
- Time horizon"""
        
        super().__init__(
            name="sentiment",
            system_prompt=system_prompt,
            shared_context=shared_context,
            reasoning_engine=reasoning_engine,
            config=config
        )
        
        self.sentiment_provider = sentiment_provider
        self.instruments = config.get("instruments", ["ES", "NQ", "RTY", "YM"])
        self.sentiment_window = config.get("sentiment_window", 3600)
        
        # Add description for swarm coordination
        self.description = "Gathers and analyzes market sentiment from news and economic sources to provide sentiment-based trading insights."
        
        # Create tools for the agent
        self._setup_tools()
    
    def _setup_tools(self):
        """Set up tools for the agent."""
        
        def get_news_sentiment(query: str = "futures market", hours_back: int = 24) -> Dict[str, Any]:
            """Get news sentiment for a query."""
            try:
                sentiment = self.sentiment_provider.get_news_sentiment(query, hours_back)
                
                # Store in shared context
                self.shared_context.set(f"news_sentiment_{query}", sentiment, "sentiment_scores")
                
                return sentiment
            except Exception as e:
                return {"error": str(e), "sentiment_score": 0.0}
        
        def get_market_sentiment(instruments: Optional[list] = None) -> Dict[str, Any]:
            """Get overall market sentiment for instruments."""
            try:
                sentiment = self.sentiment_provider.get_market_sentiment(
                    instruments or self.instruments
                )
                
                # Store in shared context
                self.shared_context.set("market_sentiment", sentiment, "sentiment_scores")
                
                return sentiment
            except Exception as e:
                return {"error": str(e), "overall_sentiment": 0.0}
        
        def get_economic_sentiment() -> Dict[str, Any]:
            """Get economic calendar sentiment."""
            try:
                sentiment = self.sentiment_provider.get_economic_sentiment()
                
                # Store in shared context
                self.shared_context.set("economic_sentiment", sentiment, "sentiment_scores")
                
                return sentiment
            except Exception as e:
                return {"error": str(e), "sentiment_score": 0.0}
        
        # Store tools
        self.tools = {
            "get_news_sentiment": get_news_sentiment,
            "get_market_sentiment": get_market_sentiment,
            "get_economic_sentiment": get_economic_sentiment
        }
    
    def analyze(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform sentiment analysis.
        
        Args:
            market_state: Current market state
        
        Returns:
            Dict with sentiment findings
        """
        try:
            # Get market sentiment for all instruments
            market_sentiment = self.tools["get_market_sentiment"]()
            
            # Get economic sentiment
            economic_sentiment = self.tools["get_economic_sentiment"]()
            
            # Aggregate sentiment
            overall_sentiment = market_sentiment.get("overall_sentiment", 0.0)
            confidence = market_sentiment.get("confidence", 0.0)
            
            # Combine with economic sentiment if available
            if "error" not in economic_sentiment:
                econ_sentiment = economic_sentiment.get("sentiment_score", 0.0)
                # Weighted average: 70% market, 30% economic
                overall_sentiment = overall_sentiment * 0.7 + econ_sentiment * 0.3
            
            # Determine sentiment interpretation
            sentiment_interpretation = self._interpret_sentiment(overall_sentiment)
            
            findings = {
                "overall_sentiment": overall_sentiment,
                "confidence": confidence,
                "market_sentiment": market_sentiment,
                "economic_sentiment": economic_sentiment,
                "interpretation": sentiment_interpretation,
                "key_drivers": self._extract_key_drivers(market_sentiment),
                "timestamp": market_state.get("timestamp")
            }
            
            self.shared_context.set("sentiment_findings", findings, "sentiment_scores")
            self.log_action("analyze_sentiment", f"Sentiment: {overall_sentiment:.2f}, Confidence: {confidence:.2f}")
            
            return findings
            
        except Exception as e:
            error_result = {
                "error": str(e),
                "overall_sentiment": 0.0,
                "confidence": 0.0,
                "timestamp": market_state.get("timestamp")
            }
            self.log_action("analyze_sentiment", f"Error: {str(e)}")
            return error_result
    
    def _interpret_sentiment(self, sentiment_score: float) -> str:
        """Interpret sentiment score."""
        if sentiment_score > 0.7:
            return "Very bullish - strong positive sentiment"
        elif sentiment_score > 0.3:
            return "Bullish - positive sentiment"
        elif sentiment_score > -0.3:
            return "Neutral - mixed sentiment"
        elif sentiment_score > -0.7:
            return "Bearish - negative sentiment"
        else:
            return "Very bearish - strong negative sentiment"
    
    def _extract_key_drivers(self, market_sentiment: Dict[str, Any]) -> list:
        """Extract key sentiment drivers."""
        drivers = []
        
        # Check instrument-specific sentiments
        instrument_sentiments = market_sentiment.get("instrument_sentiments", {})
        for instrument, sentiment_data in instrument_sentiments.items():
            score = sentiment_data.get("sentiment_score", 0.0)
            if abs(score) > 0.5:
                direction = "positive" if score > 0 else "negative"
                drivers.append(f"{instrument}: {direction} sentiment ({score:.2f})")
        
        # Check article count
        total_articles = sum(
            s.get("articles_count", 0) 
            for s in instrument_sentiments.values()
        )
        if total_articles > 0:
            drivers.append(f"News coverage: {total_articles} articles analyzed")
        
        return drivers[:5]  # Limit to top 5 drivers

