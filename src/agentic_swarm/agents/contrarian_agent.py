"""
Contrarian Agent

Implements Warren Buffett's philosophy: "Be fearful when others are greedy, 
and greedy when others are fearful."

Detects market extremes using sentiment + volatility + volume analysis,
and provides contrarian signals to sway trading decisions.
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from copy import deepcopy
from src.agentic_swarm.base_agent import BaseSwarmAgent
from src.agentic_swarm.shared_context import SharedContext
from src.data_sources.market_data import MarketDataProvider
from src.reasoning_engine import ReasoningEngine


class ContrarianAgent(BaseSwarmAgent):
    """
    Contrarian Agent for detecting market extremes.
    
    Responsibilities:
    - Analyze sentiment + volatility + volume for greed/fear detection
    - Calculate dynamic thresholds based on recent distributions
    - Provide contrarian signals (sell on greed, buy on fear)
    - Influence RecommendationAgent when extremes are detected
    """
    
    def __init__(
        self,
        shared_context: SharedContext,
        market_data_provider: MarketDataProvider,
        reasoning_engine: Optional[ReasoningEngine] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Contrarian Agent.
        
        Args:
            shared_context: Shared context instance
            market_data_provider: Market data provider for volatility/volume
            reasoning_engine: Optional reasoning engine
            config: Optional configuration
        """
        config = deepcopy(config) if config else {}
        reasoning_config = config.setdefault("reasoning", {})
        reasoning_config.setdefault("model", "claude-sonnet-4-20250514")
        system_prompt = """You are a contrarian market analyst following Warren Buffett's philosophy:
"Be fearful when others are greedy, and greedy when others are fearful."

Your role is to:
1. Detect market extremes (greed/fear) using sentiment, volatility, and volume
2. Identify when markets are at tops (greed) or bottoms (fear)
3. Provide contrarian signals that go against the crowd
4. Help avoid buying at market tops and identify buying opportunities at bottoms

When others are GREEDY (market top):
- Sentiment is extremely positive
- Prices are high and rising rapidly
- Volatility may be elevated
- Volume may be high
- RECOMMENDATION: SELL (take profits, exit positions)

When others are FEARFUL (market bottom):
- Sentiment is extremely negative
- Prices have fallen significantly
- Volatility may be high (panic selling)
- Volume may be elevated
- RECOMMENDATION: BUY (normal position size, but higher confidence)

Format your analysis with:
- Market condition: [GREEDY/FEARFUL/NEUTRAL]
- Contrarian signal: [SELL/BUY/HOLD]
- Confidence level (0.0 to 1.0)
- Reasoning based on sentiment + volatility + volume
- Historical context (Great Depression, Internet Bust, COVID crash patterns)"""
        
        super().__init__(
            name="contrarian",
            system_prompt=system_prompt,
            shared_context=shared_context,
            reasoning_engine=reasoning_engine,
            config=config
        )
        
        self.market_data_provider = market_data_provider
        self.lookback_periods = config.get("lookback_periods", 50)  # For percentile calculation
        self.greed_threshold_percentile = config.get("greed_threshold_percentile", 90)  # Top 10%
        self.fear_threshold_percentile = config.get("fear_threshold_percentile", 10)  # Bottom 10%
        
        # Store historical sentiment for dynamic thresholds
        self.sentiment_history: List[float] = []
        
        # Add description for swarm coordination
        self.description = "Detects market extremes (greed/fear) using sentiment + volatility + volume. Provides contrarian signals following Warren Buffett's philosophy."
    
    def analyze(
        self,
        market_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze market for contrarian signals.
        
        Args:
            market_state: Current market state
            
        Returns:
            Dict with contrarian analysis
        """
        try:
            # Get sentiment findings from shared context (set by SentimentAgent)
            sentiment_findings = self.shared_context.get("sentiment_findings", "sentiment_scores")
            
            # Get market data for volatility and volume analysis
            price_data = market_state.get("price_data", {})
            volume_data = market_state.get("volume_data", {})
            
            # Extract current metrics
            sentiment_score = sentiment_findings.get("overall_sentiment", 0.0) if sentiment_findings else 0.0
            sentiment_confidence = sentiment_findings.get("confidence", 0.0) if sentiment_findings else 0.0
            
            # Calculate volatility (using price range or ATR if available)
            volatility = self._calculate_volatility(price_data, market_state)
            
            # Calculate volume metrics
            volume_metrics = self._calculate_volume_metrics(volume_data, market_state)
            
            # Calculate dynamic thresholds (before updating history with current value)
            greed_threshold, fear_threshold = self._calculate_dynamic_thresholds()
            
            # Update sentiment history for next iteration (after calculating thresholds)
            self.sentiment_history.append(sentiment_score)
            if len(self.sentiment_history) > self.lookback_periods:
                self.sentiment_history.pop(0)
            
            # Detect market condition
            market_condition = self._detect_market_condition(
                sentiment_score,
                volatility,
                volume_metrics,
                greed_threshold,
                fear_threshold
            )
            
            # Generate contrarian signal
            contrarian_signal = self._generate_contrarian_signal(
                market_condition,
                sentiment_score,
                volatility,
                volume_metrics
            )
            
            # Calculate contrarian confidence
            contrarian_confidence = self._calculate_contrarian_confidence(
                market_condition,
                sentiment_confidence,
                volatility,
                volume_metrics
            )
            
            # Build analysis result
            analysis = {
                "market_condition": market_condition,  # "GREEDY", "FEARFUL", "NEUTRAL"
                "contrarian_signal": contrarian_signal,  # "SELL", "BUY", "HOLD"
                "contrarian_confidence": contrarian_confidence,
                "sentiment_score": sentiment_score,
                "volatility": volatility,
                "volume_metrics": volume_metrics,
                "greed_threshold": greed_threshold,
                "fear_threshold": fear_threshold,
                "reasoning": self._build_reasoning(
                    market_condition,
                    sentiment_score,
                    volatility,
                    volume_metrics
                ),
                "timestamp": market_state.get("timestamp", datetime.now().isoformat())
            }
            
            # Store in shared context
            self.shared_context.set("contrarian_analysis", analysis, "contrarian_signals")
            self.log_action(
                "detect_contrarian_signal",
                f"{market_condition} -> {contrarian_signal} (confidence: {contrarian_confidence:.2f})"
            )
            
            return analysis
            
        except Exception as e:
            error_result = {
                "error": str(e),
                "market_condition": "NEUTRAL",
                "contrarian_signal": "HOLD",
                "contrarian_confidence": 0.0,
                "timestamp": market_state.get("timestamp", datetime.now().isoformat())
            }
            self.log_action("detect_contrarian_signal", f"Error: {str(e)}")
            return error_result
    
    def _calculate_volatility(
        self,
        price_data: Dict[str, Any],
        market_state: Dict[str, Any]
    ) -> float:
        """Calculate current volatility metric."""
        try:
            current_price = price_data.get("close", 0.0)
            high = price_data.get("high", current_price)
            low = price_data.get("low", current_price)
            
            if current_price == 0:
                return 0.0
            
            # Use price range as volatility proxy
            price_range = high - low
            volatility = price_range / current_price if current_price > 0 else 0.0
            
            # Try to get ATR if available in indicators
            indicators = market_state.get("indicators", {})
            if "atr" in indicators:
                atr = indicators["atr"]
                volatility = atr / current_price if current_price > 0 else volatility
            
            return float(volatility)
            
        except Exception:
            return 0.0
    
    def _calculate_volume_metrics(
        self,
        volume_data: Dict[str, Any],
        market_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate volume metrics."""
        try:
            current_volume = volume_data.get("volume", 0.0)
            
            # Get historical volume if available from market data provider
            # For now, use volume ratio if available
            volume_ratio = volume_data.get("volume_ratio", 1.0)
            
            # Calculate volume percentile if we have historical data
            volume_percentile = 0.5  # Default neutral
            
            # Try to get volume from market data provider
            # This would require historical volume data
            # For now, use volume_ratio as a proxy
            
            return {
                "current_volume": float(current_volume),
                "volume_ratio": float(volume_ratio),
                "volume_percentile": volume_percentile
            }
            
        except Exception:
            return {
                "current_volume": 0.0,
                "volume_ratio": 1.0,
                "volume_percentile": 0.5
            }
    
    def _calculate_dynamic_thresholds(self) -> Tuple[float, float]:
        """
        Calculate dynamic greed/fear thresholds based on sentiment history percentiles.
        
        Returns:
            (greed_threshold, fear_threshold)
        """
        if len(self.sentiment_history) < 10:
            # Not enough history, use fixed thresholds
            return (0.7, -0.7)
        
        sentiment_array = np.array(self.sentiment_history)
        greed_threshold = np.percentile(sentiment_array, self.greed_threshold_percentile)
        fear_threshold = np.percentile(sentiment_array, self.fear_threshold_percentile)
        
        return (float(greed_threshold), float(fear_threshold))
    
    def _detect_market_condition(
        self,
        sentiment_score: float,
        volatility: float,
        volume_metrics: Dict[str, float],
        greed_threshold: float,
        fear_threshold: float
    ) -> str:
        """
        Detect market condition: GREEDY, FEARFUL, or NEUTRAL.
        
        Uses sentiment + volatility + volume combined analysis.
        Requires sentiment to match direction (positive for greed, negative for fear).
        """
        # Check for GREEDY (others are greedy)
        # REQUIRES: sentiment must be positive AND exceed threshold
        sentiment_greedy = sentiment_score >= greed_threshold
        volatility_high = volatility > 0.02
        volume_high = volume_metrics.get("volume_ratio", 1.0) > 1.2
        
        # Check for FEARFUL (others are fearful)
        # REQUIRES: sentiment must be negative AND below threshold
        sentiment_fearful = sentiment_score <= fear_threshold
        # Same volatility and volume checks apply
        
        # For GREEDY: sentiment must be greedy + at least one other condition
        if sentiment_greedy and (volatility_high or volume_high):
            return "GREEDY"
        
        # For FEARFUL: sentiment must be fearful + at least one other condition
        if sentiment_fearful and (volatility_high or volume_high):
            return "FEARFUL"
        
        # Otherwise neutral
        return "NEUTRAL"
    
    def _generate_contrarian_signal(
        self,
        market_condition: str,
        sentiment_score: float,
        volatility: float,
        volume_metrics: Dict[str, float]
    ) -> str:
        """
        Generate contrarian signal based on market condition.
        
        - GREEDY (others greedy) -> SELL (take profits)
        - FEARFUL (others fearful) -> BUY (normal size, higher confidence)
        - NEUTRAL -> HOLD
        """
        if market_condition == "GREEDY":
            return "SELL"
        elif market_condition == "FEARFUL":
            return "BUY"
        else:
            return "HOLD"
    
    def _calculate_contrarian_confidence(
        self,
        market_condition: str,
        sentiment_confidence: float,
        volatility: float,
        volume_metrics: Dict[str, float]
    ) -> float:
        """
        Calculate confidence in contrarian signal.
        
        Higher confidence when:
        - Multiple indicators align (sentiment + volatility + volume)
        - Extreme conditions are clear
        - Historical context supports (like Great Depression, Internet Bust, COVID crash)
        """
        if market_condition == "NEUTRAL":
            return 0.0
        
        # Base confidence from sentiment
        base_confidence = sentiment_confidence
        
        # Boost confidence if volatility and volume confirm
        if volatility > 0.02:
            base_confidence += 0.1
        if volume_metrics.get("volume_ratio", 1.0) > 1.2:
            base_confidence += 0.1
        
        # Higher confidence for extreme conditions
        if market_condition == "GREEDY":
            # When others are greedy, confidence in SELL signal
            # Historical: Internet Bust, COVID crash - high confidence in contrarian
            base_confidence = min(1.0, base_confidence * 1.2)
        elif market_condition == "FEARFUL":
            # When others are fearful, confidence in BUY signal
            # Historical: Great Depression recovery, COVID recovery - high confidence
            base_confidence = min(1.0, base_confidence * 1.2)
        
        return min(1.0, max(0.0, base_confidence))
    
    def _build_reasoning(
        self,
        market_condition: str,
        sentiment_score: float,
        volatility: float,
        volume_metrics: Dict[str, float]
    ) -> str:
        """Build reasoning explanation for contrarian signal."""
        reasoning_parts = []
        
        if market_condition == "GREEDY":
            reasoning_parts.append(
                f"Market shows GREEDY conditions: sentiment={sentiment_score:.2f} (extreme positive), "
                f"volatility={volatility:.4f}, volume_ratio={volume_metrics.get('volume_ratio', 1.0):.2f}"
            )
            reasoning_parts.append(
                "Historical context: Similar to Internet Bust (2000) and COVID crash (2020) - "
                "extreme optimism before market correction"
            )
            reasoning_parts.append("Contrarian signal: SELL (take profits, exit positions)")
            
        elif market_condition == "FEARFUL":
            reasoning_parts.append(
                f"Market shows FEARFUL conditions: sentiment={sentiment_score:.2f} (extreme negative), "
                f"volatility={volatility:.4f}, volume_ratio={volume_metrics.get('volume_ratio', 1.0):.2f}"
            )
            reasoning_parts.append(
                "Historical context: Similar to Great Depression recovery and COVID recovery - "
                "extreme pessimism creating buying opportunity"
            )
            reasoning_parts.append("Contrarian signal: BUY (normal position size, higher confidence)")
            
        else:
            reasoning_parts.append("Market conditions are NEUTRAL - no extreme greed/fear detected")
            reasoning_parts.append("No contrarian signal - follow standard recommendation")
        
        return "; ".join(reasoning_parts)

