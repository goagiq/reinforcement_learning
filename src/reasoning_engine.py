"""
Deep Reasoning & Reflection Engine for NT8 RL Trading Strategy
Supports multiple LLM providers: Ollama, DeepSeek Cloud API, Grok (xAI)
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from src.llm_providers import get_provider, BaseLLMProvider


class TradeAction(Enum):
    """Trading actions"""
    HOLD = 0
    BUY = 1
    SELL = 2
    EXIT_LONG = 3
    EXIT_SHORT = 4


class RecommendationType(Enum):
    """Recommendation types from reasoning engine"""
    APPROVE = "approve"
    MODIFY = "modify"
    REJECT = "reject"


@dataclass
class MarketState:
    """Market state information"""
    price_data: Dict  # OHLC data
    volume_data: Dict  # Volume information
    indicators: Dict  # Technical indicators
    market_regime: str  # trending, ranging, volatile
    timestamp: str


@dataclass
class RLRecommendation:
    """RL model recommendation"""
    action: TradeAction
    confidence: float  # 0.0 to 1.0
    reasoning: Optional[str] = None


@dataclass
class ReasoningAnalysis:
    """Result from reasoning engine analysis"""
    recommendation: RecommendationType
    confidence: float  # 0.0 to 1.0
    reasoning_chain: str  # Step-by-step reasoning
    risk_assessment: Dict
    market_alignment: str  # "aligned", "partially_aligned", "misaligned"
    alternative_approach: Optional[str] = None


@dataclass
class TradeResult:
    """Completed trade result"""
    action: TradeAction
    entry_price: float
    exit_price: float
    pnl: float
    duration_seconds: float
    market_conditions: str
    timestamp: str


@dataclass
class ReflectionInsight:
    """Post-trade reflection insights"""
    success_factors: List[str]
    failure_factors: List[str]
    warning_signs: List[str]
    market_impact: str
    learnable_patterns: List[str]
    adaptation_recommendations: List[str]
    reasoning: str


class ReasoningEngine:
    """Deep reasoning and reflection engine using configurable LLM providers"""
    
    def __init__(
        self,
        provider_type: str = "ollama",
        model: str = "deepseek-r1:8b",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 600,
        keep_alive: str = "10m",  # Keep model loaded in memory (default 10 minutes)
        use_kong: bool = False,  # Route through Kong Gateway
        kong_api_key: Optional[str] = None  # Kong consumer API key
    ):
        """
        Initialize reasoning engine with specified provider.
        
        Args:
            provider_type: "ollama", "deepseek_cloud", or "grok"
            model: Model name (provider-specific)
            api_key: API key (required for cloud providers, unless using Kong)
            base_url: Base URL (optional, uses provider defaults if not provided, ignored if use_kong=True)
            timeout: Request timeout in seconds
            keep_alive: Keep model loaded in memory for Ollama (e.g., "10m", "5m", "0" for no keep-alive)
            use_kong: Route requests through Kong Gateway
            kong_api_key: Kong consumer API key (required if use_kong=True)
        """
        self.provider_type = provider_type.lower()
        self.model = model
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.use_kong = use_kong
        
        # Initialize provider
        provider_kwargs = {
            "use_kong": use_kong,
            "kong_api_key": kong_api_key
        }
        
        if self.provider_type == "ollama":
            provider_kwargs["base_url"] = base_url or "http://localhost:11434"
        elif self.provider_type in ["deepseek_cloud", "grok"]:
            if not use_kong and not api_key:
                raise ValueError(f"api_key is required for {self.provider_type} provider (unless using Kong)")
            provider_kwargs["api_key"] = api_key
            if base_url and not use_kong:
                provider_kwargs["base_url"] = base_url
        
        self.provider = get_provider(self.provider_type, **provider_kwargs)
    
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None, stream: bool = False) -> str:
        """Call LLM provider using unified interface"""
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Adjust model name for cloud providers if needed
        model_name = self.model
        if self.provider_type == "deepseek_cloud":
            # DeepSeek Cloud uses different model names
            if "deepseek-r1" in model_name.lower():
                model_name = "deepseek-chat"  # Use DeepSeek Chat for now (R1 may need different endpoint)
        elif self.provider_type == "grok":
            # Grok model names
            if "grok" not in model_name.lower():
                model_name = "grok-beta"  # Default Grok model
        
        # Pass keep_alive for Ollama provider (ignored by other providers)
        chat_kwargs = {
            "messages": messages,
            "model": model_name,
            "temperature": 0.3,  # Lower temperature for more focused reasoning
            "top_p": 0.9,
            "stream": stream,
            "timeout": self.timeout
        }
        
        # Add keep_alive for Ollama provider
        if self.provider_type == "ollama":
            chat_kwargs["keep_alive"] = self.keep_alive
        
        return self.provider.chat(**chat_kwargs)
    
    def pre_trade_analysis(
        self,
        market_state: MarketState,
        rl_recommendation: RLRecommendation
    ) -> ReasoningAnalysis:
        """
        Perform pre-trade reasoning analysis.
        
        Args:
            market_state: Current market state
            rl_recommendation: Recommendation from RL model
        
        Returns:
            ReasoningAnalysis with recommendation and reasoning
        """
        system_prompt = """You are an expert trading analyst with deep knowledge of market dynamics, 
price action, and volume analysis. You provide careful, step-by-step reasoning for trading decisions.
Always prioritize risk management and capital preservation."""
        
        prompt = f"""You are analyzing a potential trading decision.

Market State:
- Price action: {json.dumps(market_state.price_data, indent=2)}
- Volume: {json.dumps(market_state.volume_data, indent=2)}
- Indicators: {json.dumps(market_state.indicators, indent=2)}
- Market regime: {market_state.market_regime}
- Timestamp: {market_state.timestamp}

RL Model Recommendation:
- Action: {rl_recommendation.action.name}
- Confidence: {rl_recommendation.confidence:.2%}

Please analyze step-by-step:
1. Does this recommendation align with current market conditions? (Explain why)
2. What are the specific risks associated with this trade? (List and assess each)
3. What market patterns support or contradict this decision? (Be specific)
4. What would be an alternative approach? (If different from RL recommendation)
5. Risk-reward assessment: What's the potential reward vs potential loss?
6. Final recommendation: Should we APPROVE, MODIFY, or REJECT this trade? Why?

Provide your reasoning in a clear, structured format. End with:
RECOMMENDATION: [APPROVE/MODIFY/REJECT]
CONFIDENCE: [0.0-1.0]
RISK_LEVEL: [LOW/MEDIUM/HIGH]
"""
        
        response = self._call_llm(prompt, system_prompt)
        
        # Parse response
        analysis = self._parse_pre_trade_response(response, rl_recommendation)
        return analysis
    
    def _parse_pre_trade_response(
        self,
        response: str,
        rl_recommendation: RLRecommendation
    ) -> ReasoningAnalysis:
        """Parse the reasoning response into structured format"""
        # Extract recommendation
        recommendation = RecommendationType.APPROVE
        if "REJECT" in response.upper():
            recommendation = RecommendationType.REJECT
        elif "MODIFY" in response.upper():
            recommendation = RecommendationType.MODIFY
        
        # Extract confidence (look for confidence score in response)
        confidence = 0.7  # default
        import re
        conf_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response, re.IGNORECASE)
        if conf_match:
            confidence = float(conf_match.group(1))
            if confidence > 1.0:
                confidence = confidence / 100.0  # If given as percentage
        
        # Extract risk level
        risk_level = "MEDIUM"
        if "RISK_LEVEL: LOW" in response.upper():
            risk_level = "LOW"
        elif "RISK_LEVEL: HIGH" in response.upper():
            risk_level = "HIGH"
        
        # Determine market alignment
        market_alignment = "partially_aligned"
        if "align" in response.lower() and "not" not in response.lower()[:100]:
            market_alignment = "aligned"
        elif "contradict" in response.lower() or "misalign" in response.lower():
            market_alignment = "misaligned"
        
        # Risk assessment
        risk_assessment = {
            "level": risk_level,
            "factors": self._extract_risk_factors(response),
            "potential_loss": self._extract_potential_loss(response),
            "risk_reward_ratio": self._extract_risk_reward(response)
        }
        
        return ReasoningAnalysis(
            recommendation=recommendation,
            confidence=confidence,
            reasoning_chain=response,
            risk_assessment=risk_assessment,
            market_alignment=market_alignment,
            alternative_approach=self._extract_alternative(response)
        )
    
    def post_trade_reflection(self, trade_result: TradeResult) -> ReflectionInsight:
        """
        Perform post-trade reflection and generate learning insights.
        
        Args:
            trade_result: Completed trade result
        
        Returns:
            ReflectionInsight with analysis and recommendations
        """
        system_prompt = """You are an expert trading coach who analyzes completed trades to extract 
valuable lessons. You identify patterns, assess performance factors, and provide actionable insights 
for improving future trading decisions."""
        
        prompt = f"""A trade has been completed with the following results:

Trade Details:
- Action taken: {trade_result.action.name}
- Entry price: ${trade_result.entry_price:.2f}
- Exit price: ${trade_result.exit_price:.2f}
- PnL: ${trade_result.pnl:+.2f} ({'Profit' if trade_result.pnl > 0 else 'Loss'})
- Duration: {trade_result.duration_seconds:.0f} seconds ({trade_result.duration_seconds/60:.1f} minutes)
- Market conditions during trade: {trade_result.market_conditions}
- Timestamp: {trade_result.timestamp}

Please reflect and analyze:

1. SUCCESS FACTORS (if profitable) or FAILURE FACTORS (if loss):
   What specific factors contributed to the outcome? List each factor clearly.

2. WARNING SIGNS:
   Were there any warning signs we missed before or during the trade? 
   What should we have noticed?

3. MARKET IMPACT:
   How did market conditions (trend, volatility, volume) affect the outcome?
   Be specific about which conditions helped or hurt.

4. LEARNABLE PATTERNS:
   What patterns can we learn from this trade? What worked well? What didn't?
   Identify actionable patterns.

5. ADAPTATION RECOMMENDATIONS:
   How should the RL model adapt based on this experience?
   What changes would improve future performance?

Provide detailed reasoning with specific, actionable insights.
"""
        
        response = self._call_llm(prompt, system_prompt)
        
        # Parse reflection
        insight = self._parse_reflection_response(response, trade_result)
        return insight
    
    def _parse_reflection_response(
        self,
        response: str,
        trade_result: TradeResult
    ) -> ReflectionInsight:
        """Parse reflection response into structured format"""
        is_profitable = trade_result.pnl > 0
        
        # Extract factors
        success_factors = self._extract_list_items(response, "SUCCESS FACTORS" if is_profitable else "FAILURE FACTORS")
        failure_factors = [] if is_profitable else self._extract_list_items(response, "FAILURE FACTORS")
        
        # Extract warning signs
        warning_signs = self._extract_list_items(response, "WARNING SIGNS")
        
        # Extract market impact
        market_impact = self._extract_section(response, "MARKET IMPACT")
        
        # Extract patterns
        learnable_patterns = self._extract_list_items(response, "LEARNABLE PATTERNS")
        
        # Extract recommendations
        adaptation_recommendations = self._extract_list_items(response, "ADAPTATION RECOMMENDATIONS")
        
        return ReflectionInsight(
            success_factors=success_factors,
            failure_factors=failure_factors,
            warning_signs=warning_signs,
            market_impact=market_impact,
            learnable_patterns=learnable_patterns,
            adaptation_recommendations=adaptation_recommendations,
            reasoning=response
        )
    
    def market_regime_analysis(self, market_state: MarketState) -> Dict:
        """
        Analyze current market regime using reasoning.
        
        Args:
            market_state: Current market state
        
        Returns:
            Dict with regime classification and reasoning
        """
        system_prompt = """You are a market regime analyst specializing in identifying 
trending, ranging, and volatile market conditions."""
        
        prompt = f"""Analyze the current market regime:

Market Data:
{json.dumps({
    "price_data": market_state.price_data,
    "volume_data": market_state.volume_data,
    "indicators": market_state.indicators
}, indent=2)}

Classify the market regime:
1. Primary regime: [TRENDING/RANGING/VOLATILE]
2. Sub-regime: [UPTREND/DOWNTREND/SIDEWAYS/HIGH_VOLATILITY/LOW_VOLATILITY]
3. Confidence: [0.0-1.0]
4. Key indicators supporting this classification
5. Expected duration of this regime
6. Recommended strategy adjustments

Provide reasoning for your classification.
"""
        
        response = self._call_llm(prompt, system_prompt)
        
        # Parse regime analysis
        regime_info = self._parse_regime_response(response)
        return regime_info
    
    def _parse_regime_response(self, response: str) -> Dict:
        """Parse regime analysis response"""
        regime = "ranging"  # default
        if "TRENDING" in response.upper():
            regime = "trending"
        elif "VOLATILE" in response.upper():
            regime = "volatile"
        
        return {
            "regime": regime,
            "reasoning": response,
            "confidence": 0.7,  # Default, would parse from response
            "timestamp": datetime.now().isoformat()
        }
    
    # Helper methods for parsing
    def _extract_risk_factors(self, text: str) -> List[str]:
        """Extract risk factors from text"""
        factors = []
        # Simple extraction - can be enhanced with more sophisticated parsing
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['risk', 'danger', 'threat', 'concern']):
                factors.append(line.strip())
        return factors[:5]  # Limit to 5 factors
    
    def _extract_potential_loss(self, text: str) -> Optional[str]:
        """Extract potential loss information"""
        import re
        loss_match = re.search(r'(potential|maximum).*?loss.*?:?\s*([^\n]+)', text, re.IGNORECASE)
        return loss_match.group(2).strip() if loss_match else None
    
    def _extract_risk_reward(self, text: str) -> Optional[str]:
        """Extract risk-reward ratio"""
        import re
        rr_match = re.search(r'risk.*?reward.*?:?\s*([^\n]+)', text, re.IGNORECASE)
        return rr_match.group(1).strip() if rr_match else None
    
    def _extract_alternative(self, text: str) -> Optional[str]:
        """Extract alternative approach"""
        import re
        alt_match = re.search(r'alternative.*?approach.*?:?\s*([^\n]+)', text, re.IGNORECASE | re.DOTALL)
        return alt_match.group(1).strip() if alt_match else None
    
    def _extract_list_items(self, text: str, section_name: str) -> List[str]:
        """Extract list items from a section"""
        items = []
        lines = text.split('\n')
        in_section = False
        
        for line in lines:
            if section_name.upper() in line.upper():
                in_section = True
                continue
            if in_section:
                if line.strip().startswith(('-', '*', '•', '1.', '2.')):
                    items.append(line.strip().lstrip('-*•1234567890. '))
                elif line.strip() and not line.strip().startswith(('Please', 'Provide', 'Analyze')):
                    if len(items) == 0 or len(items[-1]) < 50:  # Continue previous item if short
                        if items:
                            items[-1] += " " + line.strip()
                        else:
                            items.append(line.strip())
                if len(items) >= 5:  # Limit items
                    break
        
        return items[:5]
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a section's content"""
        lines = text.split('\n')
        in_section = False
        content = []
        
        for line in lines:
            if section_name.upper() in line.upper():
                in_section = True
                continue
            if in_section:
                if line.strip() and not any(line.strip().startswith(s) for s in ['1.', '2.', '3.', '4.', '5.', 'Please', 'Provide']):
                    content.append(line.strip())
                if len(content) > 0 and any(next_section in line.upper() for next_section in ['LEARNABLE', 'ADAPTATION', 'SUCCESS', 'FAILURE']):
                    break
        
        return " ".join(content[:200])  # Limit length


# Example usage
if __name__ == "__main__":
    engine = ReasoningEngine()
    
    # Example: Pre-trade analysis
    market_state = MarketState(
        price_data={"open": 100, "high": 102, "low": 99, "close": 101},
        volume_data={"volume": 1000000, "avg_volume": 800000},
        indicators={"rsi": 55, "macd": 0.5},
        market_regime="trending",
        timestamp=datetime.now().isoformat()
    )
    
    rl_rec = RLRecommendation(
        action=TradeAction.BUY,
        confidence=0.75
    )
    
    print("Running pre-trade analysis...")
    analysis = engine.pre_trade_analysis(market_state, rl_rec)
    print(f"Recommendation: {analysis.recommendation.value}")
    print(f"Confidence: {analysis.confidence:.2%}")
    print(f"Reasoning: {analysis.reasoning_chain[:200]}...")

