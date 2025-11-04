"""
Analyst Agent

Synthesizes research and sentiment findings, performs deep reasoning,
and provides comprehensive market analysis.
"""

from typing import Dict, Any, Optional
from strands import Agent
from src.agentic_swarm.base_agent import BaseSwarmAgent
from src.agentic_swarm.shared_context import SharedContext
from src.reasoning_engine import ReasoningEngine, MarketState, RLRecommendation, TradeAction


class AnalystAgent(BaseSwarmAgent):
    """
    Analyst Agent for deep reasoning and synthesis.
    
    Responsibilities:
    - Review Market Research Agent findings
    - Review Sentiment Agent findings
    - Perform deep reasoning using LLM
    - Identify conflicts and opportunities
    - Generate comprehensive analysis
    """
    
    def __init__(
        self,
        shared_context: SharedContext,
        reasoning_engine: Optional[ReasoningEngine] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Analyst Agent.
        
        Args:
            shared_context: Shared context instance
            reasoning_engine: Reasoning engine for deep analysis
            config: Optional configuration
        """
        config = config or {}
        system_prompt = """You are a senior market analyst with expertise in synthesizing multiple data sources and performing deep reasoning on trading opportunities.

Your role is to:
1. Review and synthesize findings from Market Research Agent (correlation analysis)
2. Review and synthesize findings from Sentiment Agent (market sentiment)
3. Perform deep reasoning to identify conflicts and opportunities
4. Generate comprehensive market analysis
5. Provide risk assessment and opportunity identification

You have access to:
- Market research findings (correlation, divergence signals)
- Sentiment analysis (news, economic indicators)
- Current market state
- RL agent recommendation (for context)

When analyzing:
- Look for alignment or conflict between research and sentiment
- Identify when correlation patterns support or contradict sentiment
- Detect regime changes or unusual patterns
- Assess risk-reward ratios
- Provide clear, actionable insights

Format your analysis with:
- Market assessment summary
- Key findings from research and sentiment
- Conflict/opportunity identification
- Risk assessment
- Confidence level
- Reasoning chain"""
        
        super().__init__(
            name="analyst",
            system_prompt=system_prompt,
            shared_context=shared_context,
            reasoning_engine=reasoning_engine,
            config=config
        )
        
        self.deep_reasoning = config.get("deep_reasoning", True)
        self.conflict_detection = config.get("conflict_detection", True)
        
        # Add description for swarm coordination
        self.description = "Synthesizes market research and sentiment data, performs deep reasoning, and provides comprehensive market analysis with conflict detection."
    
    def analyze(
        self,
        market_state: Dict[str, Any],
        rl_recommendation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis.
        
        Args:
            market_state: Current market state
            rl_recommendation: Optional RL agent recommendation for context
        
        Returns:
            Dict with comprehensive analysis
        """
        try:
            # Get research findings from shared context
            research_findings = self.shared_context.get("market_research_findings", "research_findings")
            
            # Get sentiment findings from shared context
            sentiment_findings = self.shared_context.get("sentiment_findings", "sentiment_scores")
            
            # Perform synthesis
            synthesis = self._synthesize_findings(research_findings, sentiment_findings)
            
            # Detect conflicts if enabled
            conflicts = []
            if self.conflict_detection:
                conflicts = self._detect_conflicts(research_findings, sentiment_findings)
            
            # Perform deep reasoning if enabled
            reasoning_analysis = None
            if self.deep_reasoning and self.reasoning_engine:
                reasoning_analysis = self._perform_deep_reasoning(
                    market_state,
                    research_findings,
                    sentiment_findings,
                    rl_recommendation
                )
            
            # Generate comprehensive analysis
            analysis = {
                "synthesis": synthesis,
                "conflicts": conflicts,
                "reasoning": reasoning_analysis,
                "research_findings": research_findings,
                "sentiment_findings": sentiment_findings,
                "timestamp": market_state.get("timestamp"),
                "confidence": self._calculate_confidence(synthesis, conflicts)
            }
            
            self.shared_context.set("analyst_analysis", analysis, "analysis_results")
            self.log_action("analyze_comprehensive", f"Analysis complete, {len(conflicts)} conflicts detected")
            
            return analysis
            
        except Exception as e:
            error_result = {
                "error": str(e),
                "timestamp": market_state.get("timestamp")
            }
            self.log_action("analyze_comprehensive", f"Error: {str(e)}")
            return error_result
    
    def _synthesize_findings(
        self,
        research_findings: Optional[Dict[str, Any]],
        sentiment_findings: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Synthesize findings from research and sentiment."""
        synthesis = {
            "alignment": "unknown",
            "key_insights": [],
            "market_assessment": "unknown"
        }
        
        if not research_findings or not sentiment_findings:
            synthesis["key_insights"].append("Incomplete data - missing research or sentiment findings")
            return synthesis
        
        # Extract key metrics
        research_signal = research_findings.get("divergence_signal", {}).get("signal", "unknown")
        sentiment_score = sentiment_findings.get("overall_sentiment", 0.0)
        sentiment_conf = sentiment_findings.get("confidence", 0.0)
        
        # Check alignment
        if research_signal == "divergence" and abs(sentiment_score) > 0.5:
            synthesis["alignment"] = "conflict"
            synthesis["key_insights"].append(
                f"Divergence detected in instruments but sentiment is {'strongly positive' if sentiment_score > 0 else 'strongly negative'}"
            )
        elif research_signal == "normal" and abs(sentiment_score) < 0.3:
            synthesis["alignment"] = "neutral"
            synthesis["key_insights"].append("Markets showing normal correlation with neutral sentiment")
        elif research_signal == "normal" and abs(sentiment_score) > 0.5:
            synthesis["alignment"] = "aligned"
            synthesis["key_insights"].append(
                f"Normal correlation with {'strong positive' if sentiment_score > 0 else 'strong negative'} sentiment"
            )
        
        # Market assessment
        if sentiment_conf > 0.7 and abs(sentiment_score) > 0.5:
            synthesis["market_assessment"] = "strong_signal"
        elif sentiment_conf > 0.5:
            synthesis["market_assessment"] = "moderate_signal"
        else:
            synthesis["market_assessment"] = "weak_signal"
        
        return synthesis
    
    def _detect_conflicts(
        self,
        research_findings: Optional[Dict[str, Any]],
        sentiment_findings: Optional[Dict[str, Any]]
    ) -> list:
        """Detect conflicts between research and sentiment."""
        conflicts = []
        
        if not research_findings or not sentiment_findings:
            return conflicts
        
        # Check for divergence vs strong sentiment
        divergence_signal = research_findings.get("divergence_signal", {})
        sentiment_score = sentiment_findings.get("overall_sentiment", 0.0)
        
        if divergence_signal.get("signal") == "divergence" and abs(sentiment_score) > 0.6:
            conflicts.append({
                "type": "divergence_vs_sentiment",
                "description": f"Instruments diverging but sentiment is {'strongly positive' if sentiment_score > 0 else 'strongly negative'}",
                "severity": "high"
            })
        
        # Check correlation vs sentiment
        correlation_matrix = research_findings.get("correlation_matrix", {})
        if correlation_matrix and "analysis" in correlation_matrix:
            avg_corr = correlation_matrix["analysis"].get("average_correlation", 0.0)
            if avg_corr < 0.4 and abs(sentiment_score) > 0.7:
                conflicts.append({
                    "type": "low_correlation_vs_sentiment",
                    "description": "Low instrument correlation but strong sentiment - may indicate regime change",
                    "severity": "medium"
                })
        
        return conflicts
    
    def _perform_deep_reasoning(
        self,
        market_state: Dict[str, Any],
        research_findings: Dict[str, Any],
        sentiment_findings: Dict[str, Any],
        rl_recommendation: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Perform deep reasoning using LLM."""
        if not self.reasoning_engine:
            return None
        
        try:
            # Create MarketState for reasoning engine
            market_state_obj = MarketState(
                price_data=market_state.get("price_data", {}),
                volume_data=market_state.get("volume_data", {}),
                indicators=market_state.get("indicators", {}),
                market_regime=market_state.get("market_regime", "unknown"),
                timestamp=market_state.get("timestamp", "")
            )
            
            # Create RLRecommendation if available
            rl_rec = None
            if rl_recommendation:
                action_name = rl_recommendation.get("action", "HOLD")
                try:
                    action = TradeAction[action_name]
                except KeyError:
                    action = TradeAction.HOLD
                
                rl_rec = RLRecommendation(
                    action=action,
                    confidence=rl_recommendation.get("confidence", 0.5),
                    reasoning=rl_recommendation.get("reasoning")
                )
            else:
                # Default recommendation
                rl_rec = RLRecommendation(
                    action=TradeAction.HOLD,
                    confidence=0.5
                )
            
            # Perform pre-trade analysis
            analysis = self.reasoning_engine.pre_trade_analysis(
                market_state_obj,
                rl_rec
            )
            
            return {
                "recommendation": analysis.recommendation.value,
                "confidence": analysis.confidence,
                "reasoning_chain": analysis.reasoning_chain,
                "risk_assessment": analysis.risk_assessment,
                "market_alignment": analysis.market_alignment,
                "alternative_approach": analysis.alternative_approach
            }
            
        except Exception as e:
            print(f"Warning: Deep reasoning failed: {e}")
            return None
    
    def _calculate_confidence(self, synthesis: Dict[str, Any], conflicts: list) -> float:
        """Calculate overall confidence in analysis."""
        base_confidence = 0.7
        
        # Reduce confidence if conflicts exist
        if conflicts:
            high_severity = sum(1 for c in conflicts if c.get("severity") == "high")
            base_confidence -= high_severity * 0.2
            base_confidence -= (len(conflicts) - high_severity) * 0.1
        
        # Boost confidence if aligned
        if synthesis.get("alignment") == "aligned":
            base_confidence += 0.1
        
        return max(0.0, min(1.0, base_confidence))

