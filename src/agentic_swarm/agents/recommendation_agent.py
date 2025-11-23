"""
Recommendation Agent

Makes final buy/sell/hold recommendations with risk management integration.
"""

from typing import Dict, Any, Optional
from strands import Agent
from src.agentic_swarm.base_agent import BaseSwarmAgent
from src.agentic_swarm.shared_context import SharedContext
from src.reasoning_engine import ReasoningEngine
from src.risk_manager import RiskManager


class RecommendationAgent(BaseSwarmAgent):
    """
    Recommendation Agent for final trading decisions.
    
    Responsibilities:
    - Synthesize all agent inputs
    - Generate actionable recommendation
    - Apply risk management constraints
    - Create execution plan
    """
    
    def __init__(
        self,
        shared_context: SharedContext,
        risk_manager: RiskManager,
        reasoning_engine: Optional[ReasoningEngine] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Recommendation Agent.
        
        Args:
            shared_context: Shared context instance
            risk_manager: Risk manager instance
            reasoning_engine: Optional reasoning engine
            config: Optional configuration
        """
        config = config or {}
        system_prompt = """You are a senior trading strategist making final trading recommendations based on comprehensive market analysis.

Your role is to:
1. Synthesize findings from Market Research, Sentiment, and Analyst agents
2. Consider RL agent recommendation (for context)
3. Generate actionable buy/sell/hold recommendation
4. Apply risk management constraints
5. Calculate appropriate position size
6. Set stop loss and take profit levels

You have access to:
- Analyst Agent comprehensive analysis
- Market Research Agent correlation findings
- Sentiment Agent sentiment scores
- Risk Manager for risk constraints
- Current position and market state

When making recommendations:
- Prioritize risk management above all
- Consider correlation signals for confirmation
- Use sentiment to gauge market mood
- Apply conservative position sizing
- Always set stop losses
- Provide clear reasoning for your recommendation

Format your recommendation with:
- Action: BUY, SELL, or HOLD
- Position size (normalized -1.0 to +1.0)
- Confidence level (0.0 to 1.0)
- Stop loss level
- Take profit level
- Reasoning summary
- Risk assessment"""
        
        super().__init__(
            name="recommendation",
            system_prompt=system_prompt,
            shared_context=shared_context,
            reasoning_engine=reasoning_engine,
            config=config
        )
        
        self.risk_manager = risk_manager
        self.risk_integration = config.get("risk_integration", True)
        self.position_sizing = config.get("position_sizing", True)
        
        # Add description for swarm coordination
        self.description = "Makes final trading recommendations (buy/sell/hold) with risk management, position sizing, and stop loss calculations."
    
    def recommend(
        self,
        market_state: Dict[str, Any],
        rl_recommendation: Optional[Dict[str, Any]] = None,
        current_position: float = 0.0
    ) -> Dict[str, Any]:
        """
        Generate final trading recommendation.
        
        Args:
            market_state: Current market state
            rl_recommendation: RL agent recommendation (for context)
            current_position: Current position size
        
        Returns:
            Dict with final recommendation
        """
        try:
            # Get all agent findings from shared context
            analyst_analysis = self.shared_context.get("analyst_analysis", "analysis_results")
            research_findings = self.shared_context.get("market_research_findings", "research_findings")
            sentiment_findings = self.shared_context.get("sentiment_findings", "sentiment_scores")
            contrarian_analysis = self.shared_context.get("contrarian_analysis", "contrarian_signals")
            elliott_analysis = self.shared_context.get("elliott_wave_analysis", "analysis_results")
            
            # Synthesize recommendation
            recommendation = self._synthesize_recommendation(
                analyst_analysis,
                research_findings,
                sentiment_findings,
                rl_recommendation,
                market_state,
                contrarian_analysis,
                elliott_analysis
            )
            
            # Apply risk management
            if self.risk_integration and recommendation is not None:
                risk_result = self._apply_risk_management(
                    recommendation,
                    market_state,
                    current_position
                )
                if risk_result is not None:
                    recommendation = risk_result
            
            # Calculate position size
            if self.position_sizing and recommendation is not None:
                size_result = self._calculate_position_size(
                    recommendation,
                    market_state,
                    analyst_analysis
                )
                if size_result is not None:
                    recommendation = size_result
            
            # Set stop loss and take profit
            if recommendation is not None:
                risk_levels_result = self._set_risk_levels(
                    recommendation,
                    market_state
                )
                if risk_levels_result is not None:
                    recommendation = risk_levels_result
            
            # Store final recommendation
            self.shared_context.set("final_recommendation", recommendation, "recommendation")
            self.log_action(
                "generate_recommendation",
                f"{recommendation.get('action')} @ {recommendation.get('position_size', 0):.2f}, Confidence: {recommendation.get('confidence', 0):.2f}"
            )
            
            return recommendation
            
        except Exception as e:
            error_result = {
                "error": str(e),
                "action": "HOLD",
                "position_size": 0.0,
                "confidence": 0.0,
                "timestamp": market_state.get("timestamp")
            }
            self.log_action("generate_recommendation", f"Error: {str(e)}")
            return error_result
    
    def _synthesize_recommendation(
        self,
        analyst_analysis: Optional[Dict[str, Any]],
        research_findings: Optional[Dict[str, Any]],
        sentiment_findings: Optional[Dict[str, Any]],
        rl_recommendation: Optional[Dict[str, Any]],
        market_state: Dict[str, Any],
        contrarian_analysis: Optional[Dict[str, Any]] = None,
        elliott_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synthesize recommendation from all inputs."""
        # Default recommendation
        recommendation = {
            "action": "HOLD",
            "position_size": 0.0,
            "confidence": 0.5,
            "reasoning": "Insufficient data for recommendation"
        }
        
        if not analyst_analysis:
            return recommendation
        
        # Extract key signals
        synthesis = analyst_analysis.get("synthesis", {})
        alignment = synthesis.get("alignment", "unknown")
        sentiment_score = sentiment_findings.get("overall_sentiment", 0.0) if sentiment_findings else 0.0
        sentiment_conf = sentiment_findings.get("confidence", 0.0) if sentiment_findings else 0.0
        
        # Get RL recommendation for context
        rl_action = rl_recommendation.get("action", "HOLD") if rl_recommendation else "HOLD"
        rl_confidence = rl_recommendation.get("confidence", 0.5) if rl_recommendation else 0.5
        
        # Determine action based on alignment and signals
        if alignment == "aligned" and abs(sentiment_score) > 0.5 and sentiment_conf > 0.6:
            # Strong alignment - follow sentiment
            if sentiment_score > 0.3:
                recommendation["action"] = "BUY"
                recommendation["position_size"] = min(0.8, abs(sentiment_score) * 0.8)
            elif sentiment_score < -0.3:
                recommendation["action"] = "SELL"
                recommendation["position_size"] = min(0.8, abs(sentiment_score) * 0.8)
            else:
                recommendation["action"] = "HOLD"
        elif alignment == "conflict":
            # Conflict detected - be conservative
            recommendation["action"] = "HOLD"
            recommendation["position_size"] = 0.0
            recommendation["reasoning"] = "Conflicting signals detected - holding position"
        elif rl_confidence > 0.5 and rl_action != "HOLD":
            # High RL confidence - follow RL but reduce size due to conflicts
            recommendation["action"] = rl_action
            recommendation["position_size"] = rl_confidence * 0.7  # Reduced due to conflicts
        else:
            # Default to HOLD
            recommendation["action"] = "HOLD"
        
        # Calculate confidence
        confidence = analyst_analysis.get("confidence", 0.5)
        if sentiment_conf > 0.7:
            confidence = (confidence + sentiment_conf) / 2
        
        recommendation["confidence"] = min(1.0, confidence)
        
        # Apply contrarian influence if extreme conditions detected
        if contrarian_analysis:
            recommendation = self._apply_contrarian_influence(
                recommendation,
                contrarian_analysis
            )
        
        if elliott_analysis:
            recommendation = self._apply_elliott_influence(
                recommendation,
                elliott_analysis
            )
            recommendation["elliott_wave"] = elliott_analysis
            recommendation["elliott_wave_action"] = elliott_analysis.get("action", "HOLD")
            recommendation["elliott_wave_confidence"] = elliott_analysis.get("confidence", 0.0)
            recommendation["elliott_wave_phase"] = elliott_analysis.get("phase")
            recommendation["elliott_wave_bias"] = elliott_analysis.get("bias")
        
        # Build reasoning
        reasoning_parts = []
        if alignment != "unknown":
            reasoning_parts.append(f"Market alignment: {alignment}")
        if sentiment_findings:
            reasoning_parts.append(f"Sentiment: {sentiment_score:.2f} (confidence: {sentiment_conf:.2f})")
        if rl_recommendation:
            reasoning_parts.append(f"RL recommendation: {rl_action} (confidence: {rl_confidence:.2f})")
        if contrarian_analysis:
            contrarian_signal = contrarian_analysis.get("contrarian_signal", "HOLD")
            contrarian_conf = contrarian_analysis.get("contrarian_confidence", 0.0)
            market_condition = contrarian_analysis.get("market_condition", "NEUTRAL")
            reasoning_parts.append(f"Contrarian: {market_condition} -> {contrarian_signal} (confidence: {contrarian_conf:.2f})")
        if elliott_analysis:
            ew_action = elliott_analysis.get("action", "HOLD")
            ew_conf = elliott_analysis.get("confidence", 0.0)
            ew_phase = elliott_analysis.get("phase")
            reasoning_parts.append(
                f"Elliott Wave: {ew_action} (phase: {ew_phase}, confidence: {ew_conf:.2f})"
            )
        
        recommendation["reasoning"] = "; ".join(reasoning_parts) if reasoning_parts else "Insufficient data"
        
        return recommendation
    
    def _apply_elliott_influence(
        self,
        recommendation: Dict[str, Any],
        elliott_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Blend Elliott Wave agent signal into the recommendation.

        Prefers alignment with the existing action; when the base plan is HOLD,
        the Elliott signal can promote a trade if confidence surpasses the
        configured minimum.
        """
        elliott_action = elliott_analysis.get("action", "HOLD")
        elliott_confidence = elliott_analysis.get("confidence", 0.0)
        elliott_position = elliott_analysis.get("position_size", 0.0)
        elliott_phase = elliott_analysis.get("phase")
        min_conf = elliott_analysis.get("min_confidence", 0.55)
        
        if elliott_action not in {"BUY", "SELL"} or elliott_confidence < min_conf:
            return recommendation

        current_action = recommendation.get("action", "HOLD")
        if current_action == "HOLD":
            recommendation["action"] = elliott_action
            recommendation["position_size"] = max(
                recommendation.get("position_size", 0.0),
                elliott_position
            )
            recommendation["confidence"] = max(
                recommendation.get("confidence", 0.0),
                elliott_confidence
            )
            recommendation["reasoning"] = (
                f"Elliott Wave {elliott_phase} signal adopted (confidence {elliott_confidence:.2f})"
            )
        elif current_action == elliott_action:
            recommendation["position_size"] = max(
                recommendation.get("position_size", 0.0),
                elliott_position
            )
            recommendation["confidence"] = max(
                recommendation.get("confidence", 0.0),
                elliott_confidence
            )
        else:
            recommendation.setdefault("warnings", []).append(
                f"Elliott Wave {elliott_phase} favors {elliott_action} "
                f"with confidence {elliott_confidence:.2f}"
            )
            recommendation["confidence"] = max(
                recommendation.get("confidence", 0.0),
                elliott_confidence * 0.6
            )
        
        return recommendation
    
    def _apply_contrarian_influence(
        self,
        recommendation: Dict[str, Any],
        contrarian_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply contrarian influence to sway recommendation when extremes detected.
        
        When contrarian signal is strong, it overrides/sways the recommendation.
        """
        market_condition = contrarian_analysis.get("market_condition", "NEUTRAL")
        contrarian_signal = contrarian_analysis.get("contrarian_signal", "HOLD")
        contrarian_confidence = contrarian_analysis.get("contrarian_confidence", 0.0)
        
        # Only apply if contrarian confidence is high (>= 0.6)
        if contrarian_confidence < 0.6 or market_condition == "NEUTRAL":
            return recommendation
        
        # When extreme conditions detected, contrarian signal overrides/sways
        if market_condition == "GREEDY" and contrarian_signal == "SELL":
            # Others are greedy - override to SELL
            recommendation["action"] = "SELL"
            recommendation["position_size"] = min(0.8, recommendation.get("position_size", 0.5) * 1.2)
            recommendation["confidence"] = min(1.0, recommendation.get("confidence", 0.5) + contrarian_confidence * 0.3)
            recommendation["reasoning"] = f"Contrarian override: {contrarian_analysis.get('reasoning', 'Extreme greed detected')}"
            
        elif market_condition == "FEARFUL" and contrarian_signal == "BUY":
            # Others are fearful - override to BUY with normal size but higher confidence
            recommendation["action"] = "BUY"
            recommendation["position_size"] = recommendation.get("position_size", 0.5)  # Normal size
            recommendation["confidence"] = min(1.0, recommendation.get("confidence", 0.5) + contrarian_confidence * 0.4)
            recommendation["reasoning"] = f"Contrarian override: {contrarian_analysis.get('reasoning', 'Extreme fear detected')}"
        
        # Add contrarian metadata
        recommendation["contrarian_confidence"] = contrarian_confidence
        recommendation["contrarian_signal"] = contrarian_signal
        recommendation["market_condition"] = market_condition
        
        return recommendation
    
    def _apply_risk_management(
        self,
        recommendation: Dict[str, Any],
        market_state: Dict[str, Any],
        current_position: float
    ) -> Dict[str, Any]:
        """Apply risk management constraints."""
        if not self.risk_manager:
            return recommendation
        
        try:
            # Convert action to action value for RiskManager
            action_str = recommendation.get("action", "HOLD")
            if action_str == "BUY":
                action_value = recommendation.get("position_size", 0.5)
            elif action_str == "SELL":
                action_value = -recommendation.get("position_size", 0.5)
            else:
                action_value = 0.0
            
            # Create market data dict for RiskManager
            market_data_dict = {
                "price": market_state.get("price_data", {}).get("close", 0.0),
                "high": market_state.get("price_data", {}).get("high", 0.0),
                "low": market_state.get("price_data", {}).get("low", 0.0),
                "volume": market_state.get("volume_data", {}).get("volume", 0.0)
            }
            
            # Validate with RiskManager
            # Note: validate_action now returns (position, monte_carlo_result)
            result = self.risk_manager.validate_action(
                target_position=action_value,
                current_position=current_position,
                market_data=market_data_dict,
                current_price=market_state.get("price_data", {}).get("close"),
                decision_context={
                    "source": "swarm_recommendation"
                },
                instrument=market_state.get("instrument", "default")
            )
            
            # Handle tuple return (position, monte_carlo_result)
            if isinstance(result, tuple):
                validated_value, _ = result  # Ignore monte_carlo_result for now
            else:
                # Backward compatibility
                validated_value = result
            
            # Update recommendation based on risk manager
            recommendation["position_size"] = abs(validated_value)
            
            if validated_value == 0.0:
                recommendation["action"] = "HOLD"
                current_reasoning = recommendation.get("reasoning", "")
                recommendation["reasoning"] = current_reasoning + " (Risk manager: position size reduced to 0)"
            elif abs(validated_value) < abs(action_value):
                current_reasoning = recommendation.get("reasoning", "")
                recommendation["reasoning"] = current_reasoning + f" (Risk manager: position size reduced from {abs(action_value):.2f} to {abs(validated_value):.2f})"
            
            # Get risk status
            risk_status = self.risk_manager.get_risk_status()
            recommendation["risk_status"] = risk_status
            
        except Exception as e:
            print(f"Warning: Risk management application failed: {e}")
            # Return original recommendation on error
            return recommendation
        
        return recommendation
    
    def _calculate_position_size(
        self,
        recommendation: Dict[str, Any],
        market_state: Dict[str, Any],
        analyst_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate appropriate position size."""
        if analyst_analysis:
            confidence = analyst_analysis.get("confidence", 0.5)
            conflicts = analyst_analysis.get("conflicts", [])
            
            # Reduce position size if conflicts exist
            if conflicts:
                high_severity = sum(1 for c in conflicts if c.get("severity") == "high")
                position_reduction = high_severity * 0.2 + len(conflicts) * 0.1
                recommendation["position_size"] = max(0.0, recommendation["position_size"] - position_reduction)
            
            # Scale position by confidence
            if recommendation.get("position_size") is not None:
                recommendation["position_size"] = recommendation["position_size"] * confidence
        
        # Ensure position size is within bounds
        if recommendation.get("position_size") is not None:
            recommendation["position_size"] = max(0.0, min(1.0, recommendation["position_size"]))
        else:
            recommendation["position_size"] = 0.0
        
        return recommendation
    
    def _set_risk_levels(
        self,
        recommendation: Dict[str, Any],
        market_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Set stop loss and take profit levels."""
        if not self.risk_manager:
            return recommendation
        
        try:
            current_price = market_state.get("price_data", {}).get("close", 0.0)
            position_size = recommendation.get("position_size", 0.0)
            
            if position_size == 0.0 or recommendation.get("action") == "HOLD":
                recommendation["stop_loss"] = None
                recommendation["take_profit"] = None
                return recommendation
            
            # Create market data dict for RiskManager
            market_data_dict = {
                "price": current_price,
                "high": market_state.get("price_data", {}).get("high", 0.0),
                "low": market_state.get("price_data", {}).get("low", 0.0),
                "volume": market_state.get("volume_data", {}).get("volume", 0.0)
            }
            
            # Calculate stop loss
            stop_loss = self.risk_manager.calculate_stop_loss(
                entry_price=current_price,
                position_size=position_size,
                market_data=market_data_dict
            )
            
            recommendation["stop_loss"] = float(stop_loss) if stop_loss else None
            
            # Set take profit (2x stop loss distance as default)
            if recommendation["stop_loss"] and recommendation["action"] != "HOLD":
                if recommendation["action"] == "BUY":
                    distance = current_price - recommendation["stop_loss"]
                    recommendation["take_profit"] = current_price + (distance * 2.0)
                elif recommendation["action"] == "SELL":
                    distance = recommendation["stop_loss"] - current_price
                    recommendation["take_profit"] = current_price - (distance * 2.0)
                else:
                    recommendation["take_profit"] = None
            else:
                recommendation["take_profit"] = None
                
        except Exception as e:
            print(f"Warning: Risk level calculation failed: {e}")
            if recommendation is not None:
                recommendation["stop_loss"] = None
                recommendation["take_profit"] = None
        
        return recommendation if recommendation is not None else {}

