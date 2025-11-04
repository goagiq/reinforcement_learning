"""
Decision Gate Module

Combines RL agent recommendations with reasoning engine analysis
and swarm agent recommendations to make final trading decisions.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from src.reasoning_engine import ReasoningAnalysis, RecommendationType


@dataclass
class DecisionResult:
    """Final decision result"""
    action: float  # Final position size
    confidence: float  # Combined confidence
    rl_confidence: float  # RL model confidence
    reasoning_confidence: float  # Reasoning confidence
    swarm_confidence: float  # Swarm confidence (0.0 if not used)
    agreement: str  # "agree", "disagree", "uncertain", "swarm_only", "no_swarm"
    reasoning: Optional[str] = None  # Reasoning explanation
    swarm_recommendation: Optional[Dict] = None  # Swarm recommendation details


class DecisionGate:
    """
    Decision gate that combines RL and reasoning recommendations.
    
    Strategies:
    - Weighted average of confidences
    - Agreement-based position sizing
    - Conflict resolution
    """
    
    def __init__(self, config: Dict):
        """
        Initialize decision gate.
        
        Args:
            config: Configuration with decision gate settings
        """
        self.config = config
        
        # Weight configuration
        self.rl_weight = config.get("rl_weight", 0.6)  # RL weight: 60%
        self.swarm_weight = config.get("swarm_weight", 0.4)  # Swarm weight: 40%
        self.reasoning_weight = config.get("reasoning_weight", 0.0)  # Legacy, now part of swarm
        
        # Ensure weights sum to 1.0
        total_weight = self.rl_weight + self.swarm_weight
        if total_weight > 0:
            self.rl_weight = self.rl_weight / total_weight
            self.swarm_weight = self.swarm_weight / total_weight
        
        self.min_combined_confidence = config.get("min_combined_confidence", 0.7)
        self.conflict_reduction_factor = config.get("conflict_reduction_factor", 0.5)
        
        # Swarm configuration
        self.swarm_enabled = config.get("swarm_enabled", True)
        self.swarm_timeout = config.get("swarm_timeout", 20.0)
        self.fallback_to_rl_only = config.get("fallback_to_rl_only", True)
    
    def make_decision(
        self,
        rl_action: float,
        rl_confidence: float,
        reasoning_analysis: Optional[ReasoningAnalysis] = None,
        swarm_recommendation: Optional[Dict] = None
    ) -> DecisionResult:
        """
        Make final trading decision by combining RL, reasoning, and swarm recommendations.
        
        Args:
            rl_action: Action from RL agent (-1.0 to 1.0)
            rl_confidence: Confidence from RL agent (0.0 to 1.0)
            reasoning_analysis: Analysis from reasoning engine (optional, legacy)
            swarm_recommendation: Swarm recommendation dict (optional)
        
        Returns:
            DecisionResult with final action and confidence
        """
        # If no swarm recommendation and no reasoning, use RL only
        if swarm_recommendation is None and reasoning_analysis is None:
            return DecisionResult(
                action=rl_action,
                confidence=rl_confidence,
                rl_confidence=rl_confidence,
                reasoning_confidence=0.0,
                swarm_confidence=0.0,
                agreement="no_swarm",
                reasoning=None,
                swarm_recommendation=None
            )
        
        # Use swarm recommendation if available (preferred over legacy reasoning)
        if swarm_recommendation is not None:
            return self._make_decision_with_swarm(
                rl_action, rl_confidence, swarm_recommendation
            )
        
        # Fallback to legacy reasoning analysis
        return self._make_decision_with_reasoning(
            rl_action, rl_confidence, reasoning_analysis
        )
    
    def _make_decision_with_swarm(
        self,
        rl_action: float,
        rl_confidence: float,
        swarm_recommendation: Dict
    ) -> DecisionResult:
        """Make decision using RL + Swarm fusion."""
        # Extract swarm recommendation
        swarm_action_str = swarm_recommendation.get("action", "HOLD")
        swarm_action = 0.0
        if swarm_action_str == "BUY":
            swarm_action = swarm_recommendation.get("position_size", 0.5)
        elif swarm_action_str == "SELL":
            swarm_action = -swarm_recommendation.get("position_size", 0.5)
        # else HOLD: swarm_action = 0.0
        
        swarm_confidence = swarm_recommendation.get("confidence", 0.5)
        swarm_reasoning = swarm_recommendation.get("reasoning", "")
        
        # Extract contrarian confidence (if available) to adjust weights
        contrarian_confidence = swarm_recommendation.get("contrarian_confidence", 0.0)
        market_condition = swarm_recommendation.get("market_condition", "NEUTRAL")
        
        # Determine agreement
        rl_direction = np.sign(rl_action)
        swarm_direction = np.sign(swarm_action)
        
        if swarm_action == 0.0:  # Swarm says HOLD
            agreement = "swarm_hold"
        elif rl_direction * swarm_direction > 0:
            agreement = "agree"
        elif rl_direction == 0 and swarm_direction != 0:
            agreement = "swarm_only"
        elif swarm_direction == 0 and rl_direction != 0:
            agreement = "rl_only"
        else:
            agreement = "disagree"
        
        # Adjust weights based on contrarian confidence
        # When contrarian confidence is high, increase swarm weight
        adjusted_rl_weight = self.rl_weight
        adjusted_swarm_weight = self.swarm_weight
        
        if contrarian_confidence >= 0.6 and market_condition != "NEUTRAL":
            # Contrarian signal is strong - increase swarm weight
            contrarian_boost = contrarian_confidence * 0.3  # Up to 30% boost
            adjusted_swarm_weight = min(0.8, self.swarm_weight + contrarian_boost)
            adjusted_rl_weight = 1.0 - adjusted_swarm_weight
        
        # Calculate combined confidence and action
        if agreement == "agree":
            # Both agree - boost confidence
            combined_conf = min(1.0,
                adjusted_rl_weight * rl_confidence +
                adjusted_swarm_weight * swarm_confidence * 1.1  # 10% boost
            )
            # Use weighted average of actions
            final_action = (
                adjusted_rl_weight * rl_action +
                adjusted_swarm_weight * swarm_action
            )
            
        elif agreement == "swarm_hold":
            # Swarm says HOLD - reduce RL action significantly
            combined_conf = (
                adjusted_rl_weight * rl_confidence +
                adjusted_swarm_weight * swarm_confidence
            ) * 0.6  # Reduce confidence
            final_action = rl_action * 0.3  # Reduce position by 70%
            
        elif agreement == "swarm_only":
            # Swarm has signal, RL says HOLD - use swarm but reduce size
            # If contrarian signal is strong, trust it more
            if contrarian_confidence >= 0.6:
                combined_conf = swarm_confidence * 0.95  # Higher trust for contrarian
                final_action = swarm_action * 0.9  # Less reduction
            else:
                combined_conf = swarm_confidence * 0.8  # Swarm confidence reduced
                final_action = swarm_action * 0.7  # Reduce position by 30%
            
        elif agreement == "rl_only":
            # RL has signal, Swarm says HOLD - use RL but reduce size
            combined_conf = rl_confidence * 0.8  # RL confidence reduced
            final_action = rl_action * 0.7  # Reduce position by 30%
            
        else:  # disagree
            # Conflict - use conservative approach
            # If contrarian signal is strong, it may override
            if contrarian_confidence >= 0.7 and market_condition != "NEUTRAL":
                # Strong contrarian signal - trust it more in conflict
                combined_conf = min(1.0, swarm_confidence + contrarian_confidence * 0.2)
                final_action = swarm_action * 0.8  # Trust contrarian more
            else:
                combined_conf = (
                    adjusted_rl_weight * rl_confidence +
                    adjusted_swarm_weight * swarm_confidence
                ) * self.conflict_reduction_factor
                # Use smaller of the two actions
                if abs(rl_action) < abs(swarm_action):
                    final_action = rl_action * 0.5
                else:
                    final_action = swarm_action * 0.5
        
        # Check minimum confidence threshold
        if combined_conf < self.min_combined_confidence:
            final_action = 0.0
        
        return DecisionResult(
            action=final_action,
            confidence=combined_conf,
            rl_confidence=rl_confidence,
            reasoning_confidence=0.0,  # Legacy field
            swarm_confidence=swarm_confidence,
            agreement=agreement,
            reasoning=swarm_reasoning[:500] if swarm_reasoning else None,
            swarm_recommendation=swarm_recommendation
        )
    
    def _make_decision_with_reasoning(
        self,
        rl_action: float,
        rl_confidence: float,
        reasoning_analysis: ReasoningAnalysis
    ) -> DecisionResult:
        """Legacy method: Make decision using RL + Reasoning (no swarm)."""
        
        # Extract reasoning confidence
        reasoning_conf = reasoning_analysis.confidence
        
        # Determine agreement
        rl_direction = np.sign(rl_action)
        reasoning_direction = 1.0 if reasoning_analysis.recommendation == RecommendationType.APPROVE else \
                             -1.0 if reasoning_analysis.recommendation == RecommendationType.REJECT else 0.0
        
        if reasoning_direction == 0.0:  # MODIFY recommendation
            agreement = "modify"
        elif rl_direction * reasoning_direction > 0:
            agreement = "agree"
        else:
            agreement = "disagree"
        
        # Calculate combined confidence
        if agreement == "agree":
            # Both agree - boost confidence
            combined_conf = min(1.0, 
                self.rl_weight * rl_confidence + 
                self.reasoning_weight * reasoning_conf * 1.1  # 10% boost
            )
            final_action = rl_action
            
        elif agreement == "modify":
            # Reasoning suggests modification - reduce position size
            combined_conf = (
                self.rl_weight * rl_confidence + 
                self.reasoning_weight * reasoning_conf * 0.8
            )
            final_action = rl_action * 0.75  # Reduce by 25%
            
        else:  # disagree
            # Conflict - use conservative approach
            combined_conf = (
                self.rl_weight * rl_confidence + 
                self.reasoning_weight * reasoning_conf
            ) * self.conflict_reduction_factor
            final_action = rl_action * 0.5  # Reduce by 50%
        
        # Check minimum confidence threshold
        if combined_conf < self.min_combined_confidence:
            final_action = 0.0
        
        return DecisionResult(
            action=final_action,
            confidence=combined_conf,
            rl_confidence=rl_confidence,
            reasoning_confidence=reasoning_conf,
            swarm_confidence=0.0,
            agreement=agreement,
            reasoning=reasoning_analysis.reasoning_chain[:500] if reasoning_analysis.reasoning_chain else None,
            swarm_recommendation=None
        )
    
    def should_execute(self, decision: DecisionResult) -> bool:
        """
        Determine if decision should be executed.
        
        Args:
            decision: Decision result
        
        Returns:
            True if should execute
        """
        # Check confidence threshold
        if decision.confidence < self.min_combined_confidence:
            return False
        
        # Check if action is significant
        if abs(decision.action) < 0.01:
            return False
        
        return True


# Example usage
if __name__ == "__main__":
    from src.reasoning_engine import ReasoningAnalysis, RecommendationType
    
    # Test decision gate
    config = {
        "rl_weight": 0.6,
        "swarm_weight": 0.4,
        "min_combined_confidence": 0.7,
        "conflict_reduction_factor": 0.5
    }
    
    gate = DecisionGate(config)
    
    # Test agreement case
    rl_action = 0.8
    rl_conf = 0.85
    reasoning = ReasoningAnalysis(
        recommendation=RecommendationType.APPROVE,
        confidence=0.9,
        reasoning_chain="Good trade opportunity",
        risk_assessment={"level": "LOW"},
        market_alignment="aligned"
    )
    
    decision = gate.make_decision(rl_action, rl_conf, reasoning)
    print(f"Agreement case: {decision}")
    print(f"Should execute: {gate.should_execute(decision)}")
    
    # Test conflict case
    reasoning.recommendation = RecommendationType.REJECT
    decision = gate.make_decision(rl_action, rl_conf, reasoning)
    print(f"\nConflict case: {decision}")
    print(f"Should execute: {gate.should_execute(decision)}")

