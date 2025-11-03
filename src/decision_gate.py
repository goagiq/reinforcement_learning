"""
Decision Gate Module

Combines RL agent recommendations with reasoning engine analysis
to make final trading decisions.
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
    agreement: str  # "agree", "disagree", "uncertain"
    reasoning: Optional[str] = None  # Reasoning explanation


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
        self.reasoning_weight = config.get("reasoning_weight", 0.4)
        self.rl_weight = 1.0 - self.reasoning_weight
        self.min_combined_confidence = config.get("min_combined_confidence", 0.7)
        self.conflict_reduction_factor = config.get("conflict_reduction_factor", 0.5)
    
    def make_decision(
        self,
        rl_action: float,
        rl_confidence: float,
        reasoning_analysis: Optional[ReasoningAnalysis] = None
    ) -> DecisionResult:
        """
        Make final trading decision by combining RL and reasoning.
        
        Args:
            rl_action: Action from RL agent (-1.0 to 1.0)
            rl_confidence: Confidence from RL agent (0.0 to 1.0)
            reasoning_analysis: Analysis from reasoning engine (optional)
        
        Returns:
            DecisionResult with final action and confidence
        """
        # If no reasoning, use RL only
        if reasoning_analysis is None:
            return DecisionResult(
                action=rl_action,
                confidence=rl_confidence,
                rl_confidence=rl_confidence,
                reasoning_confidence=0.0,
                agreement="no_reasoning"
            )
        
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
            # Below threshold - don't trade
            final_action = 0.0
            combined_conf = combined_conf  # Keep low confidence for logging
        
        return DecisionResult(
            action=final_action,
            confidence=combined_conf,
            rl_confidence=rl_confidence,
            reasoning_confidence=reasoning_conf,
            agreement=agreement,
            reasoning=reasoning_analysis.reasoning_chain[:500] if reasoning_analysis.reasoning_chain else None
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
        "reasoning_weight": 0.4,
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

