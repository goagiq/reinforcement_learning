"""
Integration Tests for DecisionGate with Swarm

Tests RL + Swarm fusion logic and fallback mechanisms.
"""

import pytest
import numpy as np
from src.decision_gate import DecisionGate, DecisionResult


class TestDecisionGateSwarmIntegration:
    """Test DecisionGate with swarm integration."""
    
    @pytest.fixture
    def decision_gate(self):
        """Create decision gate."""
        return DecisionGate({
            "rl_weight": 0.6,
            "swarm_weight": 0.4,
            "min_combined_confidence": 0.7,
            "conflict_reduction_factor": 0.5,
            "swarm_enabled": True,
            "swarm_timeout": 20.0
        })
    
    def test_agree_scenario(self, decision_gate):
        """Test when RL and Swarm agree."""
        rl_action = 0.8
        rl_confidence = 0.85
        
        swarm_recommendation = {
            "action": "BUY",
            "position_size": 0.7,
            "confidence": 0.9,
            "reasoning": "Strong correlation and positive sentiment"
        }
        
        result = decision_gate.make_decision(
            rl_action=rl_action,
            rl_confidence=rl_confidence,
            swarm_recommendation=swarm_recommendation
        )
        
        assert result.agreement == "agree"
        assert result.confidence > 0.85  # Boosted confidence
        assert abs(result.action) > 0.7  # Strong action
    
    def test_disagree_scenario(self, decision_gate):
        """Test when RL and Swarm disagree."""
        rl_action = 0.8  # RL says BUY
        rl_confidence = 0.85
        
        swarm_recommendation = {
            "action": "SELL",
            "position_size": 0.6,
            "confidence": 0.8,
            "reasoning": "Divergence detected"
        }
        
        result = decision_gate.make_decision(
            rl_action=rl_action,
            rl_confidence=rl_confidence,
            swarm_recommendation=swarm_recommendation
        )
        
        assert result.agreement == "disagree"
        assert result.confidence < 0.7  # Reduced confidence
        assert abs(result.action) < 0.5  # Conservative action
    
    def test_swarm_hold_scenario(self, decision_gate):
        """Test when Swarm says HOLD."""
        rl_action = 0.8  # RL says BUY
        rl_confidence = 0.85
        
        swarm_recommendation = {
            "action": "HOLD",
            "position_size": 0.0,
            "confidence": 0.7,
            "reasoning": "Uncertain market conditions"
        }
        
        result = decision_gate.make_decision(
            rl_action=rl_action,
            rl_confidence=rl_confidence,
            swarm_recommendation=swarm_recommendation
        )
        
        assert result.agreement == "swarm_hold"
        assert result.confidence < 0.7  # Reduced confidence
        assert abs(result.action) < 0.3  # Significantly reduced
    
    def test_no_swarm_fallback(self, decision_gate):
        """Test fallback to RL-only when swarm unavailable."""
        rl_action = 0.8
        rl_confidence = 0.85
        
        result = decision_gate.make_decision(
            rl_action=rl_action,
            rl_confidence=rl_confidence,
            swarm_recommendation=None
        )
        
        assert result.agreement == "no_swarm"
        assert result.action == rl_action
        assert result.confidence == rl_confidence
        assert result.swarm_confidence == 0.0
    
    def test_confidence_threshold(self, decision_gate):
        """Test minimum confidence threshold."""
        rl_action = 0.5
        rl_confidence = 0.5  # Low confidence
        
        swarm_recommendation = {
            "action": "BUY",
            "position_size": 0.4,
            "confidence": 0.5,  # Low confidence
            "reasoning": "Weak signals"
        }
        
        result = decision_gate.make_decision(
            rl_action=rl_action,
            rl_confidence=rl_confidence,
            swarm_recommendation=swarm_recommendation
        )
        
        # If combined confidence < threshold, action should be 0
        if result.confidence < decision_gate.min_combined_confidence:
            assert result.action == 0.0
    
    def test_should_execute(self, decision_gate):
        """Test execution decision."""
        # High confidence decision
        high_conf_decision = DecisionResult(
            action=0.8,
            confidence=0.85,
            rl_confidence=0.85,
            reasoning_confidence=0.0,
            swarm_confidence=0.9,
            agreement="agree"
        )
        
        assert decision_gate.should_execute(high_conf_decision) is True
        
        # Low confidence decision
        low_conf_decision = DecisionResult(
            action=0.5,
            confidence=0.6,  # Below threshold
            rl_confidence=0.6,
            reasoning_confidence=0.0,
            swarm_confidence=0.6,
            agreement="disagree"
        )
        
        assert decision_gate.should_execute(low_conf_decision) is False
        
        # Tiny action
        tiny_action_decision = DecisionResult(
            action=0.005,  # Too small
            confidence=0.8,
            rl_confidence=0.8,
            reasoning_confidence=0.0,
            swarm_confidence=0.8,
            agreement="agree"
        )
        
        assert decision_gate.should_execute(tiny_action_decision) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

