"""
Decision Gate Module

Combines RL agent recommendations with reasoning engine analysis
and swarm agent recommendations to make final trading decisions.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from src.reasoning_engine import ReasoningAnalysis, RecommendationType
from src.quality_scorer import QualityScorer, QualityScore
from src.time_of_day_filter import TimeOfDayFilter


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
    scale_factor: float = 1.0  # Position scale applied from confluences
    confluence_count: int = 0  # Number of signals in agreement
    confluence_score: float = 0.0  # Normalised confluence score (0-1)
    confluence_signals: Optional[Dict[str, bool]] = None  # Breakdown of contributing signals
    quality_score: Optional[QualityScore] = None  # Quality score (if quality scorer enabled)
    expected_value: Optional[float] = None  # Expected value of trade
    risk_reward_ratio: Optional[float] = None  # Risk/reward ratio


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
        
        # Confluence requirement (configurable, default: 2)
        self.min_confluence_required = config.get("min_confluence_required", 2)
        
        # Quality scorer configuration
        quality_scorer_config = config.get("quality_scorer", {})
        self.quality_scorer_enabled = quality_scorer_config.get("enabled", True)
        self.quality_scorer = QualityScorer(quality_scorer_config) if self.quality_scorer_enabled else None
        
        # Swarm configuration
        self.swarm_enabled = config.get("swarm_enabled", True)
        self.swarm_timeout = config.get("swarm_timeout", 20.0)
        self.fallback_to_rl_only = config.get("fallback_to_rl_only", True)

        # Position sizing controls
        self.position_sizing_cfg = config.get("position_sizing", {})
        self.position_sizing_enabled = self.position_sizing_cfg.get("enabled", False)
        self.scale_multipliers = {
            int(k): float(v)
            for k, v in self.position_sizing_cfg.get("scale_multipliers", {
                "1": 1.0,
                "2": 1.25,
                "3": 1.5
            }).items()
        }
        if not self.scale_multipliers:
            self.scale_multipliers = {1: 1.0}
        self.max_scale = float(self.position_sizing_cfg.get("max_scale", 2.0))
        self.min_scale = float(self.position_sizing_cfg.get("min_scale", 0.3))
        self.rl_only_scale = float(self.position_sizing_cfg.get("rl_only_scale", 0.7))
        self.swarm_only_scale = float(self.position_sizing_cfg.get("swarm_only_scale", 0.8))
        self.hold_scale = float(self.position_sizing_cfg.get("hold_scale", 0.3))
        self.disagreement_scale = float(self.position_sizing_cfg.get("disagreement_scale", 0.5))
        self.contrarian_threshold = float(self.position_sizing_cfg.get("contrarian_threshold", 0.6))
        self.elliott_threshold = float(self.position_sizing_cfg.get("elliott_threshold", 0.6))
        
        # NEW: Read adaptive parameters if available (for quality filters)
        self.adaptive_config_path = Path("logs/adaptive_training/current_reward_config.json")
        self.last_adaptive_config_read = 0
        self.adaptive_config_read_interval = 1000  # Read every 1000 calls (to avoid too frequent file I/O)
        self.adaptive_call_count = 0
        self._load_adaptive_parameters()
        
        # Phase 3.4: Time-of-day filter for entry timing
        time_filter_config = config.get("time_of_day_filter", {})
        self.time_filter = TimeOfDayFilter(time_filter_config) if time_filter_config.get("enabled", True) else None
        if self.time_filter and self.time_filter.enabled:
            print(f"[TIME FILTER] {self.time_filter.get_avoid_periods_description()}")
    
    def make_decision(
        self,
        rl_action: float,
        rl_confidence: float,
        reasoning_analysis: Optional[ReasoningAnalysis] = None,
        swarm_recommendation: Optional[Dict] = None,
        current_timestamp: Optional[datetime] = None
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
        # Phase 3.4: Apply time-of-day filter before making decision
        if self.time_filter and current_timestamp:
            filtered_action, filtered_confidence, filter_reason = self.time_filter.filter_decision(
                current_timestamp, rl_action, rl_confidence
            )
            if filter_reason == "rejected_time_of_day":
                # Trade rejected due to time of day
                return DecisionResult(
                    action=0.0,
                    confidence=0.0,
                    rl_confidence=rl_confidence,
                    reasoning_confidence=0.0,
                    swarm_confidence=0.0,
                    agreement="no_swarm",
                    reasoning=f"Trade rejected: {filter_reason}",
                    swarm_recommendation=None,
                    confluence_count=0,
                    confluence_score=0.0
                )
            elif filter_reason == "reduced_confidence_time_of_day":
                # Reduce confidence but continue
                rl_confidence = filtered_confidence
        
        # If no swarm recommendation and no reasoning, use RL only
        # BUT: Still calculate quality_score and expected_value for filtering
        if swarm_recommendation is None and reasoning_analysis is None:
            # Calculate quality score and expected value even for RL-only trades
            # This ensures quality filters are applied during training
            quality_score = None
            expected_value = None
            risk_reward_ratio = None
            
            if self.quality_scorer_enabled and self.quality_scorer:
                # Calculate expected value from recent trade performance (if available)
                # For RL-only, we need to estimate from historical data or use defaults
                # Commission cost estimate (will be calculated properly if we have market data)
                commission_cost = 0.0002  # Default commission rate
                
                # Try to get expected value from quality scorer
                # This requires recent trade history, which may not be available in training
                # But we can still calculate quality score based on confidence and action magnitude
                try:
                    # Calculate basic quality score components
                    # Use RL confidence as base, and action magnitude as confidence indicator
                    quality_score = self.quality_scorer.calculate_quality_score(
                        confidence=rl_confidence,
                        confluence_count=0,  # No confluence for RL-only
                        expected_profit=0.0,  # Unknown for RL-only without history
                        commission_cost=commission_cost,
                        risk_reward_ratio=1.5,  # Default assumption
                        market_conditions={}  # Empty - no market data in RL-only mode
                    )
                except Exception:
                    # If quality scorer fails, set to None (will be checked in should_execute)
                    quality_score = None
            
            return DecisionResult(
                action=rl_action,
                confidence=rl_confidence,
                rl_confidence=rl_confidence,
                reasoning_confidence=0.0,
                swarm_confidence=0.0,
                agreement="no_swarm",
                reasoning=None,
                swarm_recommendation=None,
                confluence_count=0,  # No confluence for RL-only trades
                confluence_score=0.0,
                quality_score=quality_score,  # Include quality score for filtering
                expected_value=expected_value,  # Include expected value for filtering
                risk_reward_ratio=risk_reward_ratio
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
    
    def _calculate_confluence_details(
        self,
        rl_action: float,
        swarm_action: float,
        swarm_recommendation: Dict,
        agreement: str
    ) -> Dict[str, object]:
        """
        Evaluate alignment across agents and timeframes for position scaling.
        
        ENHANCED: Now includes timeframe alignment check.
        """
        signals = {}
        rl_direction = np.sign(rl_action) if abs(rl_action) >= 0.05 else 0.0
        swarm_direction = np.sign(swarm_action) if abs(swarm_action) >= 0.05 else 0.0

        if rl_direction != 0:
            signals["rl_signal"] = True
        else:
            signals["rl_signal"] = False

        signals["swarm_alignment"] = agreement == "agree"

        elliott_action = swarm_recommendation.get("elliott_wave_action")
        elliott_conf = swarm_recommendation.get("elliott_wave_confidence", 0.0)
        elliott_direction = 0
        if elliott_action == "BUY":
            elliott_direction = 1
        elif elliott_action == "SELL":
            elliott_direction = -1
        signals["elliott_alignment"] = (
            rl_direction != 0
            and elliott_direction == rl_direction
            and elliott_conf >= self.elliott_threshold
        )

        contrarian_signal = swarm_recommendation.get("contrarian_signal")
        contrarian_confidence = swarm_recommendation.get("contrarian_confidence", 0.0)
        contrarian_direction = 0
        if contrarian_signal == "BUY":
            contrarian_direction = 1
        elif contrarian_signal == "SELL":
            contrarian_direction = -1
        signals["contrarian_alignment"] = (
            rl_direction != 0
            and contrarian_direction == rl_direction
            and contrarian_confidence >= self.contrarian_threshold
        )

        bias = swarm_recommendation.get("bias")
        signals["bias_alignment"] = (
            rl_direction > 0 and bias == "bullish"
        ) or (
            rl_direction < 0 and bias == "bearish"
        )
        
        # ENHANCEMENT: Check timeframe alignment
        # Check if multiple timeframes (1min, 5min, 15min) are aligned
        timeframe_signals = swarm_recommendation.get("timeframe_signals", {})
        if timeframe_signals:
            # Check alignment across timeframes
            timeframe_directions = []
            if "1min" in timeframe_signals:
                tf1_direction = 1 if timeframe_signals["1min"].get("direction") == "BUY" else (-1 if timeframe_signals["1min"].get("direction") == "SELL" else 0)
                if tf1_direction != 0:
                    timeframe_directions.append(tf1_direction)
            if "5min" in timeframe_signals:
                tf5_direction = 1 if timeframe_signals["5min"].get("direction") == "BUY" else (-1 if timeframe_signals["5min"].get("direction") == "SELL" else 0)
                if tf5_direction != 0:
                    timeframe_directions.append(tf5_direction)
            if "15min" in timeframe_signals:
                tf15_direction = 1 if timeframe_signals["15min"].get("direction") == "BUY" else (-1 if timeframe_signals["15min"].get("direction") == "SELL" else 0)
                if tf15_direction != 0:
                    timeframe_directions.append(tf15_direction)
            
            # Check if all timeframes agree
            if len(timeframe_directions) >= 2:
                # At least 2 timeframes have signals
                if all(d == timeframe_directions[0] for d in timeframe_directions):
                    # All timeframes agree
                    signals["timeframe_alignment"] = True
                    if rl_direction != 0 and rl_direction == timeframe_directions[0]:
                        # RL also agrees with timeframes
                        signals["timeframe_alignment"] = True
                    else:
                        signals["timeframe_alignment"] = False
                else:
                    signals["timeframe_alignment"] = False
            else:
                signals["timeframe_alignment"] = False
        else:
            # No timeframe signals available
            signals["timeframe_alignment"] = False

        confluence_count = sum(signals.values())
        max_signals = max(1, len(signals))
        confluence_score = min(1.0, confluence_count / max_signals)

        return {
            "signals": signals,
            "count": confluence_count,
            "score": confluence_score
        }

    def _lookup_scale_multiplier(self, count: int) -> float:
        """Get scale multiplier for given confluence count."""
        if not self.scale_multipliers:
            return 1.0
        if count in self.scale_multipliers:
            return self.scale_multipliers[count]
        max_key = max(self.scale_multipliers.keys())
        return self.scale_multipliers[max_key]

    def _apply_position_sizing(
        self,
        final_action: float,
        agreement: str,
        confluence_info: Dict[str, object],
        confidence: float = 1.0,
        win_rate: float = 0.5,
        market_conditions: Optional[Dict] = None
    ) -> Tuple[float, float]:
        """
        Enhanced position sizing based on confluence, agreement, confidence, win rate, and market conditions.

        Returns:
            Tuple of (scaled_action, applied_scale_factor)
        """
        if not self.position_sizing_enabled:
            return final_action, 1.0

        market_conditions = market_conditions or {}
        
        # Base scale factor from confluence and agreement
        scale_factor = 1.0
        count = int(confluence_info.get("count", 0))

        if agreement == "agree":
            scale_factor = self._lookup_scale_multiplier(count if count > 0 else 1)
        elif agreement == "rl_only":
            scale_factor = self.rl_only_scale
        elif agreement == "swarm_only":
            scale_factor = self.swarm_only_scale
        elif agreement == "swarm_hold":
            scale_factor = self.hold_scale
        else:  # disagree or other conflict states
            scale_factor = self.disagreement_scale
        
        # ENHANCEMENT: Adjust based on confidence (higher confidence = larger size)
        # Confidence factor: 0.7 confidence = 0.85x, 1.0 confidence = 1.0x
        if confidence < 0.7:
            confidence_factor = 0.7  # Minimum factor for low confidence
        else:
            confidence_factor = 0.7 + (confidence - 0.7) * 1.0  # Scale from 0.7-1.0 to 0.7-1.0
        scale_factor *= confidence_factor
        
        # ENHANCEMENT: Adjust based on win rate (higher win rate = larger size)
        # Win rate factor: 0.5 win rate = 0.8x, 0.6 win rate = 1.0x, 0.7+ win rate = 1.1x
        if win_rate < 0.5:
            win_rate_factor = 0.8  # Reduce size if win rate is low
        elif win_rate >= 0.7:
            win_rate_factor = 1.1  # Increase size if win rate is high
        else:
            # Linear scaling from 0.5-0.7 win rate to 0.8-1.1 factor
            win_rate_factor = 0.8 + (win_rate - 0.5) * 1.5
        scale_factor *= win_rate_factor
        
        # ENHANCEMENT: Regime-aware position sizing (Phase 2)
        # Extract regime information from market conditions
        regime = market_conditions.get("regime", "unknown")
        regime_confidence = market_conditions.get("regime_confidence", 0.5)
        volatility = market_conditions.get("volatility", 0.0)
        
        # Regime-aware adjustments based on plan
        regime_factor = 1.0  # Default: no adjustment
        
        if regime == "trending" and regime_confidence > 0.7:
            # Trending market with high confidence: increase size
            # This is favorable for trend-following strategies
            regime_factor = 1.2
        elif regime == "trending" and regime_confidence <= 0.7:
            # Trending but low confidence: slight increase
            regime_factor = 1.1
        elif regime == "ranging":
            # Ranging market: reduce size (trend-following struggles in ranges)
            regime_factor = 0.7
        elif regime == "volatile":
            # Volatile market: reduce size more (high risk, choppy price action)
            regime_factor = 0.6
        else:
            # Unknown or neutral regime: no adjustment
            regime_factor = 1.0
        
        # Additional volatility-based adjustment (if regime not available)
        if regime == "unknown" and volatility > 0.01:
            # High volatility without regime info: slight reduction
            regime_factor = 0.9
        elif regime == "unknown" and volatility < 0.005:
            # Low volatility without regime info: slight reduction
            regime_factor = 0.9
        
        scale_factor *= regime_factor

        scale_factor = float(np.clip(scale_factor, self.min_scale, self.max_scale))
        scaled_action = float(np.clip(final_action * scale_factor, -self.max_scale, self.max_scale))
        return scaled_action, scale_factor

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
        elliott_confidence = swarm_recommendation.get("elliott_wave_confidence", 0.0)
        elliott_action = swarm_recommendation.get("elliott_wave_action")
        elliott_phase = swarm_recommendation.get("elliott_wave_phase")
        elliott_bias = swarm_recommendation.get("elliott_wave_bias")
        confluence_info = {"signals": {}, "count": 0, "score": 0.0}
        
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

        if elliott_confidence >= 0.6 and elliott_action in {"BUY", "SELL"}:
            elliott_boost = elliott_confidence * 0.2
            adjusted_swarm_weight = min(0.85, adjusted_swarm_weight + elliott_boost)
            adjusted_rl_weight = 1.0 - adjusted_swarm_weight
            if elliott_phase:
                additional_reason = (
                    f"\nElliott Wave ({elliott_phase}, bias {elliott_bias}) confidence "
                    f"{elliott_confidence:.2f} favors {elliott_action}"
                )
                swarm_reasoning = (
                    f"{swarm_reasoning}{additional_reason}" if swarm_reasoning else additional_reason.strip()
                )
        
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
        
        # Calculate confluence info first (needed for position sizing and quality scoring)
        confluence_info = self._calculate_confluence_details(
            rl_action=rl_action,
            swarm_action=swarm_action,
            swarm_recommendation=swarm_recommendation,
            agreement=agreement
        )
        
        # Extract market conditions and win rate for enhanced position sizing
        market_conditions = swarm_recommendation.get("market_conditions", {}) if swarm_recommendation else {}
        
        # Phase 2: Extract regime information from swarm recommendation
        # Try to get regime from multiple sources in order of preference
        if not market_conditions.get("regime") or market_conditions.get("regime") == "unknown":
            # Try to get from swarm recommendation directly
            regime = swarm_recommendation.get("regime") or swarm_recommendation.get("market_regime")
            if regime:
                market_conditions["regime"] = regime.lower() if isinstance(regime, str) else "unknown"
        
        # Extract regime confidence if available
        if "regime_confidence" not in market_conditions:
            regime_confidence = swarm_recommendation.get("regime_confidence") or swarm_recommendation.get("market_regime_confidence")
            if regime_confidence is not None:
                market_conditions["regime_confidence"] = float(regime_confidence)
        
        win_rate = swarm_recommendation.get("win_rate", 0.5) if swarm_recommendation else 0.5
        
        # Enhanced position sizing (with confidence, win rate, market conditions)
        if self.position_sizing_enabled:
            final_action, scale_factor = self._apply_position_sizing(
                final_action=final_action,
                agreement=agreement,
                confluence_info=confluence_info,
                confidence=combined_conf,
                win_rate=win_rate,
                market_conditions=market_conditions
            )
            combined_conf = min(1.0, combined_conf * (0.9 + 0.1 * scale_factor))
        else:
            scale_factor = 1.0
        
        # Check minimum confidence threshold
        if combined_conf < self.min_combined_confidence:
            final_action = 0.0
        
        # Calculate quality score and expected value if quality scorer is enabled
        quality_score = None
        expected_value = None
        risk_reward_ratio = None
        
        if self.quality_scorer_enabled and self.quality_scorer and final_action != 0.0:
            # Calculate expected value (simplified - would need win rate, avg win/loss in real implementation)
            # For now, use confidence as proxy for expected value
            commission_cost = abs(final_action) * 100000.0 * 0.0003  # Estimate commission
            expected_profit = combined_conf * 100.0  # Estimate expected profit from confidence
            expected_value = expected_profit - commission_cost
            
            # Calculate risk/reward ratio (simplified - would need stop loss/take profit in real implementation)
            # For now, use a default ratio based on confidence
            risk_reward_ratio = 1.0 + combined_conf  # Range: 1.0-2.0
            
            # ENHANCEMENT: Add timeframe alignment to market conditions for quality scoring
            if confluence_info.get("signals", {}).get("timeframe_alignment"):
                market_conditions["timeframe_alignment"] = True
            else:
                market_conditions["timeframe_alignment"] = False
            quality_score = self.quality_scorer.calculate_quality_score(
                confidence=combined_conf,
                confluence_count=int(confluence_info.get("count", 0)),
                expected_profit=expected_profit,
                commission_cost=commission_cost,
                risk_reward_ratio=risk_reward_ratio,
                market_conditions=market_conditions
            )
            
            # Reject if quality score is too low
            if not self.quality_scorer.should_trade(quality_score):
                final_action = 0.0
        
        return DecisionResult(
            action=final_action,
            confidence=combined_conf,
            rl_confidence=rl_confidence,
            reasoning_confidence=0.0,  # Legacy field
            swarm_confidence=swarm_confidence,
            agreement=agreement,
            reasoning=swarm_reasoning[:500] if swarm_reasoning else None,
            swarm_recommendation=swarm_recommendation,
            scale_factor=scale_factor,
            confluence_count=int(confluence_info.get("count", 0)),
            confluence_score=float(confluence_info.get("score", 0.0)),
            confluence_signals=confluence_info.get("signals"),
            quality_score=quality_score,
            expected_value=expected_value,
            risk_reward_ratio=risk_reward_ratio
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
    
    def _load_adaptive_parameters(self):
        """Load adaptive parameters from current_reward_config.json if available"""
        if not self.adaptive_config_path.exists():
            return
        
        try:
            with open(self.adaptive_config_path, 'r') as f:
                adaptive_config = json.load(f)
                quality_filters = adaptive_config.get("quality_filters", {})
                
                if quality_filters:
                    # Override quality scorer min_quality_score with adaptive value
                    adaptive_min_quality = quality_filters.get("min_quality_score")
                    adaptive_min_confidence = quality_filters.get("min_action_confidence")
                    
                    if adaptive_min_quality is not None and self.quality_scorer:
                        old_value = self.quality_scorer.min_quality_score
                        self.quality_scorer.min_quality_score = adaptive_min_quality
                        if self.adaptive_call_count % 10000 == 0:  # Log periodically
                            print(f"[DecisionGate] Using adaptive min_quality_score: {old_value:.3f} -> {adaptive_min_quality:.3f}")
                    
                    # Store adaptive min_action_confidence for use in make_decision
                    if adaptive_min_confidence is not None:
                        self.adaptive_min_action_confidence = adaptive_min_confidence
        except Exception as e:
            # Silently fail - use default config values
            pass
    
    def should_execute(self, decision: DecisionResult) -> bool:
        """
        Determine if decision should be executed.
        
        Args:
            decision: Decision result
        
        Returns:
            True if should execute
        """
        # Periodically reload adaptive parameters (every N calls)
        self.adaptive_call_count += 1
        if self.adaptive_call_count % self.adaptive_config_read_interval == 0:
            self._load_adaptive_parameters()
        
        # Check confidence threshold (use adaptive if available)
        min_confidence = getattr(self, 'adaptive_min_action_confidence', None)
        if min_confidence is not None:
            # Use adaptive min_action_confidence if available
            if decision.confidence < min_confidence:
                return False
        else:
            # Use default min_combined_confidence
            if decision.confidence < self.min_combined_confidence:
                return False
        
        # Check confluence requirement (NEW - prevents low-quality RL-only trades)
        if decision.confluence_count < self.min_confluence_required:
            return False  # Reject if confluence < threshold
        
        # Check quality score (if quality scorer is enabled)
        # Quality scorer now uses adaptive min_quality_score from _load_adaptive_parameters
        # CRITICAL: If quality scorer is enabled but quality_score is None, reject the trade
        # This ensures RL-only trades must calculate quality score
        if self.quality_scorer_enabled:
            if decision.quality_score is None:
                return False  # Reject if quality scorer enabled but no score calculated
            if not self.quality_scorer.should_trade(decision.quality_score):
                return False  # Reject if quality score is too low
        
        # Check expected value (if available)
        if decision.expected_value is not None and decision.expected_value <= 0:
            return False  # Reject if expected value is negative or zero
        
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

