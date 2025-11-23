"""
Quality Scorer Module

Calculates trade quality scores based on multiple factors:
- Confidence level
- Confluence count
- Expected profit vs. commission
- Risk/reward ratio
- Market conditions (volatility, trend, regime)
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class QualityScore:
    """Quality score result"""
    score: float  # Overall quality score (0-1)
    confidence_score: float  # Confidence component (0-1)
    confluence_score: float  # Confluence component (0-1)
    expected_profit_score: float  # Expected profit component (0-1)
    risk_reward_score: float  # Risk/reward component (0-1)
    market_conditions_score: float  # Market conditions component (0-1)
    factors: Dict[str, float]  # Detailed breakdown of factors


class QualityScorer:
    """
    Calculates trade quality scores to filter low-quality trades.
    
    Quality score combines:
    - Confidence level (0-0.3)
    - Confluence count (0-0.2)
    - Expected profit vs. commission (0-0.2)
    - Risk/reward ratio (0-0.15)
    - Market conditions (0-0.15)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize quality scorer.
        
        Args:
            config: Configuration dictionary with quality score settings
        """
        self.config = config or {}
        
        # Weight configuration (must sum to 1.0)
        self.weights = {
            "confidence": self.config.get("confidence_weight", 0.30),
            "confluence": self.config.get("confluence_weight", 0.20),
            "expected_profit": self.config.get("expected_profit_weight", 0.20),
            "risk_reward": self.config.get("risk_reward_weight", 0.15),
            "market_conditions": self.config.get("market_conditions_weight", 0.15)
        }
        
        # Thresholds
        self.min_quality_score = self.config.get("min_quality_score", 0.6)  # Minimum score to trade
        self.min_risk_reward_ratio = self.config.get("min_risk_reward_ratio", 1.5)  # Minimum risk/reward (1:1.5)
        self.min_profit_margin = self.config.get("min_profit_margin", 1.5)  # Expected profit must be >= commission * 1.5
        
        # Market condition thresholds
        self.min_volatility = self.config.get("min_volatility", 0.01)  # Minimum volatility to trade
        self.min_volume_ratio = self.config.get("min_volume_ratio", 1.2)  # Volume must be >= avg * 1.2
    
    def calculate_quality_score(
        self,
        confidence: float,
        confluence_count: int,
        expected_profit: float,
        commission_cost: float,
        risk_reward_ratio: float,
        market_conditions: Optional[Dict] = None
    ) -> QualityScore:
        """
        Calculate overall quality score for a trade.
        
        Args:
            confidence: Confidence level (0-1)
            confluence_count: Number of confluent signals
            expected_profit: Expected profit from trade
            commission_cost: Commission cost for trade
            risk_reward_ratio: Risk/reward ratio (e.g., 2.0 = 1:2)
            market_conditions: Market conditions dict (volatility, trend, regime, volume_ratio)
        
        Returns:
            QualityScore object with overall score and component scores
        """
        market_conditions = market_conditions or {}
        
        # Calculate component scores
        confidence_score = self._score_confidence(confidence)
        confluence_score = self._score_confluence(confluence_count)
        expected_profit_score = self._score_expected_profit(expected_profit, commission_cost)
        risk_reward_score = self._score_risk_reward(risk_reward_ratio)
        market_conditions_score = self._score_market_conditions(market_conditions)
        
        # Calculate weighted overall score
        overall_score = (
            self.weights["confidence"] * confidence_score +
            self.weights["confluence"] * confluence_score +
            self.weights["expected_profit"] * expected_profit_score +
            self.weights["risk_reward"] * risk_reward_score +
            self.weights["market_conditions"] * market_conditions_score
        )
        
        # Clamp to [0, 1]
        overall_score = max(0.0, min(1.0, overall_score))
        
        return QualityScore(
            score=overall_score,
            confidence_score=confidence_score,
            confluence_score=confluence_score,
            expected_profit_score=expected_profit_score,
            risk_reward_score=risk_reward_score,
            market_conditions_score=market_conditions_score,
            factors={
                "confidence": confidence,
                "confluence_count": confluence_count,
                "expected_profit": expected_profit,
                "commission_cost": commission_cost,
                "risk_reward_ratio": risk_reward_ratio,
                **market_conditions
            }
        )
    
    def _score_confidence(self, confidence: float) -> float:
        """Score confidence component (0-1)"""
        # Linear scaling: 0.7 confidence = 0.7 score, 1.0 confidence = 1.0 score
        # Minimum confidence threshold: 0.7
        if confidence < 0.7:
            return 0.0
        return (confidence - 0.7) / 0.3  # Scale from 0.7-1.0 to 0-1.0
    
    def _score_confluence(self, confluence_count: int) -> float:
        """Score confluence component (0-1)"""
        # More confluence = better score
        # 2 confluence = 0.5, 3 = 0.75, 4+ = 1.0
        if confluence_count < 2:
            return 0.0
        if confluence_count >= 4:
            return 1.0
        return (confluence_count - 2) / 2.0  # Scale from 2-4 to 0-1.0
    
    def _score_expected_profit(self, expected_profit: float, commission_cost: float) -> float:
        """Score expected profit component (0-1)"""
        if commission_cost <= 0:
            return 1.0 if expected_profit > 0 else 0.0
        
        # Expected profit must be >= commission * min_profit_margin
        min_profit = commission_cost * self.min_profit_margin
        if expected_profit < min_profit:
            return 0.0
        
        # Score based on profit margin
        profit_margin = expected_profit / commission_cost
        if profit_margin >= 3.0:  # 3x commission = perfect score
            return 1.0
        # Scale from min_profit_margin to 3.0
        return (profit_margin - self.min_profit_margin) / (3.0 - self.min_profit_margin)
    
    def _score_risk_reward(self, risk_reward_ratio: float) -> float:
        """Score risk/reward component (0-1)"""
        # Target: 1:2 ratio (risk_reward_ratio = 2.0)
        # Minimum: 1:1.5 ratio (risk_reward_ratio = 1.5)
        if risk_reward_ratio < self.min_risk_reward_ratio:
            return 0.0
        
        if risk_reward_ratio >= 2.0:  # 1:2 or better = perfect score
            return 1.0
        
        # Scale from min_risk_reward_ratio to 2.0
        return (risk_reward_ratio - self.min_risk_reward_ratio) / (2.0 - self.min_risk_reward_ratio)
    
    def _score_market_conditions(self, market_conditions: Dict) -> float:
        """Score market conditions component (0-1)"""
        score = 0.0
        factors = 0
        
        # Volatility check
        volatility = market_conditions.get("volatility", 0.0)
        if volatility >= self.min_volatility:
            score += 0.3
        factors += 1
        
        # Volume check
        volume_ratio = market_conditions.get("volume_ratio", 1.0)
        if volume_ratio >= self.min_volume_ratio:
            score += 0.25
        factors += 1
        
        # Market regime check
        regime = market_conditions.get("regime", "unknown")
        if regime == "trending":
            score += 0.25  # Trending markets are favorable
        elif regime == "ranging":
            score += 0.1  # Ranging markets are less favorable
        factors += 1
        
        # ENHANCEMENT: Timeframe alignment check
        timeframe_alignment = market_conditions.get("timeframe_alignment", False)
        if timeframe_alignment:
            score += 0.2  # Bonus for timeframe alignment
        factors += 1
        
        # Normalize by number of factors checked (max score = 1.0)
        return min(1.0, score)
    
    def should_trade(self, quality_score: QualityScore) -> bool:
        """
        Determine if trade should be executed based on quality score.
        
        Args:
            quality_score: QualityScore object
        
        Returns:
            True if trade should be executed
        """
        return quality_score.score >= self.min_quality_score
    
    def calculate_expected_value(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        commission_cost: float,
        confidence: float = 1.0
    ) -> float:
        """
        Calculate expected value of a trade.
        
        Args:
            win_rate: Win rate (0-1)
            avg_win: Average profit per winning trade
            avg_loss: Average loss per losing trade (positive value)
            commission_cost: Commission cost for trade
            confidence: Confidence level (0-1) to adjust expected value
        
        Returns:
            Expected value (positive = profitable, negative = unprofitable)
        """
        # Expected profit = win_rate * avg_win * confidence
        expected_profit = win_rate * avg_win * confidence
        
        # Expected loss = (1 - win_rate) * avg_loss * confidence
        expected_loss = (1 - win_rate) * avg_loss * confidence
        
        # Expected value = expected_profit - expected_loss - commission_cost
        expected_value = expected_profit - expected_loss - commission_cost
        
        return expected_value
    
    def calculate_breakeven_win_rate(
        self,
        avg_win: float,
        avg_loss: float,
        commission_cost: float = 0.0
    ) -> float:
        """
        Calculate breakeven win rate.
        
        Args:
            avg_win: Average profit per winning trade
            avg_loss: Average loss per losing trade (positive value)
            commission_cost: Commission cost for trade
        
        Returns:
            Breakeven win rate (0-1)
        """
        if avg_win <= 0 or avg_loss <= 0:
            return 0.5  # Default to 50% if invalid inputs
        
        # Breakeven: win_rate * avg_win - (1 - win_rate) * avg_loss - commission_cost = 0
        # Solve for win_rate:
        # win_rate * avg_win - avg_loss + win_rate * avg_loss - commission_cost = 0
        # win_rate * (avg_win + avg_loss) = avg_loss + commission_cost
        # win_rate = (avg_loss + commission_cost) / (avg_win + avg_loss)
        
        breakeven_win_rate = (avg_loss + commission_cost) / (avg_win + avg_loss)
        return min(1.0, max(0.0, breakeven_win_rate))
    
    def calculate_risk_reward_ratio(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_direction: int  # 1 for long, -1 for short
    ) -> float:
        """
        Calculate risk/reward ratio.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            position_direction: 1 for long, -1 for short
        
        Returns:
            Risk/reward ratio (e.g., 2.0 = 1:2, meaning reward is 2x risk)
        """
        if position_direction > 0:  # Long
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
        else:  # Short
            risk = abs(stop_loss - entry_price)
            reward = abs(entry_price - take_profit)
        
        if risk <= 0:
            return 0.0
        
        return reward / risk


# Example usage
if __name__ == "__main__":
    # Test quality scorer
    scorer = QualityScorer()
    
    # Test case 1: High-quality trade
    quality_score = scorer.calculate_quality_score(
        confidence=0.85,
        confluence_count=3,
        expected_profit=100.0,
        commission_cost=30.0,
        risk_reward_ratio=2.0,
        market_conditions={
            "volatility": 0.02,
            "volume_ratio": 1.5,
            "regime": "trending"
        }
    )
    print(f"High-quality trade: {quality_score.score:.2f}")
    print(f"  Should trade: {scorer.should_trade(quality_score)}")
    
    # Test case 2: Low-quality trade
    quality_score = scorer.calculate_quality_score(
        confidence=0.6,
        confluence_count=1,
        expected_profit=10.0,
        commission_cost=30.0,
        risk_reward_ratio=1.0,
        market_conditions={
            "volatility": 0.005,
            "volume_ratio": 0.8,
            "regime": "ranging"
        }
    )
    print(f"\nLow-quality trade: {quality_score.score:.2f}")
    print(f"  Should trade: {scorer.should_trade(quality_score)}")
    
    # Test expected value calculation
    expected_value = scorer.calculate_expected_value(
        win_rate=0.55,
        avg_win=100.0,
        avg_loss=50.0,
        commission_cost=30.0,
        confidence=0.8
    )
    print(f"\nExpected value: {expected_value:.2f}")
    
    # Test breakeven win rate
    breakeven_win_rate = scorer.calculate_breakeven_win_rate(
        avg_win=100.0,
        avg_loss=50.0,
        commission_cost=30.0
    )
    print(f"Breakeven win rate: {breakeven_win_rate:.2%}")

