"""
Comprehensive Transaction Cost Model

Models all transaction costs:
- Commission (fixed per trade)
- Slippage (execution quality)
- Spread (bid-ask spread)
- Market impact (price movement)
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

# Import execution quality modules
try:
    from src.slippage_model import SlippageModel
    from src.market_impact import MarketImpactModel
    EXECUTION_QUALITY_AVAILABLE = True
except ImportError:
    EXECUTION_QUALITY_AVAILABLE = False
    SlippageModel = None
    MarketImpactModel = None


@dataclass
class TransactionCostBreakdown:
    """Breakdown of transaction costs"""
    commission: float
    spread: float
    slippage: float
    market_impact: float
    total_cost: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "commission": self.commission,
            "spread": self.spread,
            "slippage": self.slippage,
            "market_impact": self.market_impact,
            "total_cost": self.total_cost
        }


class TransactionCostModel:
    """
    Comprehensive transaction cost model:
    - Commission (fixed per trade)
    - Slippage (execution quality)
    - Spread (bid-ask spread)
    - Market impact (price movement)
    """
    
    def __init__(
        self,
        commission_rate: float = 0.0003,  # 0.03% commission
        spread_bps: float = 0.5,  # 0.5 bps average spread
        slippage_config: Optional[Dict] = None,
        market_impact_config: Optional[Dict] = None
    ):
        """
        Initialize transaction cost model.
        
        Args:
            commission_rate: Commission rate (0.0003 = 0.03%)
            spread_bps: Average bid-ask spread in basis points
            slippage_config: Configuration for slippage model
            market_impact_config: Configuration for market impact model
        """
        self.commission_rate = commission_rate
        self.spread_bps = spread_bps
        
        # Initialize slippage model if available
        if EXECUTION_QUALITY_AVAILABLE and SlippageModel:
            slippage_config = slippage_config or {}
            self.slippage_model = SlippageModel(slippage_config)
        else:
            self.slippage_model = None
        
        # Initialize market impact model if available
        if EXECUTION_QUALITY_AVAILABLE and MarketImpactModel:
            market_impact_config = market_impact_config or {}
            self.market_impact_model = MarketImpactModel(market_impact_config)
        else:
            self.market_impact_model = None
    
    def calculate_total_cost(
        self,
        order_size: float,
        current_price: float,
        bid_price: Optional[float] = None,
        ask_price: Optional[float] = None,
        volatility: Optional[float] = None,
        volume: Optional[float] = None,
        avg_volume: Optional[float] = None,
        timestamp: Optional[object] = None
    ) -> TransactionCostBreakdown:
        """
        Calculate all transaction costs.
        
        Args:
            order_size: Position size (-1.0 to 1.0)
            current_price: Current market price
            bid_price: Current bid price (for spread calculation)
            ask_price: Current ask price (for spread calculation)
            volatility: Market volatility (for slippage/impact)
            volume: Current volume (for slippage/impact)
            avg_volume: Average volume (for slippage/impact)
            timestamp: Timestamp (for time-of-day effects)
        
        Returns:
            TransactionCostBreakdown with all costs
        """
        notional_value = abs(order_size) * current_price
        
        # 1. Commission (fixed per trade)
        commission = notional_value * self.commission_rate
        
        # 2. Spread (half spread for market order)
        if bid_price is not None and ask_price is not None:
            spread = (ask_price - bid_price) / 2.0
            spread_cost = abs(order_size) * spread
        else:
            # Use default spread in basis points
            spread = current_price * (self.spread_bps / 10000)
            spread_cost = notional_value * (self.spread_bps / 10000)
        
        # 3. Slippage
        slippage_cost = 0.0
        if self.slippage_model and volatility is not None and volume is not None:
            is_buy = order_size > 0
            slippage = self.slippage_model.calculate_slippage(
                order_size=abs(order_size),
                current_price=current_price,
                volatility=volatility,
                volume=volume,
                avg_volume=avg_volume or volume,
                timestamp=timestamp
            )
            slippage_cost = notional_value * slippage
        else:
            # Default slippage estimate (1.5 bps)
            slippage_cost = notional_value * 0.00015
        
        # 4. Market impact
        market_impact_cost = 0.0
        if self.market_impact_model and volatility is not None and avg_volume is not None:
            is_buy = order_size > 0
            impact = self.market_impact_model.calculate_price_impact(
                order_size=abs(order_size),
                current_price=current_price,
                avg_volume=avg_volume,
                volatility=volatility
            )
            market_impact_cost = notional_value * impact
        else:
            # Default market impact estimate (0.5 bps for small orders)
            # Larger impact for larger orders
            volume_ratio = abs(order_size) / (avg_volume or 1.0)
            impact_bps = 0.5 * np.sqrt(volume_ratio)
            market_impact_cost = notional_value * (impact_bps / 10000)
        
        total_cost = commission + spread_cost + slippage_cost + market_impact_cost
        
        return TransactionCostBreakdown(
            commission=commission,
            spread=spread_cost,
            slippage=slippage_cost,
            market_impact=market_impact_cost,
            total_cost=total_cost
        )
    
    def calculate_cost_per_trade(
        self,
        order_size: float,
        current_price: float,
        **kwargs
    ) -> float:
        """
        Calculate total cost per trade (simplified interface).
        
        Returns:
            Total cost in dollars
        """
        breakdown = self.calculate_total_cost(
            order_size=order_size,
            current_price=current_price,
            **kwargs
        )
        return breakdown.total_cost
    
    def calculate_cost_bps(
        self,
        order_size: float,
        current_price: float,
        **kwargs
    ) -> float:
        """
        Calculate total cost in basis points.
        
        Returns:
            Total cost in basis points
        """
        notional_value = abs(order_size) * current_price
        if notional_value == 0:
            return 0.0
        
        breakdown = self.calculate_total_cost(
            order_size=order_size,
            current_price=current_price,
            **kwargs
        )
        
        cost_bps = (breakdown.total_cost / notional_value) * 10000
        return cost_bps
    
    def estimate_round_trip_cost(
        self,
        order_size: float,
        current_price: float,
        **kwargs
    ) -> float:
        """
        Estimate round-trip cost (entry + exit).
        
        Returns:
            Total round-trip cost in dollars
        """
        entry_cost = self.calculate_cost_per_trade(
            order_size=order_size,
            current_price=current_price,
            **kwargs
        )
        
        # Exit cost (same order size, opposite direction)
        exit_cost = self.calculate_cost_per_trade(
            order_size=-order_size,
            current_price=current_price,
            **kwargs
        )
        
        return entry_cost + exit_cost
    
    def get_cost_breakdown_summary(
        self,
        order_size: float,
        current_price: float,
        **kwargs
    ) -> Dict:
        """
        Get detailed cost breakdown summary.
        
        Returns:
            Dictionary with cost breakdown and percentages
        """
        breakdown = self.calculate_total_cost(
            order_size=order_size,
            current_price=current_price,
            **kwargs
        )
        
        notional_value = abs(order_size) * current_price
        
        if breakdown.total_cost > 0:
            commission_pct = (breakdown.commission / breakdown.total_cost) * 100
            spread_pct = (breakdown.spread / breakdown.total_cost) * 100
            slippage_pct = (breakdown.slippage / breakdown.total_cost) * 100
            impact_pct = (breakdown.market_impact / breakdown.total_cost) * 100
        else:
            commission_pct = spread_pct = slippage_pct = impact_pct = 0.0
        
        cost_bps = (breakdown.total_cost / notional_value) * 10000 if notional_value > 0 else 0.0
        
        return {
            "notional_value": notional_value,
            "cost_breakdown": breakdown.to_dict(),
            "cost_percentages": {
                "commission": commission_pct,
                "spread": spread_pct,
                "slippage": slippage_pct,
                "market_impact": impact_pct
            },
            "total_cost_bps": cost_bps,
            "round_trip_cost": self.estimate_round_trip_cost(
                order_size=order_size,
                current_price=current_price,
                **kwargs
            )
        }

