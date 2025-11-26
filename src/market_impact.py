"""
Market Impact Model

Models price impact from order execution using the square root model (Almgren-Chriss).
"""

import numpy as np
from typing import Optional, Dict


class MarketImpactModel:
    """
    Models price impact from order execution.
    
    Uses square root model: impact = alpha * sqrt(order_size / avg_volume) * volatility
    
    This is a simplified model. More sophisticated models can include:
    - Permanent vs. temporary impact
    - Order book depth
    - Time-weighted execution
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize market impact model.
        
        Args:
            config: Configuration dict with impact parameters
        """
        config = config or {}
        
        # Impact coefficient (alpha in square root model)
        # Typical values: 0.1-0.5 for futures
        self.impact_coefficient = config.get("impact_coefficient", 0.3)
        
        # Volatility multiplier
        self.vol_multiplier = config.get("vol_multiplier", 1.0)
        
        # Enable/disable market impact
        self.enabled = config.get("enabled", True)
    
    def calculate_price_impact(
        self,
        order_size: float,
        current_price: float,
        avg_volume: float,
        volatility: Optional[float] = None,
        liquidity: Optional[float] = None
    ) -> float:
        """
        Calculate expected price impact in basis points (as fraction).
        
        Args:
            order_size: Order size (absolute value, normalized 0-1)
            current_price: Current market price
            avg_volume: Average volume (for market impact calculation)
            volatility: Current volatility (optional)
            liquidity: Liquidity measure (optional, defaults to avg_volume)
        
        Returns:
            Price impact as fraction (e.g., 0.0002 = 2 bps)
        """
        if not self.enabled:
            return 0.0
        
        if abs(order_size) < 1e-6 or avg_volume <= 0:
            return 0.0
        
        # Square root model (Almgren-Chriss)
        # Impact = alpha * sqrt(order_size / avg_volume) * volatility_factor
        
        # Calculate volume ratio
        # Convert normalized order size to volume estimate
        # For ES futures, 1.0 position = ~$50k notional
        # Estimate: order_size represents fraction of typical volume
        volume_ratio = abs(order_size) / max(avg_volume, 1.0)
        
        # Square root of volume ratio
        sqrt_volume_ratio = np.sqrt(volume_ratio)
        
        # Base impact
        impact = self.impact_coefficient * sqrt_volume_ratio * 0.0001  # Convert to basis points
        
        # Volatility adjustment
        if volatility is not None and volatility > 0:
            impact *= (1.0 + volatility * self.vol_multiplier)
        
        # Liquidity adjustment (if provided)
        if liquidity is not None and liquidity > 0:
            # Lower liquidity = higher impact
            liquidity_factor = 1.0 / max(liquidity / avg_volume, 0.1)
            impact *= liquidity_factor
        
        # Ensure impact is reasonable (max 1% = 0.01)
        impact = min(max(impact, 0.0), 0.01)
        
        return impact
    
    def apply_market_impact(
        self,
        intended_price: float,
        order_size: float,
        is_buy: bool,
        avg_volume: float,
        volatility: Optional[float] = None,
        liquidity: Optional[float] = None
    ) -> float:
        """
        Apply market impact to intended execution price.
        
        Args:
            intended_price: Intended execution price
            order_size: Order size (absolute value, normalized 0-1)
            is_buy: True for buy orders, False for sell orders
            avg_volume: Average volume
            volatility: Current volatility (optional)
            liquidity: Liquidity measure (optional)
        
        Returns:
            Price with market impact applied
        """
        impact = self.calculate_price_impact(
            order_size=order_size,
            current_price=intended_price,
            avg_volume=avg_volume,
            volatility=volatility,
            liquidity=liquidity
        )
        
        # Buy orders: push price up (positive impact)
        # Sell orders: push price down (negative impact)
        if is_buy:
            actual_price = intended_price * (1.0 + impact)
        else:
            actual_price = intended_price * (1.0 - impact)
        
        return actual_price

