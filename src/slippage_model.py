"""
Slippage Model for Realistic Execution Simulation

Models execution slippage based on:
- Order size (market impact)
- Volatility (wider spreads in volatile markets)
- Time of day (spreads wider at open/close)
- Volume (lower volume = more slippage)
"""

import numpy as np
from typing import Optional, Dict
from datetime import datetime, time


class SlippageModel:
    """
    Models execution slippage for realistic backtesting and training.
    
    Formula: slippage = base_slippage + market_impact + volatility_adjustment + time_adjustment
    
    Market Impact: sqrt(order_size / avg_volume) * impact_coefficient
    Volatility Adjustment: volatility * vol_multiplier
    Time Adjustment: wider spreads at market open/close
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize slippage model.
        
        Args:
            config: Configuration dict with slippage parameters
        """
        config = config or {}
        
        # Base slippage (0.5-2 bps for ES futures)
        self.base_slippage = config.get("base_slippage", 0.00015)  # 1.5 bps
        
        # Market impact coefficient
        self.impact_coefficient = config.get("impact_coefficient", 0.0002)  # 2 bps per sqrt(volume_ratio)
        
        # Volatility multiplier (1 bp per 1% volatility)
        self.vol_multiplier = config.get("vol_multiplier", 0.0001)
        
        # Time of day multipliers
        self.time_multipliers = config.get("time_multipliers", {
            "market_open": 1.5,      # 50% wider at open (09:30-10:00 ET)
            "market_close": 1.5,     # 50% wider at close (15:30-16:00 ET)
            "normal_hours": 1.0,     # Normal spreads
            "after_hours": 2.0      # 100% wider after hours
        })
        
        # Market open/close times (ET)
        self.market_open_start = time(9, 30)
        self.market_open_end = time(10, 0)
        self.market_close_start = time(15, 30)
        self.market_close_end = time(16, 0)
        
        # Enable/disable slippage
        self.enabled = config.get("enabled", True)
    
    def calculate_slippage(
        self,
        order_size: float,
        current_price: float,
        volatility: Optional[float] = None,
        volume: Optional[float] = None,
        avg_volume: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> float:
        """
        Calculate slippage in basis points (as fraction, e.g., 0.00015 = 1.5 bps).
        
        Args:
            order_size: Order size (absolute value, normalized 0-1)
            current_price: Current market price
            volatility: Current volatility (optional, will estimate if not provided)
            volume: Current volume
            avg_volume: Average volume (for market impact calculation)
            timestamp: Timestamp for time-of-day adjustment (optional)
        
        Returns:
            Slippage as fraction (e.g., 0.00015 = 1.5 bps)
        """
        if not self.enabled:
            return 0.0
        
        if abs(order_size) < 1e-6:  # No order
            return 0.0
        
        # Base slippage
        total_slippage = self.base_slippage
        
        # Market impact (increases with order size relative to volume)
        if avg_volume and avg_volume > 0:
            # Convert normalized order size to actual volume
            # Assuming order_size is normalized position change (0-1)
            # For ES futures, 1.0 = full position = ~$50k notional
            # Estimate volume impact: larger orders move prices more
            volume_ratio = abs(order_size) / max(avg_volume, 1.0)
            # Use square root model (Almgren-Chriss)
            market_impact = np.sqrt(volume_ratio) * self.impact_coefficient
            total_slippage += market_impact
        
        # Volatility adjustment (wider spreads in volatile markets)
        if volatility is not None and volatility > 0:
            vol_adjustment = volatility * self.vol_multiplier
            total_slippage += vol_adjustment
        
        # Time of day adjustment
        time_multiplier = self._get_time_multiplier(timestamp)
        total_slippage *= time_multiplier
        
        # Ensure slippage is non-negative and reasonable (max 1% = 0.01)
        total_slippage = min(max(total_slippage, 0.0), 0.01)
        
        return total_slippage
    
    def _get_time_multiplier(self, timestamp: Optional[datetime]) -> float:
        """
        Get time-of-day multiplier for slippage.
        
        Args:
            timestamp: Current timestamp (optional)
        
        Returns:
            Multiplier (1.0 = normal, >1.0 = wider spreads)
        """
        if timestamp is None:
            return self.time_multipliers["normal_hours"]
        
        current_time = timestamp.time()
        
        # Market open (09:30-10:00 ET)
        if self.market_open_start <= current_time <= self.market_open_end:
            return self.time_multipliers["market_open"]
        
        # Market close (15:30-16:00 ET)
        if self.market_close_start <= current_time <= self.market_close_end:
            return self.time_multipliers["market_close"]
        
        # After hours (before 09:30 or after 16:00)
        if current_time < self.market_open_start or current_time > self.market_close_end:
            return self.time_multipliers["after_hours"]
        
        # Normal trading hours
        return self.time_multipliers["normal_hours"]
    
    def apply_slippage(
        self,
        intended_price: float,
        order_size: float,
        is_buy: bool,
        volatility: Optional[float] = None,
        volume: Optional[float] = None,
        avg_volume: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> float:
        """
        Apply slippage to intended execution price.
        
        Args:
            intended_price: Intended execution price
            order_size: Order size (absolute value, normalized 0-1)
            is_buy: True for buy orders, False for sell orders
            volatility: Current volatility (optional)
            volume: Current volume (optional)
            avg_volume: Average volume (optional)
            timestamp: Timestamp for time-of-day adjustment (optional)
        
        Returns:
            Actual execution price (with slippage applied)
        """
        slippage = self.calculate_slippage(
            order_size=order_size,
            current_price=intended_price,
            volatility=volatility,
            volume=volume,
            avg_volume=avg_volume,
            timestamp=timestamp
        )
        
        # Buy orders: pay more (positive slippage)
        # Sell orders: receive less (negative slippage)
        if is_buy:
            actual_price = intended_price * (1.0 + slippage)
        else:
            actual_price = intended_price * (1.0 - slippage)
        
        return actual_price

