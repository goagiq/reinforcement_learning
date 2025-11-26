"""
Order Book Simulation

Simulates order book from tick/bar data to analyze depth and liquidity.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict


@dataclass
class OrderBookLevel:
    """Single level in the order book"""
    price: float
    bid_quantity: float  # Quantity available at bid
    ask_quantity: float  # Quantity available at ask
    bid_orders: int  # Number of bid orders
    ask_orders: int  # Number of ask orders


@dataclass
class OrderBookSnapshot:
    """Snapshot of the order book at a point in time"""
    timestamp: datetime
    bid_levels: List[OrderBookLevel]  # Sorted by price (highest first)
    ask_levels: List[OrderBookLevel]  # Sorted by price (lowest first)
    best_bid: float
    best_ask: float
    spread: float
    mid_price: float
    total_bid_depth: float  # Total quantity on bid side
    total_ask_depth: float  # Total quantity on ask side


class OrderBookSimulator:
    """
    Simulates order book from tick/bar data.
    
    Features:
    - Depth analysis at price levels
    - Liquidity assessment
    - Spread calculation
    - Market impact estimation
    """
    
    def __init__(
        self,
        num_levels: int = 10,  # Number of price levels to simulate
        base_depth: float = 100.0,  # Base depth per level
        depth_volatility: float = 0.3,  # Volatility of depth
        spread_bps: float = 0.5  # Base spread in basis points
    ):
        """
        Initialize order book simulator.
        
        Args:
            num_levels: Number of price levels to simulate
            base_depth: Base depth (quantity) per level
            depth_volatility: Volatility of depth (0-1)
            spread_bps: Base spread in basis points
        """
        self.num_levels = num_levels
        self.base_depth = base_depth
        self.depth_volatility = depth_volatility
        self.spread_bps = spread_bps
        
        # Current order book state
        self.current_snapshot: Optional[OrderBookSnapshot] = None
    
    def generate_order_book(
        self,
        current_price: float,
        volume: float,
        volatility: float,
        timestamp: Optional[datetime] = None
    ) -> OrderBookSnapshot:
        """
        Generate order book snapshot from market data.
        
        Args:
            current_price: Current market price
            volume: Current volume
            volatility: Market volatility
            timestamp: Timestamp for snapshot
        
        Returns:
            OrderBookSnapshot
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate spread (wider in volatile markets)
        spread_factor = 1.0 + (volatility * 2.0)  # Wider spread with volatility
        spread = current_price * (self.spread_bps / 10000) * spread_factor
        
        # Best bid/ask
        best_bid = current_price - (spread / 2.0)
        best_ask = current_price + (spread / 2.0)
        mid_price = current_price
        
        # Generate bid levels (below current price)
        bid_levels = []
        total_bid_depth = 0.0
        
        for i in range(self.num_levels):
            # Price decreases from best_bid
            price_offset = i * (spread / 2.0)  # Each level is spread/2 apart
            price = best_bid - price_offset
            
            # Depth decreases with distance from best bid
            depth_factor = np.exp(-i * 0.2)  # Exponential decay
            depth = self.base_depth * depth_factor * (1.0 + np.random.normal(0, self.depth_volatility))
            depth = max(0.1, depth)  # Minimum depth
            
            # Number of orders (more at best bid)
            num_orders = max(1, int(depth / 10.0) + np.random.randint(-2, 3))
            
            bid_levels.append(OrderBookLevel(
                price=price,
                bid_quantity=depth,
                ask_quantity=0.0,
                bid_orders=num_orders,
                ask_orders=0
            ))
            total_bid_depth += depth
        
        # Generate ask levels (above current price)
        ask_levels = []
        total_ask_depth = 0.0
        
        for i in range(self.num_levels):
            # Price increases from best_ask
            price_offset = i * (spread / 2.0)
            price = best_ask + price_offset
            
            # Depth decreases with distance from best ask
            depth_factor = np.exp(-i * 0.2)
            depth = self.base_depth * depth_factor * (1.0 + np.random.normal(0, self.depth_volatility))
            depth = max(0.1, depth)
            
            num_orders = max(1, int(depth / 10.0) + np.random.randint(-2, 3))
            
            ask_levels.append(OrderBookLevel(
                price=price,
                bid_quantity=0.0,
                ask_quantity=depth,
                bid_orders=0,
                ask_orders=num_orders
            ))
            total_ask_depth += depth
        
        # Sort levels (bids descending, asks ascending)
        bid_levels.sort(key=lambda x: x.price, reverse=True)
        ask_levels.sort(key=lambda x: x.price)
        
        snapshot = OrderBookSnapshot(
            timestamp=timestamp,
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            mid_price=mid_price,
            total_bid_depth=total_bid_depth,
            total_ask_depth=total_ask_depth
        )
        
        self.current_snapshot = snapshot
        return snapshot
    
    def assess_liquidity(
        self,
        order_size: float,
        is_buy: bool = True
    ) -> Dict:
        """
        Assess liquidity for a given order size.
        
        Args:
            order_size: Order size in units
            is_buy: True for buy order, False for sell
        
        Returns:
            Dictionary with liquidity metrics
        """
        if self.current_snapshot is None:
            return {"error": "No order book snapshot available"}
        
        if is_buy:
            levels = self.current_snapshot.ask_levels
            best_price = self.current_snapshot.best_ask
        else:
            levels = self.current_snapshot.bid_levels
            best_price = self.current_snapshot.best_bid
        
        # Calculate how many levels needed to fill order
        remaining_size = order_size
        levels_needed = 0
        total_cost = 0.0
        weighted_avg_price = 0.0
        
        for level in levels:
            if remaining_size <= 0:
                break
            
            available = level.ask_quantity if is_buy else level.bid_quantity
            fill_size = min(remaining_size, available)
            
            cost = fill_size * level.price
            total_cost += cost
            weighted_avg_price += level.price * fill_size
            
            remaining_size -= fill_size
            levels_needed += 1
        
        if order_size > 0:
            weighted_avg_price = weighted_avg_price / order_size
        else:
            weighted_avg_price = best_price
        
        # Calculate market impact
        market_impact = abs(weighted_avg_price - best_price) / best_price if best_price > 0 else 0.0
        
        # Liquidity score (0-1, higher = more liquid)
        if remaining_size > 0:
            # Order cannot be fully filled
            fill_rate = (order_size - remaining_size) / order_size
            liquidity_score = fill_rate * 0.5  # Penalty for partial fill
        else:
            # Order can be fully filled
            liquidity_score = 1.0 - (levels_needed / self.num_levels) * 0.5  # Penalty for using multiple levels
        
        return {
            "order_size": order_size,
            "is_buy": is_buy,
            "best_price": best_price,
            "weighted_avg_price": weighted_avg_price,
            "market_impact": market_impact,
            "market_impact_bps": market_impact * 10000,
            "levels_needed": levels_needed,
            "fill_rate": 1.0 if remaining_size <= 0 else (order_size - remaining_size) / order_size,
            "can_fill_fully": remaining_size <= 0,
            "liquidity_score": liquidity_score,
            "estimated_cost": total_cost
        }
    
    def get_depth_at_price(
        self,
        price: float,
        is_buy: bool = True
    ) -> float:
        """
        Get available depth at a specific price level.
        
        Args:
            price: Price level to check
            is_buy: True for ask side, False for bid side
        
        Returns:
            Available quantity at price level
        """
        if self.current_snapshot is None:
            return 0.0
        
        levels = self.current_snapshot.ask_levels if is_buy else self.current_snapshot.bid_levels
        
        # Find closest level
        closest_level = None
        min_diff = float('inf')
        
        for level in levels:
            diff = abs(level.price - price)
            if diff < min_diff:
                min_diff = diff
                closest_level = level
        
        if closest_level and min_diff < (closest_level.price * 0.001):  # Within 0.1%
            return closest_level.ask_quantity if is_buy else closest_level.bid_quantity
        
        return 0.0
    
    def get_order_book_summary(self) -> Dict:
        """
        Get summary of current order book.
        
        Returns:
            Dictionary with order book summary
        """
        if self.current_snapshot is None:
            return {"error": "No order book snapshot available"}
        
        return {
            "timestamp": self.current_snapshot.timestamp,
            "best_bid": self.current_snapshot.best_bid,
            "best_ask": self.current_snapshot.best_ask,
            "spread": self.current_snapshot.spread,
            "spread_bps": (self.current_snapshot.spread / self.current_snapshot.mid_price) * 10000,
            "mid_price": self.current_snapshot.mid_price,
            "total_bid_depth": self.current_snapshot.total_bid_depth,
            "total_ask_depth": self.current_snapshot.total_ask_depth,
            "num_bid_levels": len(self.current_snapshot.bid_levels),
            "num_ask_levels": len(self.current_snapshot.ask_levels)
        }

