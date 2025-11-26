"""
Partial Fill Model

Models partial fills based on volume and order book depth.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Fill:
    """Single fill event"""
    timestamp: datetime
    quantity: float  # Filled quantity
    price: float  # Fill price
    fill_type: str  # "full", "partial", "none"


@dataclass
class OrderFillResult:
    """Result of order fill attempt"""
    order_id: str
    total_quantity: float  # Original order quantity
    filled_quantity: float  # Total filled quantity
    remaining_quantity: float  # Remaining quantity
    fills: List[Fill]  # List of fill events
    weighted_avg_price: float  # Weighted average fill price
    fill_rate: float  # Fill rate (0-1)
    is_fully_filled: bool


class PartialFillModel:
    """
    Models partial fills based on:
    - Order book depth
    - Volume at price levels
    - Time in force
    - Market conditions
    """
    
    def __init__(
        self,
        base_fill_probability: float = 0.95,  # Base probability of full fill
        volume_impact_factor: float = 0.5,  # Impact of order size vs volume
        time_decay_factor: float = 0.1  # Fill probability decay over time
    ):
        """
        Initialize partial fill model.
        
        Args:
            base_fill_probability: Base probability of full fill for small orders
            volume_impact_factor: How much order size affects fill probability
            time_decay_factor: How fill probability decays over time
        """
        self.base_fill_probability = base_fill_probability
        self.volume_impact_factor = volume_impact_factor
        self.time_decay_factor = time_decay_factor
        
        # Fill history tracking
        self.fill_history: List[OrderFillResult] = []
    
    def simulate_fill(
        self,
        order_id: str,
        order_quantity: float,
        order_price: Optional[float],  # Limit price (None for market orders)
        current_price: float,
        order_book_depth: float,  # Available depth at price
        avg_volume: float,
        is_buy: bool = True,
        order_type: str = "market",  # "market", "limit"
        timestamp: Optional[datetime] = None
    ) -> OrderFillResult:
        """
        Simulate order fill with potential partial fills.
        
        Args:
            order_id: Order identifier
            order_quantity: Total order quantity
            order_price: Limit price (if limit order)
            current_price: Current market price
            order_book_depth: Available depth at price level
            avg_volume: Average volume (for size comparison)
            is_buy: True for buy order, False for sell
            order_type: Order type ("market" or "limit")
            timestamp: Order timestamp
        
        Returns:
            OrderFillResult with fill details
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        fills = []
        filled_quantity = 0.0
        total_cost = 0.0
        
        # Market orders: fill immediately based on available depth
        if order_type == "market":
            # Calculate fill probability based on order size vs depth
            if order_book_depth > 0:
                fill_ratio = min(1.0, order_book_depth / order_quantity)
            else:
                fill_ratio = 0.0
            
            # Adjust for order size (larger orders harder to fill)
            size_factor = 1.0 - (order_quantity / max(avg_volume, order_quantity)) * self.volume_impact_factor
            fill_probability = self.base_fill_probability * fill_ratio * size_factor
            
            # Simulate fill
            if np.random.random() < fill_probability:
                # Full fill
                fill_quantity = order_quantity
                fill_price = current_price
                filled_quantity = fill_quantity
                fills.append(Fill(
                    timestamp=timestamp,
                    quantity=fill_quantity,
                    price=fill_price,
                    fill_type="full"
                ))
                total_cost = fill_quantity * fill_price
            else:
                # Partial fill or no fill
                if fill_ratio > 0.1:  # At least 10% available
                    # Partial fill
                    fill_quantity = order_quantity * fill_ratio * np.random.uniform(0.5, 1.0)
                    fill_price = current_price
                    filled_quantity = fill_quantity
                    fills.append(Fill(
                        timestamp=timestamp,
                        quantity=fill_quantity,
                        price=fill_price,
                        fill_type="partial"
                    ))
                    total_cost = fill_quantity * fill_price
                else:
                    # No fill
                    fills.append(Fill(
                        timestamp=timestamp,
                        quantity=0.0,
                        price=current_price,
                        fill_type="none"
                    ))
        
        # Limit orders: fill only if price is reached
        elif order_type == "limit":
            if order_price is None:
                # Invalid limit order
                return OrderFillResult(
                    order_id=order_id,
                    total_quantity=order_quantity,
                    filled_quantity=0.0,
                    remaining_quantity=order_quantity,
                    fills=[],
                    weighted_avg_price=current_price,
                    fill_rate=0.0,
                    is_fully_filled=False
                )
            
            # Check if limit price is reached
            if is_buy:
                price_reached = current_price <= order_price
            else:
                price_reached = current_price >= order_price
            
            if price_reached:
                # Price reached, attempt fill
                # Fill probability based on depth
                if order_book_depth > 0:
                    fill_ratio = min(1.0, order_book_depth / order_quantity)
                else:
                    fill_ratio = 0.0
                
                # Limit orders have higher fill probability if price is reached
                fill_probability = min(1.0, fill_ratio * 1.2)  # 20% boost
                
                if np.random.random() < fill_probability:
                    # Full or partial fill
                    if fill_ratio >= 0.95:
                        fill_quantity = order_quantity
                        fill_type = "full"
                    else:
                        fill_quantity = order_quantity * fill_ratio
                        fill_type = "partial"
                    
                    fill_price = order_price
                    filled_quantity = fill_quantity
                    fills.append(Fill(
                        timestamp=timestamp,
                        quantity=fill_quantity,
                        price=fill_price,
                        fill_type=fill_type
                    ))
                    total_cost = fill_quantity * fill_price
                else:
                    # No fill (despite price being reached)
                    fills.append(Fill(
                        timestamp=timestamp,
                        quantity=0.0,
                        price=order_price,
                        fill_type="none"
                    ))
            else:
                # Price not reached, no fill
                fills.append(Fill(
                    timestamp=timestamp,
                    quantity=0.0,
                    price=order_price,
                    fill_type="none"
                ))
        
        # Calculate weighted average price
        if filled_quantity > 0:
            weighted_avg_price = total_cost / filled_quantity
        else:
            weighted_avg_price = current_price
        
        remaining_quantity = order_quantity - filled_quantity
        fill_rate = filled_quantity / order_quantity if order_quantity > 0 else 0.0
        is_fully_filled = filled_quantity >= order_quantity
        
        result = OrderFillResult(
            order_id=order_id,
            total_quantity=order_quantity,
            filled_quantity=filled_quantity,
            remaining_quantity=remaining_quantity,
            fills=fills,
            weighted_avg_price=weighted_avg_price,
            fill_rate=fill_rate,
            is_fully_filled=is_fully_filled
        )
        
        self.fill_history.append(result)
        return result
    
    def get_fill_statistics(self) -> Dict:
        """
        Get statistics on fill performance.
        
        Returns:
            Dictionary with fill statistics
        """
        if len(self.fill_history) == 0:
            return {"message": "No fill history available"}
        
        total_orders = len(self.fill_history)
        fully_filled = sum(1 for f in self.fill_history if f.is_fully_filled)
        partially_filled = sum(1 for f in self.fill_history if 0 < f.fill_rate < 1.0)
        not_filled = sum(1 for f in self.fill_history if f.fill_rate == 0.0)
        
        avg_fill_rate = np.mean([f.fill_rate for f in self.fill_history])
        
        return {
            "total_orders": total_orders,
            "fully_filled": fully_filled,
            "partially_filled": partially_filled,
            "not_filled": not_filled,
            "full_fill_rate": fully_filled / total_orders if total_orders > 0 else 0.0,
            "avg_fill_rate": avg_fill_rate,
            "partial_fill_rate": partially_filled / total_orders if total_orders > 0 else 0.0
        }

