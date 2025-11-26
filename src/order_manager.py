"""
Order Manager for Different Order Types

Supports market orders, limit orders, stop orders, and stop-limit orders.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order representation"""
    order_id: str
    instrument: str
    order_type: OrderType
    quantity: float  # Position size (-1.0 to 1.0)
    price: Optional[float] = None  # Limit price for limit orders
    stop_price: Optional[float] = None  # Stop price for stop orders
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    fill_price: Optional[float] = None
    submit_time: Optional[datetime] = None
    fill_time: Optional[datetime] = None
    expiry_time: Optional[datetime] = None  # Order expiry (for GTC orders)
    
    def is_filled(self) -> bool:
        """Check if order is fully filled"""
        return self.status == OrderStatus.FILLED
    
    def is_active(self) -> bool:
        """Check if order is still active"""
        return self.status == OrderStatus.PENDING or self.status == OrderStatus.PARTIALLY_FILLED


class OrderManager:
    """
    Manages different order types:
    - Market orders (immediate, guaranteed fill)
    - Limit orders (better price, may not fill)
    - Stop orders (risk management)
    - Stop-limit orders (hybrid)
    """
    
    def __init__(
        self,
        fill_probability_limit: float = 0.8,  # Probability limit orders fill
        max_order_age_seconds: int = 3600  # Max age for limit orders (1 hour)
    ):
        """
        Initialize order manager.
        
        Args:
            fill_probability_limit: Probability that limit orders fill (0-1)
            max_order_age_seconds: Maximum age for limit orders before cancellation
        """
        self.fill_probability_limit = fill_probability_limit
        self.max_order_age_seconds = max_order_age_seconds
        
        # Active orders
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []
        self.cancelled_orders: List[Order] = []
        
        self._order_counter = 0
    
    def submit_order(
        self,
        order_type: OrderType,
        instrument: str,
        quantity: float,
        current_price: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        bid_price: Optional[float] = None,
        ask_price: Optional[float] = None
    ) -> Order:
        """
        Submit order with appropriate type.
        
        Args:
            order_type: Type of order (MARKET, LIMIT, STOP, STOP_LIMIT)
            instrument: Instrument symbol
            quantity: Position size (-1.0 to 1.0)
            current_price: Current market price
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            bid_price: Current bid price (for limit order validation)
            ask_price: Current ask price (for limit order validation)
        
        Returns:
            Order object
        """
        # Generate order ID
        self._order_counter += 1
        order_id = f"{instrument}_{order_type.value}_{self._order_counter}_{datetime.now().timestamp()}"
        
        # Validate order parameters
        if order_type == OrderType.LIMIT and price is None:
            raise ValueError("Limit orders require a price")
        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price is None:
            raise ValueError("Stop orders require a stop_price")
        
        # Create order
        order = Order(
            order_id=order_id,
            instrument=instrument,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            submit_time=datetime.now()
        )
        
        # Market orders execute immediately
        if order_type == OrderType.MARKET:
            order = self._execute_market_order(order, current_price, bid_price, ask_price)
        else:
            # Other orders go to pending
            self.pending_orders[order_id] = order
        
        return order
    
    def _execute_market_order(
        self,
        order: Order,
        current_price: float,
        bid_price: Optional[float],
        ask_price: Optional[float]
    ) -> Order:
        """Execute market order immediately"""
        # Market orders execute at current price (or mid-price if available)
        if bid_price is not None and ask_price is not None:
            # Use mid-price for market orders
            fill_price = (bid_price + ask_price) / 2.0
        else:
            fill_price = current_price
        
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.fill_price = fill_price
        order.fill_time = datetime.now()
        
        self.filled_orders.append(order)
        return order
    
    def process_pending_orders(
        self,
        instrument: str,
        current_price: float,
        high_price: float,
        low_price: float,
        bid_price: Optional[float] = None,
        ask_price: Optional[float] = None
    ) -> List[Order]:
        """
        Process pending orders for an instrument.
        
        Args:
            instrument: Instrument symbol
            current_price: Current market price
            high_price: High price of the bar
            low_price: Low price of the bar
            bid_price: Current bid price
            ask_price: Current ask price
        
        Returns:
            List of filled orders
        """
        filled_orders = []
        
        # Get orders for this instrument
        instrument_orders = [
            order for order in self.pending_orders.values()
            if order.instrument == instrument and order.is_active()
        ]
        
        for order in instrument_orders:
            filled = False
            
            if order.order_type == OrderType.LIMIT:
                filled = self._check_limit_order_fill(order, current_price, high_price, low_price, bid_price, ask_price)
            
            elif order.order_type == OrderType.STOP:
                filled = self._check_stop_order_fill(order, current_price, high_price, low_price)
            
            elif order.order_type == OrderType.STOP_LIMIT:
                filled = self._check_stop_limit_order_fill(order, current_price, high_price, low_price, bid_price, ask_price)
            
            # Check order expiry
            if not filled and order.submit_time:
                age_seconds = (datetime.now() - order.submit_time).total_seconds()
                if age_seconds > self.max_order_age_seconds:
                    order.status = OrderStatus.CANCELLED
                    self.cancelled_orders.append(order)
                    if order.order_id in self.pending_orders:
                        del self.pending_orders[order.order_id]
            
            if filled:
                filled_orders.append(order)
                if order.order_id in self.pending_orders:
                    del self.pending_orders[order.order_id]
        
        return filled_orders
    
    def _check_limit_order_fill(
        self,
        order: Order,
        current_price: float,
        high_price: float,
        low_price: float,
        bid_price: Optional[float],
        ask_price: Optional[float]
    ) -> bool:
        """
        Check if limit order should fill.
        
        Buy limit: Fills if price <= limit price
        Sell limit: Fills if price >= limit price
        """
        if order.price is None:
            return False
        
        is_buy = order.quantity > 0
        
        if is_buy:
            # Buy limit: price must be <= limit price
            # Check if price touched limit during the bar
            if low_price <= order.price:
                # Determine fill probability based on how close price got to limit
                if current_price <= order.price:
                    fill_prob = 1.0  # Price is at or below limit
                else:
                    # Price touched limit but moved back up
                    fill_prob = self.fill_probability_limit * (1.0 - (current_price - order.price) / (high_price - order.price + 1e-6))
                
                if np.random.random() < fill_prob:
                    fill_price = order.price if bid_price is None else min(order.price, ask_price or order.price)
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.fill_price = fill_price
                    order.fill_time = datetime.now()
                    self.filled_orders.append(order)
                    return True
        else:
            # Sell limit: price must be >= limit price
            if high_price >= order.price:
                if current_price >= order.price:
                    fill_prob = 1.0
                else:
                    fill_prob = self.fill_probability_limit * (1.0 - (order.price - current_price) / (order.price - low_price + 1e-6))
                
                if np.random.random() < fill_prob:
                    fill_price = order.price if ask_price is None else max(order.price, bid_price or order.price)
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.fill_price = fill_price
                    order.fill_time = datetime.now()
                    self.filled_orders.append(order)
                    return True
        
        return False
    
    def _check_stop_order_fill(
        self,
        order: Order,
        current_price: float,
        high_price: float,
        low_price: float
    ) -> bool:
        """
        Check if stop order should trigger.
        
        Buy stop: Triggers if price >= stop price
        Sell stop: Triggers if price <= stop price
        """
        if order.stop_price is None:
            return False
        
        is_buy = order.quantity > 0
        
        if is_buy:
            # Buy stop: triggers if price >= stop price
            if high_price >= order.stop_price:
                # Execute as market order
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.fill_price = max(order.stop_price, current_price)  # Slippage on stop orders
                order.fill_time = datetime.now()
                self.filled_orders.append(order)
                return True
        else:
            # Sell stop: triggers if price <= stop price
            if low_price <= order.stop_price:
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.fill_price = min(order.stop_price, current_price)  # Slippage on stop orders
                order.fill_time = datetime.now()
                self.filled_orders.append(order)
                return True
        
        return False
    
    def _check_stop_limit_order_fill(
        self,
        order: Order,
        current_price: float,
        high_price: float,
        low_price: float,
        bid_price: Optional[float],
        ask_price: Optional[float]
    ) -> bool:
        """
        Check if stop-limit order should trigger and fill.
        
        First checks if stop is triggered, then checks if limit can fill.
        """
        if order.stop_price is None or order.price is None:
            return False
        
        is_buy = order.quantity > 0
        
        if is_buy:
            # Buy stop-limit: stop triggers if price >= stop, then limit must be <= price
            if high_price >= order.stop_price:
                # Stop triggered, now check limit
                if low_price <= order.price:
                    fill_price = order.price if bid_price is None else min(order.price, ask_price or order.price)
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.fill_price = fill_price
                    order.fill_time = datetime.now()
                    self.filled_orders.append(order)
                    return True
        else:
            # Sell stop-limit: stop triggers if price <= stop, then limit must be >= price
            if low_price <= order.stop_price:
                if high_price >= order.price:
                    fill_price = order.price if ask_price is None else max(order.price, bid_price or order.price)
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.fill_price = fill_price
                    order.fill_time = datetime.now()
                    self.filled_orders.append(order)
                    return True
        
        return False
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.
        
        Returns:
            True if order was cancelled, False if not found
        """
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.status = OrderStatus.CANCELLED
            self.cancelled_orders.append(order)
            del self.pending_orders[order_id]
            return True
        return False
    
    def get_pending_orders(self, instrument: Optional[str] = None) -> List[Order]:
        """Get list of pending orders"""
        orders = list(self.pending_orders.values())
        if instrument:
            orders = [o for o in orders if o.instrument == instrument]
        return orders
    
    def get_order_statistics(self) -> Dict:
        """Get order execution statistics"""
        total_orders = len(self.filled_orders) + len(self.cancelled_orders) + len(self.pending_orders)
        
        return {
            "total_orders": total_orders,
            "filled_orders": len(self.filled_orders),
            "cancelled_orders": len(self.cancelled_orders),
            "pending_orders": len(self.pending_orders),
            "fill_rate": len(self.filled_orders) / total_orders if total_orders > 0 else 0.0
        }

