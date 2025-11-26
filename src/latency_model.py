"""
Latency Modeling

Models execution latency including network delays, processing time, and market data delays.
"""

import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import time


@dataclass
class LatencyBreakdown:
    """Breakdown of latency components"""
    network_latency: float  # Network transmission time (seconds)
    processing_latency: float  # Order processing time (seconds)
    exchange_latency: float  # Exchange processing time (seconds)
    market_data_latency: float  # Market data delay (seconds)
    total_latency: float  # Total latency (seconds)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "network_latency": self.network_latency,
            "processing_latency": self.processing_latency,
            "exchange_latency": self.exchange_latency,
            "market_data_latency": self.market_data_latency,
            "total_latency": self.total_latency
        }


class LatencyModel:
    """
    Models execution latency for realistic backtesting and live trading.
    
    Components:
    - Network latency (order transmission)
    - Processing latency (system processing)
    - Exchange latency (exchange processing)
    - Market data latency (data feed delay)
    """
    
    def __init__(
        self,
        base_network_latency: float = 0.001,  # 1ms base network latency
        network_latency_std: float = 0.0005,  # 0.5ms std dev
        processing_latency: float = 0.0005,  # 0.5ms processing
        exchange_latency: float = 0.002,  # 2ms exchange processing
        market_data_latency: float = 0.001,  # 1ms market data delay
        high_latency_threshold: float = 0.01,  # 10ms threshold for high latency
        latency_spike_probability: float = 0.01  # 1% chance of latency spike
    ):
        """
        Initialize latency model.
        
        Args:
            base_network_latency: Base network latency in seconds
            network_latency_std: Standard deviation of network latency
            processing_latency: Processing latency in seconds
            exchange_latency: Exchange processing latency in seconds
            market_data_latency: Market data feed latency in seconds
            high_latency_threshold: Threshold for high latency events
            latency_spike_probability: Probability of latency spike
        """
        self.base_network_latency = base_network_latency
        self.network_latency_std = network_latency_std
        self.processing_latency = processing_latency
        self.exchange_latency = exchange_latency
        self.market_data_latency = market_data_latency
        self.high_latency_threshold = high_latency_threshold
        self.latency_spike_probability = latency_spike_probability
        
        # Latency history tracking
        self.latency_history: List[float] = []
        self.max_history = 1000
    
    def simulate_latency(
        self,
        order_size: float = 1.0,
        volatility: float = 0.01,
        volume: float = 1000.0,
        is_market_hours: bool = True
    ) -> LatencyBreakdown:
        """
        Simulate total execution latency.
        
        Args:
            order_size: Order size (larger orders may have more latency)
            volatility: Market volatility (high vol may increase latency)
            volume: Current volume (high volume may increase latency)
            is_market_hours: Whether it's market hours (affects latency)
        
        Returns:
            LatencyBreakdown with all components
        """
        # Network latency (varies with conditions)
        network_latency = self.base_network_latency
        
        # Add variability
        network_latency += np.random.normal(0, self.network_latency_std)
        
        # High volume can increase latency
        if volume > 10000:
            network_latency *= 1.2
        
        # High volatility can increase latency
        if volatility > 0.03:
            network_latency *= 1.1
        
        # Latency spikes (rare events)
        if np.random.random() < self.latency_spike_probability:
            network_latency *= np.random.uniform(5.0, 20.0)  # 5-20x spike
        
        network_latency = max(0.0001, network_latency)  # Minimum 0.1ms
        
        # Processing latency (relatively constant)
        processing = self.processing_latency
        
        # Exchange latency (varies slightly)
        exchange = self.exchange_latency * np.random.uniform(0.8, 1.2)
        
        # Market data latency (depends on feed)
        market_data = self.market_data_latency
        
        # After hours may have higher latency
        if not is_market_hours:
            network_latency *= 1.5
            market_data *= 2.0
        
        total_latency = network_latency + processing + exchange + market_data
        
        # Track history
        self.latency_history.append(total_latency)
        if len(self.latency_history) > self.max_history:
            self.latency_history.pop(0)
        
        return LatencyBreakdown(
            network_latency=network_latency,
            processing_latency=processing,
            exchange_latency=exchange,
            market_data_latency=market_data,
            total_latency=total_latency
        )
    
    def apply_latency_delay(
        self,
        intended_price: float,
        latency: LatencyBreakdown,
        price_change_rate: float = 0.0  # Price change per second
    ) -> float:
        """
        Apply latency delay to price.
        
        Simulates price movement during latency period.
        
        Args:
            intended_price: Price when order was submitted
            latency: Latency breakdown
            price_change_rate: Price change rate (dollars per second)
        
        Returns:
            Actual execution price after latency
        """
        # Price moves during latency period
        price_movement = price_change_rate * latency.total_latency
        
        # Add some randomness
        price_movement += np.random.normal(0, abs(price_movement) * 0.1)
        
        actual_price = intended_price + price_movement
        
        return actual_price
    
    def get_latency_statistics(self) -> Dict:
        """
        Get latency statistics from history.
        
        Returns:
            Dictionary with latency statistics
        """
        if len(self.latency_history) == 0:
            return {"message": "No latency history available"}
        
        latencies = np.array(self.latency_history)
        
        return {
            "count": len(latencies),
            "mean_latency": float(np.mean(latencies)),
            "median_latency": float(np.median(latencies)),
            "std_latency": float(np.std(latencies)),
            "min_latency": float(np.min(latencies)),
            "max_latency": float(np.max(latencies)),
            "p95_latency": float(np.percentile(latencies, 95)),
            "p99_latency": float(np.percentile(latencies, 99)),
            "high_latency_count": int(np.sum(latencies > self.high_latency_threshold)),
            "high_latency_rate": float(np.mean(latencies > self.high_latency_threshold))
        }
    
    def is_low_latency(self, latency: float) -> bool:
        """
        Check if latency is considered low.
        
        Args:
            latency: Latency in seconds
        
        Returns:
            True if latency is low
        """
        return latency < (self.base_network_latency + self.processing_latency + self.exchange_latency) * 1.5
    
    def estimate_slippage_from_latency(
        self,
        latency: float,
        volatility: float,
        current_price: float
    ) -> float:
        """
        Estimate slippage caused by latency.
        
        Args:
            latency: Total latency in seconds
            volatility: Market volatility
            current_price: Current market price
        
        Returns:
            Estimated slippage in basis points
        """
        # Slippage increases with latency and volatility
        # Formula: slippage_bps = latency_seconds * volatility * price * factor
        factor = 100.0  # Conversion factor
        slippage_bps = latency * volatility * factor
        
        return slippage_bps

