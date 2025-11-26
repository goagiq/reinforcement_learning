"""
Execution Quality Tracker

Tracks execution quality metrics:
- Slippage (actual vs. expected)
- Fill rate (partial vs. full fills)
- Latency (order submission to fill)
- Market impact (price movement from order)
"""

import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque


@dataclass
class ExecutionRecord:
    """Record of a single execution"""
    timestamp: datetime
    expected_price: float
    actual_price: float
    order_size: float
    fill_time: datetime
    order_submit_time: datetime
    slippage: float
    latency_seconds: float
    market_impact: Optional[float] = None
    volatility: Optional[float] = None
    volume: Optional[float] = None


class ExecutionQualityTracker:
    """
    Tracks execution quality metrics for performance analysis.
    
    Metrics tracked:
    - Average slippage (actual vs. expected)
    - Slippage distribution (percentiles)
    - Average latency
    - Fill rate
    - Market impact
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize execution quality tracker.
        
        Args:
            max_history: Maximum number of executions to keep in history
        """
        self.max_history = max_history
        self.executions: deque = deque(maxlen=max_history)
        self.slippage_history: deque = deque(maxlen=max_history)
        self.latency_history: deque = deque(maxlen=max_history)
        
        # Statistics (updated incrementally)
        self.total_executions = 0
        self.total_slippage = 0.0
        self.total_latency = 0.0
    
    def track_execution(
        self,
        expected_price: float,
        actual_price: float,
        order_size: float,
        fill_time: datetime,
        order_submit_time: Optional[datetime] = None,
        market_impact: Optional[float] = None,
        volatility: Optional[float] = None,
        volume: Optional[float] = None
    ) -> ExecutionRecord:
        """
        Track an execution and calculate metrics.
        
        Args:
            expected_price: Expected execution price
            actual_price: Actual execution price
            order_size: Order size (absolute value)
            fill_time: Time when order was filled
            order_submit_time: Time when order was submitted (optional)
            market_impact: Market impact in basis points (optional)
            volatility: Volatility at execution time (optional)
            volume: Volume at execution time (optional)
        
        Returns:
            ExecutionRecord with all metrics
        """
        if order_submit_time is None:
            order_submit_time = fill_time
        
        # Calculate slippage (as fraction)
        if expected_price > 0:
            slippage = (actual_price - expected_price) / expected_price
        else:
            slippage = 0.0
        
        # Calculate latency
        latency_seconds = (fill_time - order_submit_time).total_seconds()
        
        # Create record
        record = ExecutionRecord(
            timestamp=fill_time,
            expected_price=expected_price,
            actual_price=actual_price,
            order_size=order_size,
            fill_time=fill_time,
            order_submit_time=order_submit_time,
            slippage=slippage,
            latency_seconds=latency_seconds,
            market_impact=market_impact,
            volatility=volatility,
            volume=volume
        )
        
        # Store
        self.executions.append(record)
        self.slippage_history.append(slippage)
        self.latency_history.append(latency_seconds)
        
        # Update statistics
        self.total_executions += 1
        self.total_slippage += abs(slippage)
        self.total_latency += latency_seconds
        
        return record
    
    def get_statistics(self) -> Dict:
        """
        Get execution quality statistics.
        
        Returns:
            Dictionary with statistics:
            - avg_slippage: Average absolute slippage
            - median_slippage: Median absolute slippage
            - p95_slippage: 95th percentile slippage
            - avg_latency: Average latency in seconds
            - total_executions: Total number of executions
            - slippage_distribution: Percentiles of slippage
        """
        if len(self.slippage_history) == 0:
            return {
                "avg_slippage": 0.0,
                "median_slippage": 0.0,
                "p95_slippage": 0.0,
                "avg_latency": 0.0,
                "total_executions": 0,
                "slippage_distribution": {}
            }
        
        slippage_array = np.array(list(self.slippage_history))
        latency_array = np.array(list(self.latency_history))
        
        # Calculate percentiles
        abs_slippage = np.abs(slippage_array)
        
        return {
            "avg_slippage": float(np.mean(abs_slippage)),
            "median_slippage": float(np.median(abs_slippage)),
            "p95_slippage": float(np.percentile(abs_slippage, 95)),
            "p99_slippage": float(np.percentile(abs_slippage, 99)),
            "max_slippage": float(np.max(abs_slippage)),
            "avg_latency": float(np.mean(latency_array)),
            "median_latency": float(np.median(latency_array)),
            "p95_latency": float(np.percentile(latency_array, 95)),
            "total_executions": self.total_executions,
            "slippage_distribution": {
                "p10": float(np.percentile(abs_slippage, 10)),
                "p25": float(np.percentile(abs_slippage, 25)),
                "p50": float(np.percentile(abs_slippage, 50)),
                "p75": float(np.percentile(abs_slippage, 75)),
                "p90": float(np.percentile(abs_slippage, 90)),
                "p95": float(np.percentile(abs_slippage, 95)),
                "p99": float(np.percentile(abs_slippage, 99))
            }
        }
    
    def get_recent_statistics(self, n: int = 100) -> Dict:
        """
        Get statistics for recent N executions.
        
        Args:
            n: Number of recent executions to analyze
        
        Returns:
            Dictionary with statistics for recent executions
        """
        if len(self.executions) == 0:
            return self.get_statistics()
        
        recent_executions = list(self.executions)[-n:]
        recent_slippage = [abs(e.slippage) for e in recent_executions]
        recent_latency = [e.latency_seconds for e in recent_executions]
        
        if len(recent_slippage) == 0:
            return self.get_statistics()
        
        return {
            "avg_slippage": float(np.mean(recent_slippage)),
            "median_slippage": float(np.median(recent_slippage)),
            "p95_slippage": float(np.percentile(recent_slippage, 95)),
            "avg_latency": float(np.mean(recent_latency)),
            "total_executions": len(recent_executions),
            "period": f"Last {len(recent_executions)} executions"
        }
    
    def reset(self):
        """Reset all tracking data"""
        self.executions.clear()
        self.slippage_history.clear()
        self.latency_history.clear()
        self.total_executions = 0
        self.total_slippage = 0.0
        self.total_latency = 0.0

