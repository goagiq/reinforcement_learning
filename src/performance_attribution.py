"""
Performance Attribution Analysis

Attributes performance to different factors:
- Market timing (entry/exit skill)
- Position sizing (size optimization)
- Instrument selection (if multi-instrument)
- Time-of-day effects
- Market regime effects
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, time


@dataclass
class Trade:
    """Trade representation for attribution"""
    instrument: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_size: float  # Position size (-1.0 to 1.0)
    pnl: float
    pnl_per_unit: float  # PnL per unit of position
    time_of_day: str  # "morning", "midday", "afternoon", "close"
    market_regime: Optional[str] = None  # "trending", "ranging", "volatile"


class PerformanceAttribution:
    """
    Attributes performance to factors:
    - Market timing (entry/exit skill)
    - Position sizing (size optimization)
    - Instrument selection (if multi-instrument)
    - Time-of-day effects
    - Market regime effects
    """
    
    def __init__(self):
        """Initialize performance attribution analyzer"""
        self.trades: List[Trade] = []
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
    
    def add_trade(self, trade: Trade) -> None:
        """Add a trade for attribution analysis"""
        self.trades.append(trade)
    
    def attribute_returns(
        self,
        trades: Optional[List[Trade]] = None,
        market_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict:
        """
        Attribute returns to different factors.
        
        Args:
            trades: List of trades (uses self.trades if None)
            market_data: Market data for optimal timing calculation
        
        Returns:
            Dictionary with attribution breakdown
        """
        if trades is None:
            trades = self.trades
        
        if len(trades) == 0:
            return {
                "total_pnl": 0.0,
                "market_timing": 0.0,
                "position_sizing": 0.0,
                "instrument_selection": 0.0,
                "time_of_day": {},
                "regime": {},
                "unexplained": 0.0
            }
        
        attribution = {
            "total_pnl": sum(t.pnl for t in trades),
            "market_timing": 0.0,
            "position_sizing": 0.0,
            "instrument_selection": 0.0,
            "time_of_day": {},
            "regime": {},
            "unexplained": 0.0
        }
        
        # Market timing attribution
        timing_contributions = []
        sizing_contributions = []
        
        for trade in trades:
            # Market timing: compare entry/exit to optimal
            if market_data and trade.instrument in market_data:
                optimal_entry, optimal_exit = self._find_optimal_prices(
                    trade, market_data[trade.instrument]
                )
                
                # Timing contribution = (actual_entry - optimal_entry) * size
                entry_timing = (trade.entry_price - optimal_entry) * trade.position_size
                exit_timing = (optimal_exit - trade.exit_price) * trade.position_size
                timing_contribution = entry_timing + exit_timing
                timing_contributions.append(timing_contribution)
            else:
                # Use PnL as proxy for timing if no market data
                timing_contributions.append(trade.pnl * 0.6)  # Assume 60% from timing
            
            # Position sizing: compare actual size to optimal
            optimal_size = self._calculate_optimal_size(trade)
            sizing_contribution = (trade.position_size - optimal_size) * trade.pnl_per_unit
            sizing_contributions.append(sizing_contribution)
        
        attribution["market_timing"] = sum(timing_contributions)
        attribution["position_sizing"] = sum(sizing_contributions)
        
        # Instrument selection (if multi-instrument)
        if len(set(t.instrument for t in trades)) > 1:
            instrument_pnl = {}
            for trade in trades:
                if trade.instrument not in instrument_pnl:
                    instrument_pnl[trade.instrument] = 0.0
                instrument_pnl[trade.instrument] += trade.pnl
            
            # Calculate contribution from instrument selection
            avg_pnl_per_trade = attribution["total_pnl"] / len(trades)
            instrument_selection_contribution = 0.0
            for inst, pnl in instrument_pnl.items():
                inst_trades = [t for t in trades if t.instrument == inst]
                expected_pnl = avg_pnl_per_trade * len(inst_trades)
                instrument_selection_contribution += (pnl - expected_pnl)
            
            attribution["instrument_selection"] = instrument_selection_contribution
        
        # Time-of-day effects
        time_of_day_pnl = {}
        for trade in trades:
            if trade.time_of_day not in time_of_day_pnl:
                time_of_day_pnl[trade.time_of_day] = {"pnl": 0.0, "count": 0}
            time_of_day_pnl[trade.time_of_day]["pnl"] += trade.pnl
            time_of_day_pnl[trade.time_of_day]["count"] += 1
        
        for tod, data in time_of_day_pnl.items():
            attribution["time_of_day"][tod] = {
                "total_pnl": data["pnl"],
                "avg_pnl": data["pnl"] / data["count"] if data["count"] > 0 else 0.0,
                "trade_count": data["count"]
            }
        
        # Regime effects
        if any(t.market_regime for t in trades):
            regime_pnl = {}
            for trade in trades:
                if trade.market_regime:
                    if trade.market_regime not in regime_pnl:
                        regime_pnl[trade.market_regime] = {"pnl": 0.0, "count": 0}
                    regime_pnl[trade.market_regime]["pnl"] += trade.pnl
                    regime_pnl[trade.market_regime]["count"] += 1
            
            for regime, data in regime_pnl.items():
                attribution["regime"][regime] = {
                    "total_pnl": data["pnl"],
                    "avg_pnl": data["pnl"] / data["count"] if data["count"] > 0 else 0.0,
                    "trade_count": data["count"]
                }
        
        # Unexplained (residual)
        explained = (
            attribution["market_timing"] +
            attribution["position_sizing"] +
            attribution["instrument_selection"]
        )
        attribution["unexplained"] = attribution["total_pnl"] - explained
        
        return attribution
    
    def _find_optimal_prices(
        self,
        trade: Trade,
        market_data: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        Find optimal entry and exit prices for a trade.
        
        Optimal entry: Best price during entry window
        Optimal exit: Best price during exit window
        """
        # Find entry window (e.g., ±5 bars around entry)
        entry_idx = self._find_time_index(market_data, trade.entry_time)
        exit_idx = self._find_time_index(market_data, trade.exit_time)
        
        if entry_idx is None or exit_idx is None:
            return trade.entry_price, trade.exit_price
        
        # Optimal entry: lowest price for long, highest for short
        entry_window = market_data.iloc[max(0, entry_idx-5):entry_idx+5]
        if len(entry_window) == 0:
            optimal_entry = trade.entry_price
        else:
            if trade.position_size > 0:  # Long
                optimal_entry = entry_window["low"].min()
            else:  # Short
                optimal_entry = entry_window["high"].max()
        
        # Optimal exit: highest price for long, lowest for short
        exit_window = market_data.iloc[max(0, exit_idx-5):exit_idx+5]
        if len(exit_window) == 0:
            optimal_exit = trade.exit_price
        else:
            if trade.position_size > 0:  # Long
                optimal_exit = exit_window["high"].max()
            else:  # Short
                optimal_exit = exit_window["low"].min()
        
        return optimal_entry, optimal_exit
    
    def _find_time_index(self, data: pd.DataFrame, target_time: datetime) -> Optional[int]:
        """Find index in DataFrame closest to target time"""
        if "timestamp" in data.columns:
            time_col = "timestamp"
        elif data.index.name == "timestamp" or isinstance(data.index, pd.DatetimeIndex):
            # Use index
            time_diffs = (data.index - target_time).abs()
            return time_diffs.idxmin() if len(time_diffs) > 0 else None
        else:
            return None
        
        if time_col in data.columns:
            time_diffs = (pd.to_datetime(data[time_col]) - target_time).abs()
            return time_diffs.idxmin() if len(time_diffs) > 0 else None
        
        return None
    
    def _calculate_optimal_size(self, trade: Trade) -> float:
        """
        Calculate optimal position size for a trade.
        
        Uses Kelly Criterion or fixed fractional sizing.
        """
        # Simple approach: use fixed fractional (e.g., 0.5 = 50% of capital)
        # More sophisticated: Kelly Criterion based on win rate and R:R
        
        # For now, use a simple rule: optimal size = 0.5 (50% of max position)
        # This can be enhanced with actual risk metrics
        return 0.5 * np.sign(trade.position_size)
    
    def get_time_of_day_classification(self, trade_time: datetime) -> str:
        """Classify trade time into time-of-day categories"""
        hour = trade_time.hour
        
        if 9 <= hour < 11:
            return "morning"
        elif 11 <= hour < 14:
            return "midday"
        elif 14 <= hour < 15:
            return "afternoon"
        else:
            return "close"
    
    def get_summary_report(self) -> Dict:
        """
        Get comprehensive attribution summary report.
        
        Returns:
            Dictionary with detailed attribution breakdown
        """
        if len(self.trades) == 0:
            return {"message": "No trades available for attribution"}
        
        attribution = self.attribute_returns()
        
        # Calculate percentages
        total_pnl = attribution["total_pnl"]
        if abs(total_pnl) > 1e-6:
            timing_pct = (attribution["market_timing"] / total_pnl) * 100
            sizing_pct = (attribution["position_sizing"] / total_pnl) * 100
            instrument_pct = (attribution["instrument_selection"] / total_pnl) * 100
            unexplained_pct = (attribution["unexplained"] / total_pnl) * 100
        else:
            timing_pct = sizing_pct = instrument_pct = unexplained_pct = 0.0
        
        return {
            "total_pnl": total_pnl,
            "total_trades": len(self.trades),
            "attribution_breakdown": {
                "market_timing": {
                    "contribution": attribution["market_timing"],
                    "percentage": timing_pct
                },
                "position_sizing": {
                    "contribution": attribution["position_sizing"],
                    "percentage": sizing_pct
                },
                "instrument_selection": {
                    "contribution": attribution["instrument_selection"],
                    "percentage": instrument_pct
                },
                "unexplained": {
                    "contribution": attribution["unexplained"],
                    "percentage": unexplained_pct
                }
            },
            "time_of_day_effects": attribution["time_of_day"],
            "regime_effects": attribution["regime"]
        }

