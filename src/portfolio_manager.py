"""
Portfolio Manager for Multi-Instrument Trading

Manages portfolio-level risk, position sizing, and correlation across instruments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class InstrumentPosition:
    """Position information for a single instrument"""
    instrument: str
    position: float  # Position size (-1.0 to 1.0)
    entry_price: Optional[float]
    current_price: float
    unrealized_pnl: float
    notional_value: float  # Position value in dollars


class PortfolioManager:
    """
    Manages multi-instrument portfolio:
    - Position sizing across instruments
    - Correlation management
    - Portfolio-level risk limits
    - Diversification optimization
    """
    
    def __init__(
        self,
        instruments: List[str],
        initial_capital: float = 100000.0,
        max_portfolio_risk: float = 0.20,  # 20% max portfolio risk
        correlation_window: int = 60,  # Days for correlation calculation
        diversification_target: float = 0.3  # Target correlation between instruments
    ):
        """
        Initialize portfolio manager.
        
        Args:
            instruments: List of instrument symbols (e.g., ["ES", "NQ", "RTY"])
            initial_capital: Initial capital
            max_portfolio_risk: Maximum portfolio-level risk (as fraction of capital)
            correlation_window: Window for correlation calculation
            diversification_target: Target correlation (lower = more diversification)
        """
        self.instruments = instruments
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_portfolio_risk = max_portfolio_risk
        self.correlation_window = correlation_window
        self.diversification_target = diversification_target
        
        # Position tracking
        self.positions: Dict[str, InstrumentPosition] = {}
        
        # Correlation and volatility tracking
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.volatilities: Dict[str, float] = {}
        self.returns_history: Dict[str, List[float]] = {inst: [] for inst in instruments}
        
        # Initialize positions
        for inst in instruments:
            self.positions[inst] = InstrumentPosition(
                instrument=inst,
                position=0.0,
                entry_price=None,
                current_price=0.0,
                unrealized_pnl=0.0,
                notional_value=0.0
            )
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update current prices for all instruments.
        
        Args:
            prices: Dictionary mapping instrument to current price
        """
        for inst, price in prices.items():
            if inst in self.positions:
                self.positions[inst].current_price = price
                self._update_unrealized_pnl(inst)
    
    def _update_unrealized_pnl(self, instrument: str) -> None:
        """Update unrealized PnL for an instrument"""
        pos = self.positions[instrument]
        if pos.entry_price is not None and pos.position != 0:
            price_change = (pos.current_price - pos.entry_price) / pos.entry_price
            pos.unrealized_pnl = pos.notional_value * price_change * np.sign(pos.position)
    
    def update_returns(self, returns: Dict[str, float]) -> None:
        """
        Update returns history for correlation calculation.
        
        Args:
            returns: Dictionary mapping instrument to return
        """
        for inst, ret in returns.items():
            if inst in self.returns_history:
                self.returns_history[inst].append(ret)
                # Keep only recent history
                if len(self.returns_history[inst]) > self.correlation_window:
                    self.returns_history[inst].pop(0)
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix from returns history.
        
        Returns:
            Correlation matrix DataFrame
        """
        # Build returns DataFrame
        returns_df = pd.DataFrame(self.returns_history)
        
        if len(returns_df) < 10:  # Need minimum data
            # Return identity matrix if not enough data
            return pd.DataFrame(
                np.eye(len(self.instruments)),
                index=self.instruments,
                columns=self.instruments
            )
        
        self.correlation_matrix = returns_df.corr()
        return self.correlation_matrix
    
    def calculate_portfolio_risk(
        self,
        positions: Optional[Dict[str, float]] = None,
        volatilities: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate portfolio-level risk using portfolio variance.
        
        Uses: portfolio_var = sqrt(w' * Cov * w)
        
        Args:
            positions: Optional position sizes (uses current if None)
            volatilities: Optional volatilities (uses tracked if None)
        
        Returns:
            Portfolio risk (standard deviation)
        """
        if positions is None:
            positions = {inst: self.positions[inst].position for inst in self.instruments}
        
        if volatilities is None:
            volatilities = self.volatilities
        
        # Get correlation matrix
        corr_matrix = self.calculate_correlation_matrix()
        
        # Build covariance matrix
        cov_matrix = self._build_covariance_matrix(corr_matrix, volatilities)
        
        # Calculate portfolio weights (normalized by capital)
        weights = np.array([positions.get(inst, 0.0) for inst in self.instruments])
        
        # Calculate portfolio variance
        if np.sum(np.abs(weights)) == 0:
            return 0.0
        
        portfolio_var = np.sqrt(weights.T @ cov_matrix @ weights)
        
        return float(portfolio_var)
    
    def _build_covariance_matrix(
        self,
        correlation_matrix: pd.DataFrame,
        volatilities: Dict[str, float]
    ) -> np.ndarray:
        """
        Build covariance matrix from correlation and volatilities.
        
        Cov(i,j) = corr(i,j) * vol(i) * vol(j)
        """
        n = len(self.instruments)
        cov_matrix = np.zeros((n, n))
        
        for i, inst_i in enumerate(self.instruments):
            vol_i = volatilities.get(inst_i, 0.01)  # Default 1% volatility
            for j, inst_j in enumerate(self.instruments):
                vol_j = volatilities.get(inst_j, 0.01)
                corr = correlation_matrix.loc[inst_i, inst_j]
                cov_matrix[i, j] = corr * vol_i * vol_j
        
        return cov_matrix
    
    def optimize_position_sizing(
        self,
        signals: Dict[str, float],
        risk_budget: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Optimize position sizes across instruments using risk parity.
        
        Risk parity: Equal risk contribution from each instrument.
        
        Args:
            signals: Dictionary mapping instrument to signal strength (-1.0 to 1.0)
            risk_budget: Optional risk budget (uses max_portfolio_risk if None)
        
        Returns:
            Dictionary mapping instrument to optimized position size
        """
        if risk_budget is None:
            risk_budget = self.max_portfolio_risk
        
        # Get correlation matrix
        corr_matrix = self.calculate_correlation_matrix()
        
        # Initialize volatilities if not set
        if not self.volatilities:
            for inst in self.instruments:
                self.volatilities[inst] = 0.02  # Default 2% daily volatility
        
        # Simple risk parity: equal risk contribution
        # For each instrument, target risk = risk_budget / num_instruments
        target_risk_per_instrument = risk_budget / len(self.instruments)
        
        optimized_positions = {}
        
        for inst in self.instruments:
            signal = signals.get(inst, 0.0)
            if abs(signal) < 0.05:  # Ignore weak signals
                optimized_positions[inst] = 0.0
                continue
            
            # Calculate position size to achieve target risk
            vol = self.volatilities.get(inst, 0.02)
            if vol > 0:
                # Position size = target_risk / volatility
                position_size = (target_risk_per_instrument / vol) * np.sign(signal)
                # Cap at max position size
                position_size = np.clip(position_size, -1.0, 1.0)
            else:
                position_size = 0.0
            
            optimized_positions[inst] = float(position_size)
        
        # Adjust for correlation (reduce positions if highly correlated)
        optimized_positions = self._adjust_for_correlation(
            optimized_positions,
            corr_matrix
        )
        
        return optimized_positions
    
    def _adjust_for_correlation(
        self,
        positions: Dict[str, float],
        corr_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Adjust positions to account for correlation.
        
        Reduces position sizes if instruments are highly correlated.
        """
        adjusted = positions.copy()
        
        for i, inst_i in enumerate(self.instruments):
            if abs(positions[inst_i]) < 0.01:
                continue
            
            for j, inst_j in enumerate(self.instruments):
                if i >= j or abs(positions[inst_j]) < 0.01:
                    continue
                
                # If both positions are in same direction and highly correlated
                corr = corr_matrix.loc[inst_i, inst_j]
                if corr > 0.7 and np.sign(positions[inst_i]) == np.sign(positions[inst_j]):
                    # Reduce both positions proportionally
                    reduction_factor = 1.0 - (corr - 0.7) * 0.5  # Reduce by up to 15%
                    adjusted[inst_i] *= reduction_factor
                    adjusted[inst_j] *= reduction_factor
        
        return adjusted
    
    def check_portfolio_risk_limits(self) -> Tuple[bool, str]:
        """
        Check if portfolio risk is within limits.
        
        Returns:
            (is_within_limits, message)
        """
        portfolio_risk = self.calculate_portfolio_risk()
        
        if portfolio_risk > self.max_portfolio_risk:
            return False, f"Portfolio risk {portfolio_risk:.2%} exceeds limit {self.max_portfolio_risk:.2%}"
        
        return True, f"Portfolio risk {portfolio_risk:.2%} within limits"
    
    def get_portfolio_summary(self) -> Dict:
        """
        Get portfolio summary statistics.
        
        Returns:
            Dictionary with portfolio metrics
        """
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_notional = sum(abs(pos.notional_value) for pos in self.positions.values())
        
        portfolio_risk = self.calculate_portfolio_risk()
        
        # Calculate diversification score (lower correlation = better)
        corr_matrix = self.calculate_correlation_matrix()
        avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        diversification_score = 1.0 - abs(avg_correlation)  # Higher = more diversified
        
        return {
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_notional_value": total_notional,
            "portfolio_risk": portfolio_risk,
            "portfolio_risk_pct": portfolio_risk / self.current_capital if self.current_capital > 0 else 0.0,
            "diversification_score": diversification_score,
            "avg_correlation": avg_correlation,
            "num_positions": sum(1 for pos in self.positions.values() if abs(pos.position) > 0.01),
            "positions": {
                inst: {
                    "position": pos.position,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "notional_value": pos.notional_value
                }
                for inst, pos in self.positions.items()
            }
        }

