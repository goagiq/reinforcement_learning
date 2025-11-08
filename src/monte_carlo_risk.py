"""
Monte Carlo Risk Assessment Module

Provides pre-trade and post-trade risk analysis using Monte Carlo simulation.
Estimates profit/loss distributions, Value at Risk (VaR), and tail risk scenarios.

Veteran Futures Trader Approach:
- Simulates multiple price scenarios based on historical volatility
- Estimates worst-case scenarios (tail risk)
- Provides adaptive position sizing recommendations
- Assesses overnight gap risk (critical for futures)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RiskMetrics:
    """Risk metrics from Monte Carlo simulation"""
    expected_return: float
    expected_pnl: float
    var_95: float  # Value at Risk at 95% confidence
    var_99: float  # Value at Risk at 99% confidence
    cvar_95: float  # Conditional VaR (Expected Shortfall) at 95%
    max_drawdown: float
    win_probability: float
    tail_risk: float  # Probability of extreme loss (>5% of capital)
    optimal_position_size: float  # Recommended position size based on risk


@dataclass
class MonteCarloResult:
    """Complete Monte Carlo simulation result"""
    scenario_pnls: np.ndarray
    scenario_returns: np.ndarray
    max_drawdowns: np.ndarray
    risk_metrics: RiskMetrics
    simulation_config: Dict
    timestamp: datetime


class MonteCarloRiskAnalyzer:
    """
    Monte Carlo Risk Analyzer for futures trading.
    
    Simulates multiple price scenarios to assess trade risk before execution.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        n_simulations: int = 1000,
        lookback_periods: int = 252,  # Trading days for volatility estimation
        confidence_levels: List[float] = [0.95, 0.99],
        max_position_risk: float = 0.02  # Max 2% of capital at risk per trade
    ):
        """
        Initialize Monte Carlo risk analyzer.
        
        Args:
            initial_capital: Starting capital
            n_simulations: Number of Monte Carlo simulations
            lookback_periods: Periods to use for volatility estimation
            confidence_levels: VaR confidence levels
            max_position_risk: Maximum capital at risk per trade (as fraction)
        """
        self.initial_capital = initial_capital
        self.n_simulations = n_simulations
        self.lookback_periods = lookback_periods
        self.confidence_levels = confidence_levels
        self.max_position_risk = max_position_risk
    
    def estimate_volatility(
        self,
        price_data: pd.DataFrame,
        method: str = "historical"
    ) -> Tuple[float, float]:
        """
        Estimate volatility from historical data.
        
        Args:
            price_data: Historical price data (must have 'close' column)
            method: 'historical' or 'garch' (future enhancement)
        
        Returns:
            (daily_volatility, annualized_volatility)
        """
        if len(price_data) < 2:
            return 0.02, 0.32  # Default 2% daily, 32% annual
        
        # Calculate returns
        returns = price_data['close'].pct_change().dropna()
        
        if len(returns) < 10:
            return 0.02, 0.32
        
        # Use recent data for volatility estimation (more relevant)
        recent_returns = returns.tail(min(self.lookback_periods, len(returns)))
        
        # Daily volatility (standard deviation of returns)
        daily_vol = recent_returns.std()
        
        # Annualized volatility (assuming 252 trading days)
        annual_vol = daily_vol * np.sqrt(252)
        
        # Clamp to reasonable values
        daily_vol = np.clip(daily_vol, 0.001, 0.10)  # 0.1% to 10% daily
        annual_vol = np.clip(annual_vol, 0.05, 2.0)  # 5% to 200% annual
        
        return float(daily_vol), float(annual_vol)
    
    def simulate_price_paths(
        self,
        current_price: float,
        volatility: float,
        time_horizon: int = 1,
        drift: float = 0.0,
        method: str = "geometric_brownian_motion"
    ) -> np.ndarray:
        """
        Simulate future price paths using Monte Carlo.
        
        Args:
            current_price: Current market price
            volatility: Daily volatility
            time_horizon: Number of periods ahead to simulate
            drift: Expected return per period (default: 0 for risk-neutral)
            method: Simulation method
        
        Returns:
            Array of simulated prices [n_simulations, time_horizon]
        """
        dt = 1.0  # One period (can be adjusted for different timeframes)
        
        if method == "geometric_brownian_motion":
            # Geometric Brownian Motion: dS = S * (mu*dt + sigma*dW)
            # For risk assessment, we use drift=0 (risk-neutral) or market drift
            
            # Generate random shocks
            random_shocks = np.random.normal(
                0, 1, (self.n_simulations, time_horizon)
            )
            
            # Calculate price paths
            price_paths = np.zeros((self.n_simulations, time_horizon))
            price_paths[:, 0] = current_price
            
            for t in range(1, time_horizon):
                # GBM: S_t = S_{t-1} * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
                price_paths[:, t] = price_paths[:, t-1] * np.exp(
                    (drift - 0.5 * volatility**2) * dt + 
                    volatility * np.sqrt(dt) * random_shocks[:, t]
                )
        
        else:
            # Simple random walk (fallback)
            random_shocks = np.random.normal(
                drift * dt, volatility * np.sqrt(dt),
                (self.n_simulations, time_horizon)
            )
            price_paths = current_price + np.cumsum(random_shocks, axis=1)
            price_paths = np.maximum(price_paths, current_price * 0.5)  # Floor at 50% of current price
        
        return price_paths
    
    def assess_trade_risk(
        self,
        current_price: float,
        proposed_position: float,  # Position size (-1.0 to 1.0)
        current_position: float,  # Current position (-1.0 to 1.0)
        price_data: pd.DataFrame,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        time_horizon: int = 1,
        simulate_overnight: bool = True
    ) -> MonteCarloResult:
        """
        Assess risk of a proposed trade using Monte Carlo simulation.
        
        Args:
            current_price: Current market price
            proposed_position: Proposed position size (-1.0 to 1.0)
            current_position: Current position size (-1.0 to 1.0)
            price_data: Historical price data for volatility estimation
            entry_price: Entry price (if different from current_price)
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            time_horizon: Number of periods to simulate ahead
            simulate_overnight: If True, simulate overnight gap scenarios
        
        Returns:
            MonteCarloResult with risk metrics and recommendations
        """
        if entry_price is None:
            entry_price = current_price
        
        # Estimate volatility
        daily_vol, annual_vol = self.estimate_volatility(price_data)
        
        # Calculate position change
        position_change = proposed_position - current_position
        net_position = proposed_position
        
        # Simulate price paths
        price_paths = self.simulate_price_paths(
            current_price=current_price,
            volatility=daily_vol,
            time_horizon=time_horizon,
            drift=0.0,  # Risk-neutral for conservative risk assessment
            method="geometric_brownian_motion"
        )
        
        # Calculate PnL for each scenario
        scenario_pnls = []
        scenario_returns = []
        max_drawdowns = []
        
        for i in range(self.n_simulations):
            scenario_pnl = 0.0
            scenario_equity = self.initial_capital
            max_equity = self.initial_capital
            max_dd = 0.0
            
            # Calculate PnL at each time step
            for t in range(time_horizon):
                future_price = price_paths[i, t]
                
                # Check stop loss / take profit
                hit_stop = False
                hit_target = False
                
                if stop_loss is not None and net_position > 0:
                    if future_price <= stop_loss:
                        future_price = stop_loss
                        hit_stop = True
                elif stop_loss is not None and net_position < 0:
                    if future_price >= stop_loss:
                        future_price = stop_loss
                        hit_stop = True
                
                if take_profit is not None and net_position > 0:
                    if future_price >= take_profit:
                        future_price = take_profit
                        hit_target = True
                elif take_profit is not None and net_position < 0:
                    if future_price <= take_profit:
                        future_price = take_profit
                        hit_target = True
                
                # Calculate PnL
                if net_position != 0:
                    # Position value change
                    price_change_pct = (future_price - entry_price) / entry_price
                    position_pnl = price_change_pct * net_position * self.initial_capital
                    scenario_pnl = position_pnl
                    
                    # Update equity
                    scenario_equity = self.initial_capital + scenario_pnl
                    
                    # Track drawdown
                    if scenario_equity > max_equity:
                        max_equity = scenario_equity
                    dd = (max_equity - scenario_equity) / max_equity if max_equity > 0 else 0.0
                    max_dd = max(max_dd, dd)
                    
                    # Exit if stop/target hit
                    if hit_stop or hit_target:
                        break
            
            scenario_pnls.append(scenario_pnl)
            scenario_returns.append(scenario_pnl / self.initial_capital)
            max_drawdowns.append(max_dd)
        
        scenario_pnls = np.array(scenario_pnls)
        scenario_returns = np.array(scenario_returns)
        max_drawdowns = np.array(max_drawdowns)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(
            scenario_pnls=scenario_pnls,
            scenario_returns=scenario_returns,
            max_drawdowns=max_drawdowns,
            position_size=abs(net_position),
            daily_volatility=daily_vol
        )
        
        # Add overnight gap simulation if requested
        if simulate_overnight:
            overnight_risk = self._assess_overnight_gap_risk(
                current_price=current_price,
                position_size=abs(net_position),
                price_data=price_data
            )
            # Adjust risk metrics based on overnight risk
            risk_metrics.tail_risk = max(risk_metrics.tail_risk, overnight_risk)
        
        return MonteCarloResult(
            scenario_pnls=scenario_pnls,
            scenario_returns=scenario_returns,
            max_drawdowns=max_drawdowns,
            risk_metrics=risk_metrics,
            simulation_config={
                "n_simulations": self.n_simulations,
                "volatility": daily_vol,
                "time_horizon": time_horizon,
                "current_price": current_price,
                "position_size": net_position
            },
            timestamp=datetime.now()
        )
    
    def _calculate_risk_metrics(
        self,
        scenario_pnls: np.ndarray,
        scenario_returns: np.ndarray,
        max_drawdowns: np.ndarray,
        position_size: float,
        daily_volatility: float
    ) -> RiskMetrics:
        """Calculate risk metrics from simulation results"""
        
        # Expected values
        expected_pnl = float(np.mean(scenario_pnls))
        expected_return = float(np.mean(scenario_returns))
        
        # Value at Risk (VaR) - loss that won't be exceeded with given confidence
        var_95 = float(np.percentile(scenario_pnls, 5))  # 95% VaR (5th percentile)
        var_99 = float(np.percentile(scenario_pnls, 1))  # 99% VaR (1st percentile)
        
        # Conditional VaR (Expected Shortfall) - average loss when VaR is exceeded
        cvar_95 = float(np.mean(scenario_pnls[scenario_pnls <= var_95]))
        
        # Maximum drawdown
        max_drawdown = float(np.max(max_drawdowns))
        
        # Win probability
        win_probability = float(np.mean(scenario_pnls > 0))
        
        # Tail risk - probability of extreme loss (>5% of capital)
        extreme_loss_threshold = -self.initial_capital * 0.05
        tail_risk = float(np.mean(scenario_pnls < extreme_loss_threshold))
        
        # Optimal position size based on risk
        # Kelly Criterion-inspired: maximize expected return while limiting risk
        max_risk_per_trade = self.initial_capital * self.max_position_risk
        
        # Calculate what position size would keep VaR_95 within limits
        if abs(var_95) > 0:
            risk_scaled_position = (max_risk_per_trade / abs(var_95)) * position_size
            optimal_position_size = float(np.clip(risk_scaled_position, 0.0, 1.0))
        else:
            optimal_position_size = position_size
        
        # If tail risk is too high, reduce position size
        if tail_risk > 0.05:  # More than 5% chance of extreme loss
            optimal_position_size *= 0.5  # Reduce by half
        
        return RiskMetrics(
            expected_return=expected_return,
            expected_pnl=expected_pnl,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            win_probability=win_probability,
            tail_risk=tail_risk,
            optimal_position_size=optimal_position_size
        )
    
    def _assess_overnight_gap_risk(
        self,
        current_price: float,
        position_size: float,
        price_data: pd.DataFrame
    ) -> float:
        """
        Assess overnight gap risk (critical for futures trading).
        
        Returns probability of significant overnight gap.
        """
        if len(price_data) < 2:
            return 0.05  # Default 5% gap risk
        
        # Calculate historical overnight gaps
        # For simplicity, we'll use daily returns as proxy
        returns = price_data['close'].pct_change().dropna()
        
        if len(returns) < 10:
            return 0.05
        
        # Calculate volatility of gaps (use absolute returns)
        gap_volatility = returns.abs().tail(60).std()  # Last 60 days
        
        # Estimate probability of gap > 2% (significant gap)
        gap_threshold = 0.02
        z_score = gap_threshold / gap_volatility if gap_volatility > 0 else 2.0
        
        # Probability of gap > threshold (one-tailed)
        try:
            from scipy import stats
            gap_probability = 1 - stats.norm.cdf(z_score)
        except ImportError:
            # Fallback if scipy not available - use approximation
            # Normal CDF approximation using error function
            import math
            gap_probability = 0.5 * (1 - math.erf(z_score / math.sqrt(2)))
            if gap_probability < 0:
                gap_probability = 0.05  # Default minimum
        except:
            gap_probability = 0.05  # Default
        
        # Scale by position size (larger positions = higher risk)
        risk = gap_probability * abs(position_size)
        
        return float(np.clip(risk, 0.0, 0.20))  # Cap at 20%
    
    def scenario_analysis(
        self,
        current_price: float,
        position_size: float,
        price_data: pd.DataFrame,
        scenarios: List[str] = ["normal", "high_volatility", "trending", "ranging"]
    ) -> Dict[str, MonteCarloResult]:
        """
        Analyze trade under different market scenarios.
        
        Args:
            current_price: Current market price
            position_size: Position size to analyze
            price_data: Historical price data
            scenarios: List of scenarios to simulate
        
        Returns:
            Dictionary of scenario results
        """
        results = {}
        
        daily_vol, annual_vol = self.estimate_volatility(price_data)
        
        for scenario in scenarios:
            if scenario == "normal":
                # Normal volatility
                vol_multiplier = 1.0
                drift = 0.0
            elif scenario == "high_volatility":
                # 2x volatility (stress test)
                vol_multiplier = 2.0
                drift = 0.0
            elif scenario == "trending":
                # Trending market (positive drift)
                vol_multiplier = 1.0
                drift = daily_vol * 0.5  # Moderate upward drift
            elif scenario == "ranging":
                # Ranging market (mean-reverting)
                vol_multiplier = 0.8
                drift = -daily_vol * 0.2  # Slight mean reversion
            else:
                continue
            
            # Simulate with scenario parameters
            price_paths = self.simulate_price_paths(
                current_price=current_price,
                volatility=daily_vol * vol_multiplier,
                time_horizon=5,  # 5 periods ahead
                drift=drift
            )
            
            # Calculate PnL
            scenario_pnls = []
            for i in range(self.n_simulations):
                future_price = price_paths[i, -1]  # Price at end of horizon
                price_change_pct = (future_price - current_price) / current_price
                pnl = price_change_pct * position_size * self.initial_capital
                scenario_pnls.append(pnl)
            
            scenario_pnls = np.array(scenario_pnls)
            scenario_returns = scenario_pnls / self.initial_capital
            max_drawdowns = np.array([0.0] * self.n_simulations)  # Simplified
            
            risk_metrics = self._calculate_risk_metrics(
                scenario_pnls=scenario_pnls,
                scenario_returns=scenario_returns,
                max_drawdowns=max_drawdowns,
                position_size=abs(position_size),
                daily_volatility=daily_vol * vol_multiplier
            )
            
            results[scenario] = MonteCarloResult(
                scenario_pnls=scenario_pnls,
                scenario_returns=scenario_returns,
                max_drawdowns=max_drawdowns,
                risk_metrics=risk_metrics,
                simulation_config={
                    "scenario": scenario,
                    "volatility_multiplier": vol_multiplier,
                    "drift": drift
                },
                timestamp=datetime.now()
            )
        
        return results


def assess_position_risk(
    current_price: float,
    position_size: float,
    price_data: pd.DataFrame,
    initial_capital: float = 100000.0,
    n_simulations: int = 1000
) -> Dict:
    """
    Convenience function for quick risk assessment.
    
    Args:
        current_price: Current market price
        position_size: Current position size (-1.0 to 1.0)
        price_data: Historical price data
        initial_capital: Starting capital
        n_simulations: Number of simulations
    
    Returns:
        Dictionary with risk metrics
    """
    analyzer = MonteCarloRiskAnalyzer(
        initial_capital=initial_capital,
        n_simulations=n_simulations
    )
    
    result = analyzer.assess_trade_risk(
        current_price=current_price,
        proposed_position=position_size,
        current_position=0.0,  # Assume starting from flat
        price_data=price_data,
        time_horizon=1
    )
    
    return {
        "expected_pnl": result.risk_metrics.expected_pnl,
        "var_95": result.risk_metrics.var_95,
        "var_99": result.risk_metrics.var_99,
        "win_probability": result.risk_metrics.win_probability,
        "tail_risk": result.risk_metrics.tail_risk,
        "optimal_position_size": result.risk_metrics.optimal_position_size,
        "max_drawdown": result.risk_metrics.max_drawdown
    }

