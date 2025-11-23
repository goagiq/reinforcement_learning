"""
Risk Management Module

Implements risk controls and position sizing logic.
Integrates with Monte Carlo risk assessment for advanced risk analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.monte_carlo_risk import MonteCarloRiskAnalyzer, MonteCarloResult
    MONTE_CARLO_AVAILABLE = True
except ImportError:
    MONTE_CARLO_AVAILABLE = False
    MonteCarloRiskAnalyzer = None
    MonteCarloResult = None

try:
    from src.volatility_predictor import VolatilityPredictor, VolatilityForecast
    VOLATILITY_PREDICTOR_AVAILABLE = True
except ImportError:
    VOLATILITY_PREDICTOR_AVAILABLE = False
    VolatilityPredictor = None
    VolatilityForecast = None


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_position_size: float
    max_drawdown: float
    max_daily_loss: float
    max_position_per_instrument: float
    stop_loss_atr_multiplier: float
    max_leverage: float = 1.0


class RiskManager:
    """
    Risk management system for trading.
    
    Enforces:
    - Position size limits
    - Drawdown limits
    - Daily loss limits
    - Stop loss levels
    - Maximum leverage
    """
    
    def __init__(self, risk_config: Dict):
        """
        Initialize risk manager.
        
        Args:
            risk_config: Risk management configuration from config file
        """
        self.limits = RiskLimits(
            max_position_size=risk_config.get("max_position_size", 1.0),
            max_drawdown=risk_config.get("max_drawdown", 0.20),
            max_daily_loss=risk_config.get("max_daily_loss", 0.05),
            max_position_per_instrument=risk_config.get("max_position_size", 1.0),
            stop_loss_atr_multiplier=risk_config.get("stop_loss_atr_multiplier", 2.0),
            max_leverage=risk_config.get("max_leverage", 1.0)
        )
        self.initial_capital = risk_config.get("initial_capital", 100000.0)

        self.dynamic_position_fraction = risk_config.get("max_position_fraction_of_balance", 0.02)
        self.position_value_per_unit = risk_config.get("position_value_per_unit", self.initial_capital)
        break_even_cfg = risk_config.get("break_even", {})
        self.break_even_cfg = {
            "enabled": break_even_cfg.get("enabled", True),
            "activation_pct": break_even_cfg.get("activation_pct", 0.003),
            "trail_pct": break_even_cfg.get("trail_pct", 0.0015),
            "scale_out_fraction": break_even_cfg.get("scale_out_fraction", 0.5),
            "scale_out_min_confluence": break_even_cfg.get("scale_out_min_confluence", 1),
            "free_trade_fraction": break_even_cfg.get("free_trade_fraction", 0.5),
        }
        
        # State tracking
        self.current_capital = self.initial_capital
        self.max_capital = self.initial_capital
        self.current_drawdown = 0.0
        
        # Daily tracking
        self.daily_start_capital = self.initial_capital
        self.daily_loss = 0.0
        self.last_reset_date = datetime.now().date()
        
        # Position tracking
        self.current_positions = {}  # instrument -> position_size
        self.total_exposure = 0.0
        self.position_states: Dict[str, Dict[str, float]] = {}
        
        # Monte Carlo risk analyzer (optional)
        self.monte_carlo_enabled = risk_config.get("monte_carlo_enabled", True)
        if self.monte_carlo_enabled and MONTE_CARLO_AVAILABLE:
            self.monte_carlo = MonteCarloRiskAnalyzer(
                initial_capital=self.initial_capital,
                n_simulations=risk_config.get("monte_carlo_simulations", 1000),
                max_position_risk=risk_config.get("max_position_risk", 0.02)
            )
        else:
            self.monte_carlo = None
        
        # Volatility predictor (optional)
        self.volatility_enabled = risk_config.get("volatility_prediction_enabled", True)
        if self.volatility_enabled and VOLATILITY_PREDICTOR_AVAILABLE:
            self.volatility_predictor = VolatilityPredictor(
                lookback_periods=risk_config.get("volatility_lookback", 252),
                prediction_horizon=risk_config.get("volatility_horizon", 1),
                volatility_window=risk_config.get("volatility_window", 20)
            )
        else:
            self.volatility_predictor = None
    
    def reset_daily(self):
        """Reset daily limits (call at start of each trading day)"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_start_capital = self.current_capital
            self.daily_loss = 0.0
            self.last_reset_date = today
    
    def update_capital(self, pnl: float):
        """
        Update current capital and check limits.
        
        Args:
            pnl: Profit/loss since last update
        """
        self.reset_daily()
        
        self.current_capital += pnl
        
        # Update drawdown
        if self.current_capital > self.max_capital:
            self.max_capital = self.current_capital
        
        self.current_drawdown = (self.max_capital - self.current_capital) / self.max_capital
        
        # Update daily loss
        self.daily_loss = (self.daily_start_capital - self.current_capital) / self.daily_start_capital
    
    def validate_action(
        self,
        target_position: float,
        current_position: float,
        market_data: Optional[Dict] = None,
        price_data: Optional[pd.DataFrame] = None,
        current_price: Optional[float] = None,
        use_monte_carlo: bool = True,
        decision_context: Optional[Dict] = None,
        instrument: str = "default"
    ) -> Tuple[float, Optional[MonteCarloResult]]:
        """
        Validate and adjust trading action based on risk limits.
        Optionally uses Monte Carlo simulation for risk assessment.
        
        Args:
            target_position: Desired position size from agent
            current_position: Current position size
            market_data: Market data (for stop loss calculation)
            price_data: Historical price data (for Monte Carlo)
            current_price: Current market price (for Monte Carlo)
            use_monte_carlo: Whether to use Monte Carlo risk assessment
            decision_context: Additional decision metadata (confluences, scaling, etc.)
            instrument: Instrument identifier for multi-instrument tracking
        
        Returns:
            Tuple of (adjusted_position_size, monte_carlo_result)
        """
        # Reset daily limits if needed
        self.reset_daily()
        instrument = instrument or "default"
        decision_context = decision_context or {}
        market_price = current_price
        if market_price is None and market_data:
            market_price = market_data.get("price") or market_data.get("close")
        
        # Check drawdown limit
        if self.current_drawdown >= self.limits.max_drawdown:
            print(f"⚠️  Max drawdown reached ({self.current_drawdown:.2%}). Stopping trading.")
            return 0.0, None
        
        # Check daily loss limit
        if self.daily_loss >= self.limits.max_daily_loss:
            print(f"⚠️  Daily loss limit reached ({self.daily_loss:.2%}). Stopping trading.")
            return 0.0, None
        
        # Check position size limit
        target_position = np.clip(
            target_position,
            -self.limits.max_position_size,
            self.limits.max_position_size
        )

        # Apply dynamic position cap relative to balance
        target_position = self._enforce_dynamic_position_cap(target_position)

        # Break-even / free trade management
        target_position = self._apply_break_even_logic(
            instrument=instrument,
            target_position=target_position,
            current_position=current_position,
            market_price=market_price,
            decision_context=decision_context
        )
        
        # Check leverage
        position_value = abs(target_position) * self.current_capital
        if position_value > self.current_capital * self.limits.max_leverage:
            # Reduce position to respect leverage
            max_position = self.limits.max_leverage * np.sign(target_position)
            target_position = np.clip(target_position, -max_position, max_position)
            print(f"⚠️  Position reduced to respect leverage limit")
        
        # Calculate position change
        position_change = target_position - current_position
        
        # Check if change is significant
        if abs(position_change) < 0.01:
            self._update_position_state(
                instrument=instrument,
                new_position=current_position,
                prior_position=current_position,
                execution_price=market_price
            )
            return current_position, None  # No change needed
        
        # Trend detection and asymmetric risk management for downtrends
        trend_adjustment = self._detect_trend_and_adjust(
            target_position,
            price_data,
            current_price
        )
        target_position = trend_adjustment
        
        # Volatility-based position adjustment (if available)
        volatility_multiplier = 1.0
        volatility_forecast = None
        if self.volatility_predictor and price_data is not None:
            try:
                volatility_forecast = self.volatility_predictor.predict_volatility(price_data, method="adaptive")
                volatility_multiplier = self.volatility_predictor.get_adaptive_position_multiplier(
                    target_position, volatility_forecast
                )
                # Apply volatility adjustment
                target_position = target_position * volatility_multiplier
                if abs(volatility_multiplier - 1.0) > 0.05:  # Significant adjustment
                    print(f"⚠️  Position adjusted based on volatility: {volatility_multiplier:.2f}x (volatility percentile: {volatility_forecast.volatility_percentile:.1f}%)")
            except Exception as e:
                print(f"⚠️  Volatility prediction failed: {e}")
        
        # Monte Carlo risk assessment (if available and requested)
        monte_carlo_result = None
        if use_monte_carlo and self.monte_carlo and price_data is not None and current_price is not None:
            try:
                # Calculate stop loss if available
                stop_loss = None
                if market_data:
                    stop_loss = self.calculate_stop_loss(
                        current_price, target_position, market_data
                    )
                
                # Run Monte Carlo simulation
                monte_carlo_result = self.monte_carlo.assess_trade_risk(
                    current_price=current_price,
                    proposed_position=target_position,
                    current_position=current_position,
                    price_data=price_data,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    time_horizon=1,
                    simulate_overnight=True
                )
                
                # Adjust position based on Monte Carlo recommendations
                if monte_carlo_result.risk_metrics.optimal_position_size < abs(target_position):
                    # Risk is too high - reduce position size
                    risk_adjusted_size = monte_carlo_result.risk_metrics.optimal_position_size * np.sign(target_position)
                    print(f"⚠️  Position reduced based on Monte Carlo risk: {target_position:.2f} → {risk_adjusted_size:.2f}")
                    target_position = risk_adjusted_size
                
                # Additional checks based on risk metrics
                if monte_carlo_result.risk_metrics.tail_risk > 0.10:  # More than 10% tail risk
                    print(f"⚠️  High tail risk ({monte_carlo_result.risk_metrics.tail_risk:.1%}). Reducing position.")
                    target_position *= 0.5  # Reduce by half
                
                if monte_carlo_result.risk_metrics.var_99 < -self.initial_capital * 0.05:  # VaR > 5% of capital
                    print(f"⚠️  High VaR ({monte_carlo_result.risk_metrics.var_99:.0f}). Reducing position.")
                    target_position *= 0.7  # Reduce by 30%
                    
            except Exception as e:
                print(f"⚠️  Monte Carlo risk assessment failed: {e}")
                # Continue with basic validation if Monte Carlo fails
        
        self._update_position_state(
            instrument=instrument,
            new_position=target_position,
            prior_position=current_position,
            execution_price=market_price
        )
        return target_position, monte_carlo_result
    
    def _empty_state(self) -> Dict[str, float]:
        return {
            "position": 0.0,
            "avg_entry": None,
            "break_even_active": False,
            "max_favorable": None,
            "trail_price": None,
            "protected_size": 0.0,
        }

    def _get_position_state(self, instrument: str) -> Dict[str, float]:
        if instrument not in self.position_states:
            self.position_states[instrument] = self._empty_state()
        return self.position_states[instrument]

    def get_position_state_info(self, instrument: str) -> Dict[str, float]:
        """Public accessor for current position state (for logging/debugging)."""
        state = self.position_states.get(instrument)
        if state is None:
            state = self._empty_state()
        # Return a shallow copy to avoid external mutation
        return dict(state)

    def _enforce_dynamic_position_cap(self, target_position: float) -> float:
        if self.position_value_per_unit <= 0 or self.dynamic_position_fraction <= 0:
            return target_position
        dynamic_cap = (self.current_capital * self.dynamic_position_fraction) / self.position_value_per_unit
        dynamic_cap = max(self.limits.max_position_size, dynamic_cap)
        return float(np.clip(target_position, -dynamic_cap, dynamic_cap))

    def _apply_break_even_logic(
        self,
        instrument: str,
        target_position: float,
        current_position: float,
        market_price: Optional[float],
        decision_context: Dict
    ) -> float:
        if not self.break_even_cfg.get("enabled", True):
            return target_position
        if market_price is None:
            return target_position

        state = self._get_position_state(instrument)
        existing_position = state.get("position", 0.0)
        if abs(existing_position) < 1e-6:
            # No active position to manage yet
            return target_position

        direction = np.sign(existing_position)
        avg_entry = state.get("avg_entry")
        if not avg_entry or direction == 0:
            return target_position

        move_pct = direction * (market_price - avg_entry) / avg_entry if avg_entry else 0.0
        activation_pct = self.break_even_cfg.get("activation_pct", 0.003)
        if not state.get("break_even_active") and move_pct >= activation_pct:
            state["break_even_active"] = True
            state["protected_size"] = max(state.get("protected_size", 0.0), abs(existing_position))
            state["max_favorable"] = market_price
            state["trail_price"] = market_price

        if state.get("break_even_active"):
            # Update trailing reference
            if direction > 0:
                state["max_favorable"] = max(state.get("max_favorable") or market_price, market_price)
                trail_pct = self.break_even_cfg.get("trail_pct", 0.0015)
                state["trail_price"] = state["max_favorable"] * (1 - trail_pct)
                if market_price <= state.get("trail_price", market_price):
                    return 0.0
            else:
                state["max_favorable"] = min(state.get("max_favorable") or market_price, market_price)
                trail_pct = self.break_even_cfg.get("trail_pct", 0.0015)
                state["trail_price"] = state["max_favorable"] * (1 + trail_pct)
                if market_price >= state.get("trail_price", market_price):
                    return 0.0

            min_confluence = self.break_even_cfg.get("scale_out_min_confluence", 1)
            confluence_count = int(decision_context.get("confluence_count", 0))
            if confluence_count <= min_confluence:
                free_fraction = self.break_even_cfg.get("free_trade_fraction", 0.5)
                protected_size = state.get("protected_size", abs(existing_position))
                target_size = protected_size * free_fraction
                target_position = direction * min(abs(target_position), target_size)

        return target_position

    def _update_position_state(
        self,
        instrument: str,
        new_position: float,
        prior_position: float,
        execution_price: Optional[float]
    ):
        state = self._get_position_state(instrument)
        direction = np.sign(new_position) if abs(new_position) > 1e-6 else 0.0

        if direction == 0.0:
            self.position_states[instrument] = self._empty_state()
            self.current_positions[instrument] = 0.0
            return

        price = execution_price or state.get("avg_entry")
        if price is None:
            price = 0.0

        prev_direction = np.sign(prior_position) if abs(prior_position) > 1e-6 else 0.0
        new_size = abs(new_position)
        prev_size = abs(prior_position)

        if prev_direction == 0.0 or prev_direction != direction:
            # Fresh position or reversal
            state.update({
                "position": new_position,
                "avg_entry": price,
                "break_even_active": False,
                "max_favorable": price,
                "trail_price": price,
                "protected_size": new_size,
            })
        else:
            state["position"] = new_position
            if new_size > prev_size and price:
                avg_entry = state.get("avg_entry") or price
                state["avg_entry"] = (
                    (avg_entry * prev_size) + (price * (new_size - prev_size))
                ) / max(new_size, 1e-6)
            if state.get("break_even_active"):
                state["protected_size"] = max(state.get("protected_size", 0.0), new_size)
            if price:
                if direction > 0:
                    state["max_favorable"] = max(state.get("max_favorable") or price, price)
                else:
                    state["max_favorable"] = min(state.get("max_favorable") or price, price)

        self.current_positions[instrument] = new_position

    def _detect_trend_and_adjust(
        self,
        target_position: float,
        price_data: Optional[pd.DataFrame],
        current_price: Optional[float]
    ) -> float:
        """
        Detect market trend and apply asymmetric risk management for downtrends.
        
        In downtrends:
        - Reduce long positions by 50-70%
        - Tighter stop losses (1.5x ATR instead of 2.0x)
        - Reduce position size more aggressively
        
        In uptrends:
        - Normal position sizing
        - Standard stop losses
        
        Args:
            target_position: Target position size
            price_data: Historical price data
            current_price: Current market price
        
        Returns:
            Adjusted position size
        """
        if price_data is None or current_price is None or len(price_data) < 20:
            # Not enough data to detect trend - return unchanged
            return target_position
        
        try:
            # Calculate trend indicators
            # 1. Moving average trend (20-period SMA)
            if 'close' in price_data.columns:
                closes = price_data['close'].values
                if len(closes) >= 20:
                    sma_20 = np.mean(closes[-20:])
                    sma_50 = np.mean(closes[-min(50, len(closes)):]) if len(closes) >= 50 else sma_20
                    
                    # 2. Price momentum (rate of change)
                    price_change_20 = (closes[-1] - closes[-20]) / closes[-20] if len(closes) >= 20 else 0.0
                    price_change_5 = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0.0
                    
                    # 3. Trend strength (how far price is from moving average)
                    trend_strength = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0.0
                    
                    # Detect downtrend
                    is_downtrend = (
                        (current_price < sma_20 and sma_20 < sma_50) or  # Price below both MAs
                        (price_change_20 < -0.03) or  # 20-period decline > 3%
                        (price_change_5 < -0.01 and trend_strength < -0.01)  # Recent decline with negative trend
                    )
                    
                    # Detect strong downtrend
                    is_strong_downtrend = (
                        (current_price < sma_20 * 0.98) or  # Price 2%+ below SMA
                        (price_change_20 < -0.05) or  # 20-period decline > 5%
                        (trend_strength < -0.02)  # Strong negative trend
                    )
                    
                    # Apply asymmetric risk management
                    if is_downtrend and target_position > 0:
                        # Long position in downtrend - reduce aggressively
                        if is_strong_downtrend:
                            # Strong downtrend: reduce long positions by 70%
                            reduction = 0.3
                            print(f"⚠️  Strong downtrend detected. Reducing long position by 70%.")
                        else:
                            # Moderate downtrend: reduce long positions by 50%
                            reduction = 0.5
                            print(f"⚠️  Downtrend detected. Reducing long position by 50%.")
                        
                        adjusted_position = target_position * reduction
                        return adjusted_position
                    
                    elif is_downtrend and target_position < 0:
                        # Short position in downtrend - allow but slightly reduce (10%) for safety
                        adjusted_position = target_position * 0.9
                        if abs(adjusted_position - target_position) > 0.01:
                            print(f"ℹ️  Downtrend detected. Slightly reducing short position for safety.")
                        return adjusted_position
                    
                    elif not is_downtrend and target_position < 0:
                        # Short position in uptrend or neutral - reduce by 40%
                        adjusted_position = target_position * 0.6
                        print(f"⚠️  Uptrend/neutral market. Reducing short position by 40%.")
                        return adjusted_position
                    
                    # Uptrend with long position - normal sizing (no adjustment)
                    return target_position
                
        except Exception as e:
            # If trend detection fails, return unchanged position
            print(f"⚠️  Trend detection failed: {e}. Using original position size.")
            return target_position
        
        # Default: return unchanged
        return target_position
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        position_size: float,
        market_data: Optional[Dict] = None
    ) -> float:
        """
        Calculate stop loss price.
        
        Args:
            entry_price: Entry price
            position_size: Position size (positive for long, negative for short)
            market_data: Market data with ATR or volatility
        
        Returns:
            Stop loss price
        """
        if market_data and "atr" in market_data:
            atr = market_data["atr"]
            stop_distance = atr * self.limits.stop_loss_atr_multiplier
        else:
            # Default: 2% stop loss
            stop_distance = entry_price * 0.02
        
        if position_size > 0:  # Long position
            stop_loss = entry_price - stop_distance
        else:  # Short position
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def should_close_position(
        self,
        current_position: float,
        entry_price: float,
        current_price: float,
        market_data: Optional[Dict] = None
    ) -> bool:
        """
        Determine if position should be closed based on risk rules.
        
        Returns:
            True if position should be closed
        """
        if abs(current_position) < 0.01:
            return False  # No position
        
        # Calculate PnL
        if current_position > 0:  # Long
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # Short
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Check stop loss
        stop_loss = self.calculate_stop_loss(entry_price, current_position, market_data)
        if (current_position > 0 and current_price <= stop_loss) or \
           (current_position < 0 and current_price >= stop_loss):
            return True
        
        # Check drawdown limit
        if self.current_drawdown >= self.limits.max_drawdown:
            return True
        
        return False
    
    def get_risk_status(self) -> Dict:
        """Get current risk status"""
        return {
            "current_capital": self.current_capital,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.limits.max_drawdown,
            "daily_loss": self.daily_loss,
            "daily_loss_limit": self.limits.max_daily_loss,
            "can_trade": self.current_drawdown < self.limits.max_drawdown and \
                        self.daily_loss < self.limits.max_daily_loss
        }


# Example usage
if __name__ == "__main__":
    # Test risk manager
    risk_config = {
        "max_position_size": 1.0,
        "max_drawdown": 0.20,
        "max_daily_loss": 0.05,
        "stop_loss_atr_multiplier": 2.0,
        "initial_capital": 100000.0
    }
    
    rm = RiskManager(risk_config)
    
    # Test position validation
    target_pos = 0.8
    validated = rm.validate_action(target_pos, 0.0)
    print(f"Target: {target_pos}, Validated: {validated}")
    
    # Test with drawdown
    rm.update_capital(-50000)  # Simulate loss
    validated = rm.validate_action(target_pos, 0.0)
    print(f"After loss - Target: {target_pos}, Validated: {validated}")
    print(f"Risk status: {rm.get_risk_status()}")

