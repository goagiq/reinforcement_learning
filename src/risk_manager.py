"""
Risk Management Module

Implements risk controls and position sizing logic.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


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
        
        # State tracking
        self.initial_capital = risk_config.get("initial_capital", 100000.0)
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
        market_data: Optional[Dict] = None
    ) -> float:
        """
        Validate and adjust trading action based on risk limits.
        
        Args:
            target_position: Desired position size from agent
            current_position: Current position size
            market_data: Market data (for stop loss calculation)
        
        Returns:
            Adjusted position size (may be reduced or set to 0)
        """
        # Reset daily limits if needed
        self.reset_daily()
        
        # Check drawdown limit
        if self.current_drawdown >= self.limits.max_drawdown:
            print(f"⚠️  Max drawdown reached ({self.current_drawdown:.2%}). Stopping trading.")
            return 0.0
        
        # Check daily loss limit
        if self.daily_loss >= self.limits.max_daily_loss:
            print(f"⚠️  Daily loss limit reached ({self.daily_loss:.2%}). Stopping trading.")
            return 0.0
        
        # Check position size limit
        target_position = np.clip(
            target_position,
            -self.limits.max_position_size,
            self.limits.max_position_size
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
            return current_position  # No change needed
        
        # Additional validation: volatility-based position sizing
        if market_data:
            # Could calculate ATR or volatility here
            # For now, just ensure we don't exceed limits
            pass
        
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

