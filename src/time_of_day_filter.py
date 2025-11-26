"""
Time-of-Day Filter for Trading Entry Timing

Filters out low-quality trading hours (e.g., lunch hours, low liquidity periods)
based on analysis of historical trade performance by hour.
"""

from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo


class TimeOfDayFilter:
    """
    Filters trades based on time of day to avoid low-quality entry times.
    
    Common low-quality periods:
    - Lunch hours (11:30-14:00 ET): Low liquidity, choppy price action
    - Market open (09:30-10:00 ET): High volatility, false breakouts
    - Market close (15:30-16:00 ET): Low liquidity, erratic moves
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize time-of-day filter.
        
        Args:
            config: Configuration dict with:
                - enabled: bool - Enable/disable filter
                - timezone: str - Timezone for hour calculation (default: "America/New_York")
                - avoid_hours: List[Tuple[int, int]] - List of (start_hour, end_hour) to avoid (24-hour format)
                - strict_mode: bool - If True, reject all trades in avoid_hours. If False, reduce confidence.
        """
        config = config or {}
        self.enabled = config.get("enabled", True)
        self.timezone_str = config.get("timezone", "America/New_York")
        self.timezone = ZoneInfo(self.timezone_str)
        
        # Default avoid hours: lunch hours (11:30-14:00 ET)
        default_avoid_hours = config.get("avoid_hours", [(11, 30, 14, 0)])  # (start_hour, start_min, end_hour, end_min)
        self.avoid_periods: List[Tuple[int, int, int, int]] = [
            tuple(period) if len(period) == 4 else (period[0], 0, period[1], 0)
            for period in default_avoid_hours
        ]
        
        self.strict_mode = config.get("strict_mode", False)  # If True, reject trades. If False, reduce confidence.
        self.confidence_reduction = config.get("confidence_reduction", 0.3)  # Reduce confidence by 30% in avoid hours
        
    def is_in_avoid_period(self, dt: datetime) -> bool:
        """
        Check if a datetime falls within any avoid period.
        
        Args:
            dt: Datetime to check
            
        Returns:
            True if datetime is in an avoid period, False otherwise
        """
        if not self.enabled:
            return False
        
        # Convert to target timezone
        if dt.tzinfo is None:
            localized = dt.replace(tzinfo=self.timezone)
        else:
            localized = dt.astimezone(self.timezone)
        
        current_hour = localized.hour
        current_minute = localized.minute
        current_minutes = current_hour * 60 + current_minute
        
        # Check each avoid period
        for start_hour, start_min, end_hour, end_min in self.avoid_periods:
            start_minutes = start_hour * 60 + start_min
            end_minutes = end_hour * 60 + end_min
            
            if start_minutes <= end_minutes:
                # Normal period (e.g., 11:30-14:00)
                if start_minutes <= current_minutes <= end_minutes:
                    return True
            else:
                # Overnight period (e.g., 22:00-02:00)
                if current_minutes >= start_minutes or current_minutes <= end_minutes:
                    return True
        
        return False
    
    def filter_decision(
        self,
        dt: datetime,
        action: float,
        confidence: float
    ) -> Tuple[float, float, str]:
        """
        Filter a trading decision based on time of day.
        
        Args:
            dt: Current datetime
            action: Proposed action (position size)
            confidence: Proposed confidence
            
        Returns:
            Tuple of (filtered_action, filtered_confidence, reason)
            - If strict_mode: action=0.0, confidence=0.0, reason="rejected"
            - If not strict_mode: action unchanged, confidence reduced, reason="reduced"
            - If not in avoid period: unchanged, reason="allowed"
        """
        if not self.enabled:
            return action, confidence, "time_filter_disabled"
        
        if not self.is_in_avoid_period(dt):
            return action, confidence, "allowed"
        
        if self.strict_mode:
            # Reject trade completely
            return 0.0, 0.0, "rejected_time_of_day"
        else:
            # Reduce confidence but allow trade
            reduced_confidence = confidence * (1.0 - self.confidence_reduction)
            return action, reduced_confidence, "reduced_confidence_time_of_day"
    
    def get_avoid_periods_description(self) -> str:
        """Get human-readable description of avoid periods"""
        if not self.enabled:
            return "Time-of-day filter disabled"
        
        if not self.avoid_periods:
            return "No avoid periods configured"
        
        descriptions = []
        for start_hour, start_min, end_hour, end_min in self.avoid_periods:
            start_str = f"{start_hour:02d}:{start_min:02d}"
            end_str = f"{end_hour:02d}:{end_min:02d}"
            descriptions.append(f"{start_str}-{end_str}")
        
        mode_str = "strict (reject)" if self.strict_mode else f"reduce confidence by {self.confidence_reduction*100:.0f}%"
        return f"Avoiding {', '.join(descriptions)} ({mode_str})"


def create_default_filter() -> TimeOfDayFilter:
    """Create default time-of-day filter with common avoid hours"""
    return TimeOfDayFilter({
        "enabled": True,
        "timezone": "America/New_York",
        "avoid_hours": [
            (11, 30, 14, 0),  # Lunch hours: 11:30-14:00 ET
        ],
        "strict_mode": False,  # Reduce confidence, don't reject
        "confidence_reduction": 0.3  # 30% reduction
    })

