"""
Trading hours configuration and filtering utilities.

Supports multi-session schedules across different global markets and
provides helpers to filter pandas DataFrames or individual timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd


@dataclass(frozen=True)
class TradingSession:
    """Defines a single trading session window in a specific timezone."""

    name: str
    timezone: str
    start: time
    end: time

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def contains(self, dt: datetime, source_tz: ZoneInfo) -> bool:
        """Check if a timestamp falls within this session."""
        localized = _ensure_timezone(dt, source_tz)
        session_dt = localized.astimezone(self.zone)

        start_minutes = self.start.hour * 60 + self.start.minute
        end_minutes = self.end.hour * 60 + self.end.minute
        current_minutes = (
            session_dt.hour * 60 + session_dt.minute + session_dt.second / 60.0
        )

        if start_minutes <= end_minutes:
            return start_minutes <= current_minutes <= end_minutes

        # Overnight session (e.g., 22:00 -> 05:00)
        return current_minutes >= start_minutes or current_minutes <= end_minutes


@dataclass
class TradingHoursConfig:
    enabled: bool
    source_timezone: str
    exclude_weekends: bool
    use_nt8_calendar: bool
    sessions: List[TradingSession]

    @classmethod
    def from_dict(cls, data: Dict) -> "TradingHoursConfig":
        enabled = data.get("enabled", False)
        source_timezone = data.get("source_timezone", "America/New_York")
        exclude_weekends = data.get("exclude_weekends", True)
        use_nt8_calendar = data.get("use_nt8_calendar", True)

        session_dicts = data.get("sessions", [])
        sessions: List[TradingSession] = []
        for session in session_dicts:
            start = _parse_time(session.get("start", "00:00"))
            end = _parse_time(session.get("end", "23:59"))
            sessions.append(
                TradingSession(
                    name=session.get("name", "Session"),
                    timezone=session.get("timezone", source_timezone),
                    start=start,
                    end=end,
                )
            )

        return cls(
            enabled=enabled,
            source_timezone=source_timezone,
            exclude_weekends=exclude_weekends,
            use_nt8_calendar=use_nt8_calendar,
            sessions=sessions,
        )


class TradingHoursManager:
    """Manage trading session filtering for historical and live data."""

    def __init__(self, config: TradingHoursConfig):
        self.config = config
        self.source_zone = ZoneInfo(config.source_timezone)
        self.sessions = config.sessions

    @classmethod
    def from_dict(cls, data: Dict) -> "TradingHoursManager":
        config = TradingHoursConfig.from_dict(data)
        return cls(config)

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only rows that fall within configured trading sessions."""
        if not self.config.enabled or df.empty or not self.sessions:
            return df

        timestamps = pd.DatetimeIndex(df["timestamp"])
        if timestamps.tz is None:
            localized = timestamps.tz_localize(
                self.source_zone,
                ambiguous="NaT",
                nonexistent="shift_forward",
            )
        else:
            localized = timestamps.tz_convert(self.source_zone)

        if self.config.exclude_weekends:
            weekday_mask = ~localized.weekday.isin([5, 6])
        else:
            weekday_mask = pd.Series([True] * len(localized), index=df.index)

        session_mask = pd.Series([False] * len(localized), index=df.index)
        for session in self.sessions:
            session_index = localized.tz_convert(session.zone)
            current_minutes = (
                session_index.hour * 60 + session_index.minute + session_index.second / 60.0
            )
            start_minutes = session.start.hour * 60 + session.start.minute
            end_minutes = session.end.hour * 60 + session.end.minute

            if start_minutes <= end_minutes:
                mask = (current_minutes >= start_minutes) & (current_minutes <= end_minutes)
            else:
                mask = (current_minutes >= start_minutes) | (current_minutes <= end_minutes)

            session_mask = session_mask | pd.Series(mask, index=df.index)

        final_mask = weekday_mask & session_mask
        return df.loc[final_mask].reset_index(drop=True)

    def is_in_session(self, dt: datetime) -> bool:
        """Check if a single timestamp is inside any configured session."""
        if not self.config.enabled or not self.sessions:
            return True

        if dt.tzinfo is None:
            localized = dt.replace(tzinfo=self.source_zone)
        else:
            localized = dt.astimezone(self.source_zone)

        if self.config.exclude_weekends and localized.weekday() in (5, 6):
            return False

        return any(session.contains(localized, self.source_zone) for session in self.sessions)


def _parse_time(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def _ensure_timezone(dt: datetime, source_zone: ZoneInfo) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=source_zone)
    return dt.astimezone(source_zone)


