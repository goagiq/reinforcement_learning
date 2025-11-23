"""
Market Data Provider

Provides access to historical and live market data for ES, NQ, RTY, YM.
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from src.data_extraction import DataExtractor, MarketBar
from src.data_sources.cache import DataCache
from src.trading_hours import TradingHoursManager


class MarketDataProvider:
    """
    Provides market data for multiple instruments.
    
    Supports:
    - Historical data from NT8 exports
    - Live data from NT8 bridge
    - Multi-timeframe data
    """
    
    def __init__(self, config: Dict, cache: Optional[DataCache] = None):
        """
        Initialize market data provider.
        
        Args:
            config: Configuration with data paths and settings
            cache: Optional cache instance
        """
        self.config = config
        self.cache = cache or DataCache()
        # Initialize data extractor with data path from config
        data_path = config.get("data_path", "data")
        self.data_extractor = DataExtractor(data_dir=data_path)
        
        # Instruments to track
        self.instruments = config.get("instruments", ["ES", "NQ", "RTY", "YM"])
        self.timeframes = config.get("timeframes", [1, 5, 15])
        
        trading_hours_cfg = config.get("trading_hours", {})
        self.trading_hours_manager = None
        if trading_hours_cfg.get("enabled"):
            self.trading_hours_manager = TradingHoursManager.from_dict(trading_hours_cfg)

        # Data storage
        self._historical_data: Dict[str, Dict[int, pd.DataFrame]] = {}
        self._live_data: Dict[str, List[MarketBar]] = {}
        
        # Load historical data
        self._load_historical_data()
    
    def _load_historical_data(self) -> None:
        """Load historical data for all instruments."""
        data_path = Path(self.config.get("data_path", "data"))
        
        for instrument in self.instruments:
            self._historical_data[instrument] = {}
            
            for timeframe in self.timeframes:
                try:
                    # Try to load from cache first
                    cache_key = f"historical_{instrument}_{timeframe}"
                    cached = self.cache.get(cache_key)
                    
                    if cached is not None:
                        self._historical_data[instrument][timeframe] = cached
                        continue
                    
                    # Load from data extractor
                    try:
                        # DataExtractor.load_historical_data returns a DataFrame
                        df = self.data_extractor.load_historical_data(
                            instrument=instrument,
                            timeframe=timeframe,
                            trading_hours=self.trading_hours_manager,
                        )
                        
                        if df is not None and len(df) > 0:
                            # Cache for 1 hour
                            self.cache.set(cache_key, df, ttl=3600)
                            self._historical_data[instrument][timeframe] = df
                    except Exception as e:
                        print(f"Warning: Could not load {instrument} {timeframe}min: {e}")
                    
                except Exception as e:
                    print(f"Warning: Could not load {instrument} {timeframe}min data: {e}")
    
    def get_historical_data(
        self,
        instrument: str,
        timeframe: int,
        lookback_bars: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data for an instrument.
        
        Args:
            instrument: Instrument symbol (ES, NQ, RTY, YM)
            timeframe: Timeframe in minutes (1, 5, 15)
            lookback_bars: Number of bars to return
        
        Returns:
            DataFrame with OHLCV data or None if not available
        """
        if instrument not in self._historical_data:
            return None
        
        if timeframe not in self._historical_data[instrument]:
            return None
        
        df = self._historical_data[instrument][timeframe]
        
        # Return last N bars
        if len(df) > lookback_bars:
            return df.tail(lookback_bars).copy()
        
        return df.copy()
    
    def get_correlation_matrix(
        self,
        timeframe: int = 5,
        lookback_bars: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Calculate correlation matrix between instruments.
        
        Args:
            timeframe: Timeframe to use
            lookback_bars: Number of bars for correlation
        
        Returns:
            Correlation matrix DataFrame or None
        """
        # Get data for all instruments
        data_dict = {}
        for instrument in self.instruments:
            df = self.get_historical_data(instrument, timeframe, lookback_bars)
            if df is not None and len(df) > 0:
                data_dict[instrument] = df["close"].values
        
        if len(data_dict) < 2:
            return None
        
        # Align data (handle different lengths)
        min_len = min(len(v) for v in data_dict.values())
        aligned_data = {k: v[-min_len:] for k, v in data_dict.items()}
        
        # Create DataFrame and calculate correlation
        df = pd.DataFrame(aligned_data)
        return df.corr()
    
    def get_rolling_correlation(
        self,
        instrument1: str,
        instrument2: str,
        timeframe: int = 5,
        window: int = 20
    ) -> Optional[float]:
        """
        Calculate rolling correlation between two instruments.
        
        Args:
            instrument1: First instrument
            instrument2: Second instrument
            timeframe: Timeframe to use
            window: Rolling window size
        
        Returns:
            Correlation coefficient or None
        """
        df1 = self.get_historical_data(instrument1, timeframe, window)
        df2 = self.get_historical_data(instrument2, timeframe, window)
        
        if df1 is None or df2 is None:
            return None
        
        if len(df1) < window or len(df2) < window:
            return None
        
        # Get close prices
        prices1 = df1["close"].values[-window:]
        prices2 = df2["close"].values[-window:]
        
        # Calculate correlation
        correlation = np.corrcoef(prices1, prices2)[0, 1]
        return float(correlation) if not np.isnan(correlation) else None
    
    def get_divergence_signal(
        self,
        base_instrument: str = "ES",
        comparison_instruments: Optional[List[str]] = None,
        timeframe: int = 5,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect divergence/convergence signals.
        
        Args:
            base_instrument: Base instrument (typically ES)
            comparison_instruments: Instruments to compare (default: all others)
            timeframe: Timeframe to use
            threshold: Divergence threshold
        
        Returns:
            Dict with divergence signals
        """
        if comparison_instruments is None:
            comparison_instruments = [i for i in self.instruments if i != base_instrument]
        
        base_df = self.get_historical_data(base_instrument, timeframe, 20)
        if base_df is None or len(base_df) < 10:
            return {"signal": "insufficient_data", "correlations": {}}
        
        base_returns = base_df["close"].pct_change().dropna()
        
        signals = {}
        correlations = {}
        
        for instrument in comparison_instruments:
            comp_df = self.get_historical_data(instrument, timeframe, 20)
            if comp_df is None or len(comp_df) < 10:
                continue
            
            comp_returns = comp_df["close"].pct_change().dropna()
            
            # Align lengths
            min_len = min(len(base_returns), len(comp_returns))
            base_ret = base_returns.values[-min_len:]
            comp_ret = comp_returns.values[-min_len:]
            
            # Calculate correlation
            corr = np.corrcoef(base_ret, comp_ret)[0, 1]
            correlations[instrument] = float(corr) if not np.isnan(corr) else 0.0
            
            # Detect divergence
            if abs(corr) < (1.0 - threshold):
                signals[instrument] = "divergence"
            elif abs(corr) > (1.0 - threshold * 0.5):
                signals[instrument] = "convergence"
            else:
                signals[instrument] = "normal"
        
        return {
            "signal": "divergence" if any(s == "divergence" for s in signals.values()) else "normal",
            "signals": signals,
            "correlations": correlations,
            "base_instrument": base_instrument
        }
    
    def update_live_data(self, instrument: str, bar: MarketBar) -> None:
        """
        Update live market data.
        
        Args:
            instrument: Instrument symbol
            bar: Market bar data
        """
        if instrument not in self._live_data:
            self._live_data[instrument] = []
        
        self._live_data[instrument].append(bar)
        
        # Keep only last 100 bars
        if len(self._live_data[instrument]) > 100:
            self._live_data[instrument] = self._live_data[instrument][-100:]
    
    def get_latest_data(self, instrument: str) -> Optional[MarketBar]:
        """
        Get latest live data for an instrument.
        
        Args:
            instrument: Instrument symbol
        
        Returns:
            Latest MarketBar or None
        """
        if instrument not in self._live_data or not self._live_data[instrument]:
            return None
        
        return self._live_data[instrument][-1]

