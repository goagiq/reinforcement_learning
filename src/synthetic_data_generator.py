"""
Synthetic Market Data Generator

Generates realistic synthetic OHLCV data with known patterns for testing
and supervised pre-training. This helps the model learn clear trading signals.

For quant traders: This is like creating a "training course" with known
answers, so the model can learn basic patterns before facing real market noise.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path
from src.utils.colors import success, info, warn, error


class SyntheticDataGenerator:
    """
    Generate synthetic market data with various patterns:
    - Trend following (uptrends/downtrends)
    - Mean reversion
    - Breakouts
    - Volatility clusters
    """
    
    def __init__(
        self,
        base_price: float = 4000.0,
        volatility: float = 0.01,  # 1% daily volatility
        trend_strength: float = 0.0005,  # 0.05% per bar trend
        seed: Optional[int] = None
    ):
        """
        Initialize synthetic data generator.
        
        Args:
            base_price: Starting price
            volatility: Price volatility (std dev of returns)
            trend_strength: Strength of trend component
            seed: Random seed for reproducibility
        """
        self.base_price = base_price
        self.volatility = volatility
        self.trend_strength = trend_strength
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
    
    def generate_trend_data(
        self,
        n_bars: int,
        trend_direction: str = "up",  # "up", "down", or "mixed"
        trend_duration: int = 100,  # Bars per trend segment
        noise_level: float = 0.3  # How much noise vs trend
    ) -> pd.DataFrame:
        """
        Generate data with clear trends that should produce buy/sell signals.
        
        Args:
            n_bars: Number of bars to generate
            trend_direction: Overall trend direction
            trend_duration: How many bars before trend changes
            noise_level: 0.0 = pure trend, 1.0 = pure noise
        
        Returns:
            DataFrame with OHLCV data
        """
        prices = []
        volumes = []
        current_price = self.base_price
        
        # Generate trend segments
        n_segments = (n_bars // trend_duration) + 1
        
        for segment in range(n_segments):
            # Determine trend for this segment
            if trend_direction == "up":
                segment_trend = self.trend_strength * (1.0 - noise_level)
            elif trend_direction == "down":
                segment_trend = -self.trend_strength * (1.0 - noise_level)
            else:  # mixed
                segment_trend = self.trend_strength * (1.0 - noise_level) * (1 if segment % 2 == 0 else -1)
            
            # Generate bars for this segment
            segment_start = segment * trend_duration
            segment_end = min((segment + 1) * trend_duration, n_bars)
            
            for i in range(segment_start, segment_end):
                # Trend component
                trend_change = segment_trend * current_price
                
                # Noise component
                noise = self.rng.normal(0, self.volatility * current_price * noise_level)
                
                # Price change
                price_change = trend_change + noise
                new_price = current_price + price_change
                
                # Generate OHLC from price movement
                high = max(current_price, new_price) * (1 + abs(self.rng.normal(0, 0.001)))
                low = min(current_price, new_price) * (1 - abs(self.rng.normal(0, 0.001)))
                open_price = current_price
                close_price = new_price
                
                # Volume (higher on trend changes)
                base_volume = 1000000
                volume_multiplier = 1.0 + abs(price_change / current_price) * 2.0
                volume = int(base_volume * volume_multiplier * (1 + self.rng.normal(0, 0.2)))
                volume = max(1000, volume)  # Minimum volume
                
                prices.append({
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': volume
                })
                
                current_price = new_price
        
        df = pd.DataFrame(prices)
        # Create proper datetime index for resampling
        start_time = datetime.now()
        df.index = pd.date_range(
            start=start_time,
            periods=len(df),
            freq='1min'
        )
        return df
    
    def generate_mean_reversion_data(
        self,
        n_bars: int,
        mean_price: float = None,
        reversion_strength: float = 0.1,  # How quickly price reverts to mean
        volatility: float = None
    ) -> pd.DataFrame:
        """
        Generate mean-reverting data (good for range trading strategies).
        
        Args:
            n_bars: Number of bars
            mean_price: Mean price to revert to (default: base_price)
            reversion_strength: Strength of mean reversion (0-1)
            volatility: Price volatility (default: self.volatility)
        
        Returns:
            DataFrame with OHLCV data
        """
        if mean_price is None:
            mean_price = self.base_price
        if volatility is None:
            volatility = self.volatility
        
        prices = []
        current_price = mean_price
        
        for i in range(n_bars):
            # Mean reversion component
            deviation = current_price - mean_price
            reversion_force = -reversion_strength * deviation
            
            # Random walk component
            random_walk = self.rng.normal(0, volatility * mean_price)
            
            # Price change
            price_change = reversion_force + random_walk
            new_price = current_price + price_change
            
            # Generate OHLC
            high = max(current_price, new_price) * (1 + abs(self.rng.normal(0, 0.001)))
            low = min(current_price, new_price) * (1 - abs(self.rng.normal(0, 0.001)))
            open_price = current_price
            close_price = new_price
            
            # Volume
            volume = int(1000000 * (1 + self.rng.normal(0, 0.2)))
            volume = max(1000, volume)
            
            prices.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
            
            current_price = new_price
        
        df = pd.DataFrame(prices)
        # Create proper datetime index for resampling
        start_time = datetime.now()
        df.index = pd.date_range(
            start=start_time,
            periods=len(df),
            freq='1min'
        )
        return df
    
    def generate_breakout_data(
        self,
        n_bars: int,
        consolidation_bars: int = 50,
        breakout_strength: float = 0.02  # 2% breakout move
    ) -> pd.DataFrame:
        """
        Generate data with consolidation and breakout patterns.
        
        Args:
            n_bars: Total number of bars
            consolidation_bars: Bars in consolidation before breakout
            breakout_strength: Strength of breakout move
        
        Returns:
            DataFrame with OHLCV data
        """
        prices = []
        current_price = self.base_price
        
        n_consolidations = n_bars // consolidation_bars
        
        for consolidation_idx in range(n_consolidations):
            # Consolidation phase
            consolidation_start = consolidation_idx * consolidation_bars
            consolidation_end = min((consolidation_idx + 1) * consolidation_bars, n_bars)
            
            # Determine breakout direction (random)
            breakout_direction = 1 if self.rng.random() > 0.5 else -1
            
            for i in range(consolidation_start, consolidation_end):
                # In consolidation: small moves around base price
                if i == consolidation_start:
                    # Start of consolidation
                    price_change = self.rng.normal(0, 0.001 * current_price)
                elif i == consolidation_end - 1:
                    # Breakout!
                    price_change = breakout_direction * breakout_strength * current_price
                else:
                    # During consolidation: small random moves
                    price_change = self.rng.normal(0, 0.002 * current_price)
                
                new_price = current_price + price_change
                
                # Generate OHLC
                high = max(current_price, new_price) * (1 + abs(self.rng.normal(0, 0.001)))
                low = min(current_price, new_price) * (1 - abs(self.rng.normal(0, 0.001)))
                open_price = current_price
                close_price = new_price
                
                # Volume (higher on breakout)
                if i == consolidation_end - 1:
                    volume = int(2000000)  # High volume on breakout
                else:
                    volume = int(1000000 * (1 + self.rng.normal(0, 0.2)))
                volume = max(1000, volume)
                
                prices.append({
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': volume
                })
                
                current_price = new_price
        
        df = pd.DataFrame(prices)
        # Create proper datetime index for resampling
        start_time = datetime.now()
        df.index = pd.date_range(
            start=start_time,
            periods=len(df),
            freq='1min'
        )
        return df
    
    def generate_mixed_pattern_data(
        self,
        n_bars: int,
        pattern_length: int = 200
    ) -> pd.DataFrame:
        """
        Generate data with mixed patterns (trends, mean reversion, breakouts).
        This is most realistic for training.
        
        Args:
            n_bars: Total number of bars
            pattern_length: Bars per pattern type
        
        Returns:
            DataFrame with OHLCV data
        """
        all_data = []
        
        n_patterns = n_bars // pattern_length
        
        for i in range(n_patterns):
            pattern_type = i % 3
            
            if pattern_type == 0:
                # Trend pattern
                pattern_data = self.generate_trend_data(
                    n_bars=pattern_length,
                    trend_direction="mixed",
                    trend_duration=50,
                    noise_level=0.3
                )
            elif pattern_type == 1:
                # Mean reversion pattern
                mean_price = self.base_price * (1 + (i - n_patterns/2) * 0.01)
                pattern_data = self.generate_mean_reversion_data(
                    n_bars=pattern_length,
                    mean_price=mean_price,
                    reversion_strength=0.1
                )
            else:
                # Breakout pattern
                pattern_data = self.generate_breakout_data(
                    n_bars=pattern_length,
                    consolidation_bars=50,
                    breakout_strength=0.02
                )
            
            all_data.append(pattern_data)
        
        # Combine all patterns
        combined_df = pd.concat(all_data, ignore_index=False)
        combined_df = combined_df.sort_index()
        
        return combined_df
    
    def save_to_file(
        self,
        df: pd.DataFrame,
        filepath: Path,
        instrument: str = "ES",
        timeframe: int = 1
    ):
        """
        Save synthetic data to file in NT8 export format.
        
        Args:
            df: DataFrame with OHLCV data
            filepath: Path to save file
            instrument: Instrument symbol
            timeframe: Timeframe in minutes
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Format: datetime, open, high, low, close, volume
        df_export = df.copy()
        
        # Reset index to get datetime as a column
        df_export.reset_index(inplace=True)
        
        # Rename the index column to 'datetime' if it's not already named
        if df_export.index.name is None:
            # Check if first column after reset is the datetime index
            if len(df_export.columns) > 5:
                # The index was reset, so the first column should be datetime
                first_col = df_export.columns[0]
                if first_col not in ['open', 'high', 'low', 'close', 'volume']:
                    df_export.rename(columns={first_col: 'datetime'}, inplace=True)
        
        # If datetime column doesn't exist, create it from index
        if 'datetime' not in df_export.columns:
            # Create datetime column from sequential timestamps
            start_time = pd.Timestamp.now() - pd.Timedelta(minutes=len(df_export))
            df_export['datetime'] = pd.date_range(
                start=start_time,
                periods=len(df_export),
                freq='1min'
            )
        
        # Format datetime column
        df_export['datetime'] = pd.to_datetime(df_export['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save as CSV
        df_export.to_csv(
            filepath,
            index=False,
            columns=['datetime', 'open', 'high', 'low', 'close', 'volume']
        )
        
        print(success(f"[SYNTHETIC] Saved {len(df)} bars to {filepath}"))


def generate_synthetic_training_data(
    output_dir: str = "data/raw",
    instrument: str = "ES",
    timeframes: List[int] = [1, 5],
    n_bars_per_tf: Dict[int, int] = None,
    seed: int = 42
):
    """
    Generate synthetic data files for supervised pre-training.
    
    Args:
        output_dir: Directory to save files
        instrument: Instrument symbol
        timeframes: List of timeframes to generate
        n_bars_per_tf: Number of bars per timeframe (default: 10000 for 1min, scaled for others)
        seed: Random seed for reproducibility
    """
    if n_bars_per_tf is None:
        n_bars_per_tf = {
            1: 10000,   # 10k bars of 1min data
            5: 5000,    # 5k bars of 5min data (resampled from 1min)
            15: 2000    # 2k bars of 15min data
        }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(info("\n" + "="*70))
    print(info("GENERATING SYNTHETIC TRAINING DATA"))
    print(info("="*70))
    
    generator = SyntheticDataGenerator(
        base_price=4000.0,
        volatility=0.01,
        trend_strength=0.0005,
        seed=seed
    )
    
    # Generate 1min data first (base)
    print(info(f"\n[SYNTHETIC] Generating 1min data ({n_bars_per_tf.get(1, 10000)} bars)..."))
    df_1min = generator.generate_mixed_pattern_data(
        n_bars=n_bars_per_tf.get(1, 10000),
        pattern_length=200
    )
    
    # Save 1min data
    filepath_1min = output_path / f"{instrument}_1min.csv"
    generator.save_to_file(df_1min, filepath_1min, instrument, 1)
    
    # Generate higher timeframes by resampling
    for tf in timeframes:
        if tf == 1:
            continue  # Already generated
        
        print(info(f"\n[SYNTHETIC] Resampling to {tf}min data..."))
        
        # Resample 1min data to higher timeframe
        df_resampled = df_1min.copy()
        
        # Ensure index is datetime
        if not isinstance(df_resampled.index, pd.DatetimeIndex):
            df_resampled.index = pd.to_datetime(df_resampled.index)
        
        # Resample OHLCV
        resampled = df_resampled.resample(f'{tf}min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Limit to requested number of bars
        n_bars = n_bars_per_tf.get(tf, len(resampled))
        if len(resampled) > n_bars:
            resampled = resampled.iloc[:n_bars]
        
        # Save resampled data
        filepath_tf = output_path / f"{instrument}_{tf}min.csv"
        generator.save_to_file(resampled, filepath_tf, instrument, tf)
    
    print(success("\n[SYNTHETIC] Synthetic data generation complete!"))
    print(info("="*70 + "\n"))


if __name__ == "__main__":
    # Generate synthetic data for testing
    generate_synthetic_training_data(
        output_dir="data/raw",
        instrument="ES",
        timeframes=[1, 5],
        n_bars_per_tf={1: 10000, 5: 5000},
        seed=42
    )

