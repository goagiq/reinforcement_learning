"""
Simple Forecast Predictor

A lightweight forecasting module that provides basic price forecasts
for RL features. Works without external dependencies like Chronos.

This is a simplified implementation that can be enhanced later with
Chronos-Bolt or other forecasting models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Try to import torch for Chronos (optional)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ForecastResult:
    """Forecast result"""
    direction: float  # -1 to +1 (bearish to bullish)
    confidence: float  # 0-1 (confidence in forecast)
    expected_return: float  # Expected price change percentage (negative = price down, positive = price up)
    forecast_horizon: int  # Number of periods ahead


class SimpleForecastPredictor:
    """
    Simple forecast predictor using basic statistical methods.
    
    Methods:
    - Moving average momentum
    - Price trend analysis
    - Volume-weighted price direction
    """
    
    def __init__(
        self,
        lookback_window: int = 20,
        forecast_horizon: int = 5,
        cache_steps: int = 10  # Cache predictions for N steps to speed up training
    ):
        """
        Initialize simple forecast predictor.
        
        Args:
            lookback_window: Number of periods to look back for analysis
            forecast_horizon: Number of periods ahead to forecast
            cache_steps: Number of steps to cache predictions (default: 10 for speed)
        """
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.cache_steps = cache_steps
        self._cache = {}  # Cache: {cached_step: ForecastResult}
    
    def clear_cache(self):
        """Clear the forecast cache (call when starting a new episode)"""
        self._cache.clear()
    
    def predict(
        self,
        price_data: pd.DataFrame,
        current_step: Optional[int] = None
    ) -> ForecastResult:
        """
        Generate forecast from price data.
        
        Args:
            price_data: Historical price data (OHLCV)
            current_step: Current step index (if None, uses last row)
        
        Returns:
            ForecastResult with direction, confidence, and expected return
        """
        if len(price_data) < self.lookback_window:
            # Not enough data - return neutral forecast
            return ForecastResult(
                direction=0.0,
                confidence=0.0,
                expected_return=0.0,
                forecast_horizon=self.forecast_horizon
            )
        
        # Use last row if current_step not specified
        if current_step is None:
            current_step = len(price_data) - 1
        
        # Ensure we have enough data
        start_idx = max(0, current_step - self.lookback_window)
        end_idx = min(current_step + 1, len(price_data))
        recent_data = price_data.iloc[start_idx:end_idx]
        
        if len(recent_data) < 2:
            return ForecastResult(
                direction=0.0,
                confidence=0.0,
                expected_return=0.0,
                forecast_horizon=self.forecast_horizon
            )
        
        # Extract price series
        closes = recent_data['close'].values if 'close' in recent_data.columns else recent_data.iloc[:, -1].values
        
        # Calculate momentum indicators
        short_ma = np.mean(closes[-5:]) if len(closes) >= 5 else closes[-1]
        long_ma = np.mean(closes[-self.lookback_window:]) if len(closes) >= self.lookback_window else closes[-1]
        
        # Price trend
        price_change = closes[-1] - closes[0] if len(closes) > 1 else 0.0
        price_change_pct = price_change / closes[0] if closes[0] != 0 else 0.0
        
        # Momentum
        momentum = short_ma - long_ma
        momentum_pct = momentum / long_ma if long_ma != 0 else 0.0
        
        # Volume-weighted direction (if volume available)
        volume_weighted_direction = 0.0
        if 'volume' in recent_data.columns and len(recent_data) > 1:
            volumes = recent_data['volume'].values
            price_changes = np.diff(closes)
            if len(price_changes) > 0 and np.sum(volumes[:-1]) > 0:
                volume_weighted_direction = np.sum(price_changes * volumes[:-1]) / np.sum(volumes[:-1])
                volume_weighted_direction = volume_weighted_direction / closes[0] if closes[0] != 0 else 0.0
        
        # Combine signals
        # Direction: -1 (bearish) to +1 (bullish)
        direction = np.tanh(momentum_pct * 10 + price_change_pct * 5 + volume_weighted_direction * 3)
        
        # Confidence: based on signal strength and consistency
        signal_strength = abs(momentum_pct) + abs(price_change_pct)
        consistency = 1.0 - abs(direction - np.tanh(momentum_pct * 10))  # How consistent are signals?
        confidence = min(1.0, signal_strength * 2 + consistency * 0.5)
        
        # Expected return: simple projection based on momentum
        expected_return = momentum_pct * self.forecast_horizon * 0.5  # Scale by horizon
        
        return ForecastResult(
            direction=float(direction),
            confidence=float(confidence),
            expected_return=float(expected_return),
            forecast_horizon=self.forecast_horizon
        )
    
    def get_forecast_features(self, price_data: pd.DataFrame, current_step: Optional[int] = None) -> List[float]:
        """
        Get forecast features for RL state vector.
        
        Uses caching to avoid recalculating on every step (speeds up training).
        
        Returns:
            List of 3 features: [direction, confidence, expected_return]
            - direction: -1 to +1 (bearish to bullish signal)
            - confidence: 0-1 (confidence in forecast)
            - expected_return: Expected price change % (negative = price down, positive = price up)
            
        Note: expected_return is the expected price movement, not the trade return.
        For a sell signal (negative direction), expected_return will be negative (price down),
        and the RL agent should learn to go SHORT to profit from this.
        """
        if current_step is None:
            current_step = len(price_data) - 1
        
        # Check cache: only recalculate every N steps
        # Calculate which cache interval this step belongs to
        cached_step = (current_step // self.cache_steps) * self.cache_steps
        
        # Check if we have a cached forecast for this interval
        if cached_step in self._cache:
            # Use cached result (forecast calculated at the start of this cache interval)
            forecast = self._cache[cached_step]
        else:
            # Recalculate for the cached step (not current_step) to ensure consistency
            # This ensures all steps in the same cache interval use the same forecast
            forecast = self.predict(price_data, cached_step)
            self._cache[cached_step] = forecast
            # Clean old cache entries (keep only last 10)
            if len(self._cache) > 10:
                oldest_key = min(self._cache.keys())
                del self._cache[oldest_key]
        
        return [
            forecast.direction,      # -1 to +1
            forecast.confidence,     # 0-1
            forecast.expected_return  # Percentage
        ]


# Optional: Chronos-Bolt integration (if available)
try:
    from chronos import ChronosPipeline
    
    class ChronosForecastPredictor(SimpleForecastPredictor):
        """
        Chronos-Bolt based forecast predictor.
        
        Uses Amazon's Chronos-Bolt model for more accurate forecasts.
        Falls back to SimpleForecastPredictor if Chronos is not available.
        """
        
        def __init__(
            self,
            lookback_window: int = 20,
            forecast_horizon: int = 5,
            model_name: str = "amazon/chronos-t5-tiny",
            cache_steps: int = 20  # Default: cache for 20 steps (Chronos is slower, so cache longer)
        ):
            """
            Initialize Chronos predictor.
            
            Args:
                lookback_window: Number of periods to look back
                forecast_horizon: Number of periods ahead to forecast
                model_name: Chronos model name (default: tiny for speed)
                cache_steps: Number of steps to cache predictions (default: 20 for Chronos)
            """
            super().__init__(lookback_window, forecast_horizon, cache_steps=cache_steps)
            try:
                if not TORCH_AVAILABLE:
                    raise ImportError("PyTorch not available")
                self.chronos_pipeline = ChronosPipeline.from_pretrained(model_name)
                self.use_chronos = True
                print(f"[OK] Chronos model '{model_name}' loaded successfully")
            except Exception as e:
                print(f"[WARN] Chronos not available, using simple predictor: {e}")
                self.chronos_pipeline = None
                self.use_chronos = False
        
        def predict(
            self,
            price_data: pd.DataFrame,
            current_step: Optional[int] = None
        ) -> ForecastResult:
            """
            Generate forecast using Chronos if available, otherwise fallback to simple.
            """
            if not self.use_chronos or self.chronos_pipeline is None:
                return super().predict(price_data, current_step)
            
            try:
                # Prepare data for Chronos
                if current_step is None:
                    current_step = len(price_data) - 1
                
                start_idx = max(0, current_step - self.lookback_window)
                end_idx = min(current_step + 1, len(price_data))
                recent_data = price_data.iloc[start_idx:end_idx]
                
                if len(recent_data) < self.lookback_window:
                    return super().predict(price_data, current_step)
                
                # Extract close prices
                closes = recent_data['close'].values if 'close' in recent_data.columns else recent_data.iloc[:, -1].values
                
                # Convert to tensor format for Chronos
                # Chronos expects shape: (batch_size, context_length)
                context_tensor = torch.tensor(
                    closes[-self.lookback_window:].reshape(1, -1),
                    dtype=torch.float32
                )
                
                # Generate forecast
                # ChronosPipeline.predict() takes 'inputs' not 'context'
                forecast_values = self.chronos_pipeline.predict(
                    inputs=context_tensor,
                    prediction_length=self.forecast_horizon
                )
                
                # Extract forecast (result is a torch.Tensor with shape [batch, context_length, prediction_length])
                if isinstance(forecast_values, torch.Tensor):
                    forecast_array = forecast_values[0].cpu().numpy() if forecast_values.is_cuda else forecast_values[0].numpy()
                elif isinstance(forecast_values, (list, tuple)) and len(forecast_values) > 0:
                    forecast_array = forecast_values[0].cpu().numpy() if hasattr(forecast_values[0], 'is_cuda') and forecast_values[0].is_cuda else forecast_values[0].numpy()
                else:
                    forecast_array = np.array(forecast_values) if isinstance(forecast_values, np.ndarray) else np.array([forecast_values])
                
                # Chronos returns shape [batch, context_length, prediction_length]
                # We want the forecast for the last context point, which is the future prediction
                if len(forecast_array.shape) == 3:
                    # Shape: [batch, context_length, prediction_length]
                    # Take the last context point's forecast (the actual future prediction)
                    forecast_series = forecast_array[-1, :]  # Shape: [prediction_length]
                elif len(forecast_array.shape) == 2:
                    # Shape: [context_length, prediction_length] or [batch, prediction_length]
                    forecast_series = forecast_array[-1, :] if forecast_array.shape[0] > 1 else forecast_array[0, :]
                else:
                    # Shape: [prediction_length] or flat array
                    forecast_series = forecast_array.flatten()
                
                # Use the last prediction point as the forecast price
                current_price = closes[-1]
                forecast_price = forecast_series[-1] if len(forecast_series) > 0 else current_price
                
                direction = np.tanh((forecast_price - current_price) / current_price * 10) if current_price != 0 else 0.0
                expected_return = (forecast_price - current_price) / current_price * 100 if current_price != 0 else 0.0
                
                # Confidence: based on forecast variance (lower variance = higher confidence)
                if len(forecast_series) > 1:
                    forecast_std = np.std(forecast_series)
                    price_range = np.max(forecast_series) - np.min(forecast_series)
                    confidence = 1.0 / (1.0 + forecast_std / (price_range + 1e-6))
                else:
                    confidence = 0.5
                
                return ForecastResult(
                    direction=float(direction),
                    confidence=float(confidence),
                    expected_return=float(expected_return),
                    forecast_horizon=self.forecast_horizon
                )
            except Exception as e:
                print(f"[WARN] Chronos forecast failed, using simple predictor: {e}")
                return super().predict(price_data, current_step)
    
    # Export Chronos predictor as default if available
    ForecastPredictor = ChronosForecastPredictor
    
except ImportError:
    # Chronos not available - use simple predictor
    ForecastPredictor = SimpleForecastPredictor

