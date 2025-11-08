"""
Volatility Prediction Module

Provides realized volatility forecasting for adaptive risk management.
Uses historical data and statistical models to predict future volatility,
enabling dynamic position sizing and stop-loss placement.

Veteran Futures Trader Approach:
- Predicts volatility for next N periods
- Adjusts position sizing based on predicted volatility
- Sets adaptive stop-losses (wider when volatility is high)
- Assesses overnight gap risk based on volatility forecasts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class VolatilityForecast:
    """Volatility forecast result"""
    current_volatility: float  # Current realized volatility
    predicted_volatility: float  # Predicted volatility for next period
    predicted_volatility_5period: float  # Predicted volatility for next 5 periods
    predicted_volatility_20period: float  # Predicted volatility for next 20 periods
    volatility_trend: str  # "increasing", "decreasing", "stable"
    confidence: float  # Confidence in prediction (0-1)
    volatility_percentile: float  # Current volatility percentile (0-100)
    gap_risk_probability: float  # Probability of significant overnight gap
    recommendations: Dict[str, str]  # Trading recommendations based on volatility


class VolatilityPredictor:
    """
    Volatility Predictor for futures trading.
    
    Predicts future volatility using multiple methods:
    - Historical volatility (simple moving average)
    - GARCH-based volatility (if scipy available)
    - Exponential weighted moving average (EWMA)
    - Volatility clustering (regime detection)
    """
    
    def __init__(
        self,
        lookback_periods: int = 252,  # Trading days for historical analysis
        prediction_horizon: int = 1,  # Periods ahead to predict
        volatility_window: int = 20  # Window for realized volatility calculation
    ):
        """
        Initialize volatility predictor.
        
        Args:
            lookback_periods: Periods to use for historical analysis
            prediction_horizon: Number of periods ahead to predict
            volatility_window: Window size for realized volatility calculation
        """
        self.lookback_periods = lookback_periods
        self.prediction_horizon = prediction_horizon
        self.volatility_window = volatility_window
    
    def calculate_realized_volatility(
        self,
        price_data: pd.DataFrame,
        method: str = "standard_deviation"
    ) -> pd.Series:
        """
        Calculate realized volatility from price data.
        
        Args:
            price_data: Historical price data (must have 'close' column)
            method: 'standard_deviation', 'parkinson', 'garman_klass', 'rogers_satchell'
        
        Returns:
            Series of realized volatility values
        """
        if len(price_data) < 2:
            return pd.Series([0.02] * len(price_data))  # Default 2% volatility
        
        # Calculate returns
        returns = price_data['close'].pct_change().dropna()
        
        if len(returns) < self.volatility_window:
            # Not enough data, use simple volatility
            return pd.Series([returns.std()] * len(price_data))
        
        if method == "standard_deviation":
            # Simple rolling standard deviation
            realized_vol = returns.rolling(window=self.volatility_window).std()
        elif method == "parkinson":
            # Parkinson estimator (uses high/low, more efficient)
            if 'high' in price_data.columns and 'low' in price_data.columns:
                hl_ratio = np.log(price_data['high'] / price_data['low'])
                realized_vol = hl_ratio.rolling(window=self.volatility_window).std() * np.sqrt(1 / (4 * np.log(2)))
            else:
                realized_vol = returns.rolling(window=self.volatility_window).std()
        elif method == "ewma":
            # Exponential weighted moving average (more responsive to recent changes)
            realized_vol = returns.ewm(span=self.volatility_window, adjust=False).std()
        else:
            # Default to standard deviation
            realized_vol = returns.rolling(window=self.volatility_window).std()
        
        # Fill NaN values
        realized_vol = realized_vol.fillna(method='bfill').fillna(0.02)
        
        # Annualize (assuming daily data, 252 trading days)
        realized_vol_annualized = realized_vol * np.sqrt(252)
        
        return realized_vol_annualized
    
    def predict_volatility(
        self,
        price_data: pd.DataFrame,
        method: str = "adaptive"
    ) -> VolatilityForecast:
        """
        Predict future volatility.
        
        Args:
            price_data: Historical price data
            method: 'adaptive', 'ewma', 'historical_mean', 'garch'
        
        Returns:
            VolatilityForecast with predictions and recommendations
        """
        if len(price_data) < self.volatility_window:
            # Not enough data
            return self._default_forecast()
        
        # Calculate current realized volatility
        current_vol = self.calculate_realized_volatility(price_data, method="standard_deviation")
        current_vol_value = float(current_vol.iloc[-1] if len(current_vol) > 0 else 0.02)
        
        # Calculate historical volatility for context
        historical_vol = self.calculate_realized_volatility(price_data, method="standard_deviation")
        
        # Predict using selected method
        if method == "adaptive":
            # Adaptive method: combines multiple approaches
            prediction = self._predict_adaptive(historical_vol, current_vol_value)
        elif method == "ewma":
            # Exponential weighted moving average
            prediction = self._predict_ewma(historical_vol)
        elif method == "historical_mean":
            # Simple mean reversion
            prediction = self._predict_mean_reversion(historical_vol)
        else:
            # Default to adaptive
            prediction = self._predict_adaptive(historical_vol, current_vol_value)
        
        # Calculate volatility trend
        if len(historical_vol) >= 5:
            recent_avg = historical_vol.tail(5).mean()
            older_avg = historical_vol.tail(20).head(15).mean() if len(historical_vol) >= 20 else recent_avg
            if recent_avg > older_avg * 1.1:
                trend = "increasing"
            elif recent_avg < older_avg * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        # Calculate confidence (based on consistency of recent volatility)
        if len(historical_vol) >= 10:
            recent_vol_std = historical_vol.tail(10).std()
            recent_vol_mean = historical_vol.tail(10).mean()
            if recent_vol_mean > 0:
                consistency = 1.0 - min(recent_vol_std / recent_vol_mean, 1.0)
                confidence = max(consistency, 0.3)  # Minimum 30% confidence
            else:
                confidence = 0.5
        else:
            confidence = 0.5
        
        # Calculate volatility percentile
        if len(historical_vol) >= self.lookback_periods:
            vol_percentile = float((historical_vol < current_vol_value).sum() / len(historical_vol) * 100)
        else:
            vol_percentile = 50.0  # Default to median
        
        # Predict volatility for multiple horizons
        predicted_vol_1period = prediction
        predicted_vol_5period = self._predict_horizon(historical_vol, horizon=5)
        predicted_vol_20period = self._predict_horizon(historical_vol, horizon=20)
        
        # Estimate overnight gap risk
        gap_risk = self._estimate_gap_risk(current_vol_value, historical_vol)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            current_vol=current_vol_value,
            predicted_vol=predicted_vol_1period,
            trend=trend,
            percentile=vol_percentile
        )
        
        return VolatilityForecast(
            current_volatility=current_vol_value,
            predicted_volatility=predicted_vol_1period,
            predicted_volatility_5period=predicted_vol_5period,
            predicted_volatility_20period=predicted_vol_20period,
            volatility_trend=trend,
            confidence=confidence,
            volatility_percentile=vol_percentile,
            gap_risk_probability=gap_risk,
            recommendations=recommendations
        )
    
    def _predict_adaptive(
        self,
        historical_vol: pd.Series,
        current_vol: float
    ) -> float:
        """Adaptive prediction combining multiple methods"""
        if len(historical_vol) < 10:
            return current_vol
        
        # Weight recent volatility more heavily
        recent_vol = historical_vol.tail(10).mean()
        long_term_vol = historical_vol.mean()
        
        # Mean reversion component
        mean_reversion_target = long_term_vol
        
        # Trend component (if volatility is trending)
        if len(historical_vol) >= 5:
            recent_trend = historical_vol.tail(5).mean() - historical_vol.tail(20).head(15).mean() if len(historical_vol) >= 20 else 0
        else:
            recent_trend = 0
        
        # Combine: 60% recent, 30% mean reversion, 10% trend
        prediction = (
            0.6 * recent_vol +
            0.3 * mean_reversion_target +
            0.1 * (current_vol + recent_trend)
        )
        
        return float(np.clip(prediction, 0.01, 2.0))  # Clamp to reasonable range
    
    def _predict_ewma(self, historical_vol: pd.Series) -> float:
        """Predict using Exponential Weighted Moving Average"""
        if len(historical_vol) < 5:
            return float(historical_vol.iloc[-1] if len(historical_vol) > 0 else 0.02)
        
        # EWMA with alpha=0.3 (more responsive to recent changes)
        ewma = historical_vol.ewm(alpha=0.3, adjust=False).mean()
        return float(ewma.iloc[-1])
    
    def _predict_mean_reversion(self, historical_vol: pd.Series) -> float:
        """Predict using mean reversion"""
        if len(historical_vol) < 10:
            return float(historical_vol.mean() if len(historical_vol) > 0 else 0.02)
        
        # Mean reverts toward long-term average
        current = historical_vol.iloc[-1]
        mean_vol = historical_vol.mean()
        
        # Partial mean reversion (70% toward mean)
        prediction = 0.3 * current + 0.7 * mean_vol
        
        return float(prediction)
    
    def _predict_horizon(self, historical_vol: pd.Series, horizon: int) -> float:
        """Predict volatility for longer horizons"""
        if len(historical_vol) < 10:
            return float(historical_vol.mean() if len(historical_vol) > 0 else 0.02)
        
        # For longer horizons, volatility tends to mean-revert
        current = historical_vol.iloc[-1]
        mean_vol = historical_vol.mean()
        
        # More mean reversion for longer horizons
        reversion_rate = min(horizon / 20.0, 0.8)  # Up to 80% reversion
        prediction = (1 - reversion_rate) * current + reversion_rate * mean_vol
        
        return float(np.clip(prediction, 0.01, 2.0))
    
    def _estimate_gap_risk(
        self,
        current_vol: float,
        historical_vol: pd.Series
    ) -> float:
        """
        Estimate probability of significant overnight gap.
        
        For futures trading, overnight gaps are critical risk.
        """
        if len(historical_vol) < 20:
            # Default gap risk based on volatility level
            if current_vol > 0.5:
                return 0.15  # 15% chance of gap
            elif current_vol > 0.3:
                return 0.10  # 10% chance
            else:
                return 0.05  # 5% chance
        
        # Calculate gap risk based on volatility regime
        # Higher volatility = higher gap risk
        vol_percentile = float((historical_vol < current_vol).sum() / len(historical_vol))
        
        # Base gap risk
        base_risk = 0.05  # 5% base
        
        # Increase risk if volatility is high
        if vol_percentile > 0.8:  # Top 20% volatility
            gap_risk = base_risk + 0.15  # Up to 20% gap risk
        elif vol_percentile > 0.6:  # Top 40% volatility
            gap_risk = base_risk + 0.10  # Up to 15% gap risk
        else:
            gap_risk = base_risk + 0.05  # Up to 10% gap risk
        
        return float(np.clip(gap_risk, 0.0, 0.25))  # Cap at 25%
    
    def _generate_recommendations(
        self,
        current_vol: float,
        predicted_vol: float,
        trend: str,
        percentile: float
    ) -> Dict[str, str]:
        """Generate trading recommendations based on volatility"""
        recommendations = {}
        
        # Position sizing recommendation
        if predicted_vol > current_vol * 1.2:
            recommendations["position_sizing"] = "REDUCE - Volatility expected to increase"
        elif predicted_vol < current_vol * 0.8:
            recommendations["position_sizing"] = "INCREASE - Volatility expected to decrease"
        else:
            recommendations["position_sizing"] = "MAINTAIN - Volatility stable"
        
        # Stop loss recommendation
        if percentile > 80:  # High volatility percentile
            recommendations["stop_loss"] = "WIDEN - High volatility environment, use wider stops"
        elif percentile < 30:  # Low volatility percentile
            recommendations["stop_loss"] = "TIGHTEN - Low volatility, tighter stops acceptable"
        else:
            recommendations["stop_loss"] = "STANDARD - Use normal stop loss distances"
        
        # Trading frequency recommendation
        if trend == "increasing" and percentile > 70:
            recommendations["trading_frequency"] = "REDUCE - Volatility increasing, reduce trading"
        elif trend == "decreasing" and percentile < 40:
            recommendations["trading_frequency"] = "INCREASE - Volatility decreasing, safe to trade more"
        else:
            recommendations["trading_frequency"] = "NORMAL - Continue normal trading frequency"
        
        # Risk management recommendation
        if percentile > 85:
            recommendations["risk_management"] = "HIGH_ALERT - Very high volatility, maximum caution"
        elif percentile > 70:
            recommendations["risk_management"] = "CAUTION - Elevated volatility, be cautious"
        else:
            recommendations["risk_management"] = "NORMAL - Standard risk management"
        
        return recommendations
    
    def _default_forecast(self) -> VolatilityForecast:
        """Return default forecast when insufficient data"""
        return VolatilityForecast(
            current_volatility=0.02,
            predicted_volatility=0.02,
            predicted_volatility_5period=0.02,
            predicted_volatility_20period=0.02,
            volatility_trend="stable",
            confidence=0.3,
            volatility_percentile=50.0,
            gap_risk_probability=0.05,
            recommendations={
                "position_sizing": "MAINTAIN - Insufficient data",
                "stop_loss": "STANDARD - Insufficient data",
                "trading_frequency": "NORMAL - Insufficient data",
                "risk_management": "NORMAL - Insufficient data"
            }
        )
    
    def get_adaptive_position_multiplier(
        self,
        base_position: float,
        volatility_forecast: VolatilityForecast
    ) -> float:
        """
        Calculate adaptive position size multiplier based on volatility.
        
        Args:
            base_position: Base position size (from RL agent)
            volatility_forecast: Volatility forecast
        
        Returns:
            Adjusted position multiplier (0.0 to 1.0)
        """
        # Reduce position size when volatility is high
        vol_percentile = volatility_forecast.volatility_percentile
        
        if vol_percentile > 90:
            # Top 10% volatility - reduce position by 50%
            multiplier = 0.5
        elif vol_percentile > 80:
            # Top 20% volatility - reduce position by 30%
            multiplier = 0.7
        elif vol_percentile > 70:
            # Top 30% volatility - reduce position by 15%
            multiplier = 0.85
        elif vol_percentile < 30:
            # Bottom 30% volatility - can increase position by 10%
            multiplier = 1.1
        else:
            # Normal volatility - no adjustment
            multiplier = 1.0
        
        # Further adjust if volatility is increasing
        if volatility_forecast.volatility_trend == "increasing":
            multiplier *= 0.9  # Reduce by additional 10%
        elif volatility_forecast.volatility_trend == "decreasing" and vol_percentile < 50:
            multiplier *= 1.05  # Increase by 5% if volatility decreasing
        
        return float(np.clip(multiplier, 0.3, 1.2))  # Clamp between 30% and 120%
    
    def get_adaptive_stop_loss_multiplier(
        self,
        base_stop_distance: float,
        volatility_forecast: VolatilityForecast
    ) -> float:
        """
        Calculate adaptive stop loss distance multiplier based on volatility.
        
        Args:
            base_stop_distance: Base stop loss distance (e.g., 2% of price)
            volatility_forecast: Volatility forecast
        
        Returns:
            Adjusted stop loss multiplier
        """
        # Widen stops when volatility is high
        vol_percentile = volatility_forecast.volatility_percentile
        
        if vol_percentile > 80:
            # High volatility - widen stops by 50%
            multiplier = 1.5
        elif vol_percentile > 60:
            # Above average volatility - widen stops by 25%
            multiplier = 1.25
        elif vol_percentile < 30:
            # Low volatility - can tighten stops by 15%
            multiplier = 0.85
        else:
            # Normal volatility - standard stops
            multiplier = 1.0
        
        return float(np.clip(multiplier, 0.7, 2.0))  # Clamp between 70% and 200%


def predict_volatility(
    price_data: pd.DataFrame,
    lookback_periods: int = 252,
    method: str = "adaptive"
) -> Dict:
    """
    Convenience function for quick volatility prediction.
    
    Args:
        price_data: Historical price data
        lookback_periods: Periods for historical analysis
        method: Prediction method
    
    Returns:
        Dictionary with volatility forecast
    """
    predictor = VolatilityPredictor(lookback_periods=lookback_periods)
    forecast = predictor.predict_volatility(price_data, method=method)
    
    return {
        "current_volatility": forecast.current_volatility,
        "predicted_volatility": forecast.predicted_volatility,
        "predicted_volatility_5period": forecast.predicted_volatility_5period,
        "predicted_volatility_20period": forecast.predicted_volatility_20period,
        "volatility_trend": forecast.volatility_trend,
        "confidence": forecast.confidence,
        "volatility_percentile": forecast.volatility_percentile,
        "gap_risk_probability": forecast.gap_risk_probability,
        "recommendations": forecast.recommendations
    }

