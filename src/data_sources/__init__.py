"""
Data Sources Module

Provides data access for:
- Historical market data (NT8 exports)
- Live market data (NT8 bridge)
- Sentiment data (free APIs: NewsAPI, Reddit)
- Economic calendar data
"""

from src.data_sources.market_data import MarketDataProvider
from src.data_sources.sentiment_sources import SentimentDataProvider
from src.data_sources.cache import DataCache

__all__ = ["MarketDataProvider", "SentimentDataProvider", "DataCache"]

