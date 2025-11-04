"""
Sentiment Data Provider

Provides market and economic sentiment from free sources.
Supports: NewsAPI, Reddit (via PRAW), and other free APIs.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import requests
import os
from src.data_sources.cache import DataCache


class SentimentDataProvider:
    """
    Provides sentiment data from free sources.
    
    Supported sources:
    - NewsAPI (free tier)
    - Reddit (via PRAW, free)
    - Economic calendar (free sources)
    """
    
    def __init__(self, config: Dict, cache: Optional[DataCache] = None):
        """
        Initialize sentiment data provider.
        
        Args:
            config: Configuration with API keys and settings
            cache: Optional cache instance
        """
        self.config = config
        self.cache = cache or DataCache()
        
        # API keys (from config or environment)
        self.newsapi_key = config.get("newsapi_key") or os.getenv("NEWSAPI_KEY")
        
        # Configuration
        self.sources = config.get("sources", ["newsapi"])
        self.sentiment_window = config.get("sentiment_window", 3600)  # 1 hour
        
        # Base URLs
        self.newsapi_url = "https://newsapi.org/v2/everything"
    
    def get_news_sentiment(
        self,
        query: str = "futures market",
        hours_back: int = 24
    ) -> Dict[str, Any]:
        """
        Get news sentiment from NewsAPI.
        
        Args:
            query: Search query
            hours_back: Hours to look back
        
        Returns:
            Dict with sentiment data
        """
        # Check cache
        cache_key = f"news_sentiment_{query}_{hours_back}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        if not self.newsapi_key:
            return {
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "articles_count": 0,
                "error": "NewsAPI key not configured"
            }
        
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(hours=hours_back)
            
            params = {
                "q": query,
                "from": from_date.strftime("%Y-%m-%dT%H:%M:%S"),
                "to": to_date.strftime("%Y-%m-%dT%H:%M:%S"),
                "sortBy": "popularity",
                "pageSize": 50,
                "apiKey": self.newsapi_key,
                "language": "en"
            }
            
            response = requests.get(self.newsapi_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get("articles", [])
            
            # Simple sentiment (positive/negative keywords)
            # In production, would use LLM for better sentiment analysis
            positive_keywords = ["surge", "rally", "gain", "rise", "bullish", "up", "positive", "growth"]
            negative_keywords = ["drop", "fall", "decline", "bearish", "down", "negative", "crash", "loss"]
            
            sentiment_score = 0.0
            total_articles = len(articles)
            
            if total_articles == 0:
                return {
                    "sentiment_score": 0.0,
                    "confidence": 0.0,
                    "articles_count": 0,
                    "source": "newsapi"
                }
            
            for article in articles:
                title = article.get("title", "").lower()
                description = article.get("description", "").lower()
                text = f"{title} {description}"
                
                positive_count = sum(1 for word in positive_keywords if word in text)
                negative_count = sum(1 for word in negative_keywords if word in text)
                
                if positive_count > negative_count:
                    sentiment_score += 0.1
                elif negative_count > positive_count:
                    sentiment_score -= 0.1
            
            # Normalize to -1 to +1
            sentiment_score = max(-1.0, min(1.0, sentiment_score / total_articles))
            
            result = {
                "sentiment_score": sentiment_score,
                "confidence": min(1.0, total_articles / 50.0),  # More articles = higher confidence
                "articles_count": total_articles,
                "source": "newsapi",
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache for 15 minutes
            self.cache.set(cache_key, result, ttl=900)
            
            return result
            
        except requests.exceptions.RequestException as e:
            return {
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "articles_count": 0,
                "error": str(e),
                "source": "newsapi"
            }
    
    def get_market_sentiment(
        self,
        instruments: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get overall market sentiment.
        
        Args:
            instruments: List of instruments to analyze (default: ES, NQ, RTY, YM)
        
        Returns:
            Dict with aggregated sentiment
        """
        if instruments is None:
            instruments = ["ES", "NQ", "RTY", "YM"]
        
        # Check cache
        cache_key = f"market_sentiment_{'_'.join(instruments)}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Get sentiment for each instrument
        sentiments = {}
        for instrument in instruments:
            query = f"{instrument} futures"
            sentiment = self.get_news_sentiment(query, hours_back=24)
            sentiments[instrument] = sentiment
        
        # Aggregate sentiment
        scores = [s.get("sentiment_score", 0.0) for s in sentiments.values()]
        confidences = [s.get("confidence", 0.0) for s in sentiments.values()]
        
        if scores:
            avg_sentiment = sum(scores) / len(scores)
            avg_confidence = sum(confidences) / len(confidences)
        else:
            avg_sentiment = 0.0
            avg_confidence = 0.0
        
        result = {
            "overall_sentiment": avg_sentiment,
            "confidence": avg_confidence,
            "instrument_sentiments": sentiments,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache for 15 minutes
        self.cache.set(cache_key, result, ttl=900)
        
        return result
    
    def get_economic_sentiment(self) -> Dict[str, Any]:
        """
        Get economic calendar sentiment.
        
        Returns:
            Dict with economic sentiment indicators
        """
        # Placeholder for economic calendar integration
        # Would integrate with free economic calendar APIs
        
        # For now, return neutral sentiment
        return {
            "sentiment_score": 0.0,
            "confidence": 0.0,
            "upcoming_events": [],
            "source": "economic_calendar",
            "timestamp": datetime.now().isoformat()
        }

