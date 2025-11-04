"""
Data Cache for Cost Optimization

Implements in-memory caching with TTL to reduce API calls.
"""

from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import threading
import hashlib
import json


class DataCache:
    """
    In-memory cache with TTL support.
    
    Reduces API calls by caching:
    - Market data (1 minute TTL)
    - Sentiment data (15 minutes TTL)
    - LLM responses (2 minutes TTL for similar queries)
    """
    
    def __init__(self, default_ttl: int = 300):
        """
        Initialize cache.
        
        Args:
            default_ttl: Default TTL in seconds (default: 5 minutes)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._default_ttl = timedelta(seconds=default_ttl)
        self._lock = threading.RLock()
    
    def _make_key(self, *args, **kwargs) -> str:
        """
        Create cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Cache key string
        """
        # Create hash from serialized arguments
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set cache value.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        with self._lock:
            expires_at = datetime.now() + timedelta(seconds=ttl or self._default_ttl.total_seconds())
            
            self._cache[key] = {
                "value": value,
                "expires_at": expires_at,
                "created_at": datetime.now()
            }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get cache value.
        
        Args:
            key: Cache key
            default: Default value if not found or expired
        
        Returns:
            Cached value or default
        """
        with self._lock:
            if key not in self._cache:
                return default
            
            entry = self._cache[key]
            
            # Check if expired
            if datetime.now() > entry["expires_at"]:
                # Expired - remove and return default
                del self._cache[key]
                return default
            
            return entry["value"]
    
    def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            pattern: Optional pattern to match keys (clears all if None)
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            if pattern is None:
                count = len(self._cache)
                self._cache.clear()
                return count
            
            # Clear matching keys
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self._cache[key]
            
            return len(keys_to_remove)
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            now = datetime.now()
            expired_keys = [
                key for key, entry in self._cache.items()
                if now > entry["expires_at"]
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache stats
        """
        with self._lock:
            now = datetime.now()
            total = len(self._cache)
            expired = sum(1 for e in self._cache.values() if now > e["expires_at"])
            active = total - expired
            
            return {
                "total_entries": total,
                "active_entries": active,
                "expired_entries": expired,
                "cache_size_mb": 0  # Would calculate actual size if needed
            }

