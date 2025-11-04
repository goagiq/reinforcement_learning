"""
Shared Context for Agentic Swarm

Manages shared state and context between swarm agents.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import threading
import json


class SharedContext:
    """
    Shared context storage for swarm agents.
    
    Stores:
    - Market data (all instruments)
    - Research findings
    - Sentiment scores
    - Analysis results
    - Final recommendation
    """
    
    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize shared context.
        
        Args:
            ttl_seconds: Time-to-live for cached data (default: 5 minutes)
        """
        self._data: Dict[str, Any] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._ttl = timedelta(seconds=ttl_seconds)
        self._lock = threading.RLock()
        
        # Initialize context structure
        self._data = {
            "market_data": {},
            "research_findings": {},
            "sentiment_scores": {},
            "analysis_results": {},
            "recommendation": None,
            "agent_history": [],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
        }
    
    def set(self, key: str, value: Any, namespace: str = "general") -> None:
        """
        Set a value in shared context.
        
        Args:
            key: Key to store
            value: Value to store
            namespace: Optional namespace (default: "general")
        """
        with self._lock:
            if namespace not in self._data:
                self._data[namespace] = {}
            
            full_key = f"{namespace}.{key}"
            self._data[namespace][key] = value
            self._timestamps[full_key] = datetime.now()
            self._data["metadata"]["updated_at"] = datetime.now().isoformat()
    
    def get(self, key: str, namespace: str = "general", default: Any = None) -> Any:
        """
        Get a value from shared context.
        
        Args:
            key: Key to retrieve
            namespace: Optional namespace (default: "general")
            default: Default value if not found or expired
        
        Returns:
            Value or default if not found/expired
        """
        with self._lock:
            if namespace not in self._data:
                return default
            
            full_key = f"{namespace}.{key}"
            
            # Check if expired
            if full_key in self._timestamps:
                age = datetime.now() - self._timestamps[full_key]
                if age > self._ttl:
                    # Expired - remove and return default
                    self._data[namespace].pop(key, None)
                    self._timestamps.pop(full_key, None)
                    return default
            
            return self._data[namespace].get(key, default)
    
    def get_all(self, namespace: str = "general") -> Dict[str, Any]:
        """
        Get all values from a namespace.
        
        Args:
            namespace: Namespace to retrieve
        
        Returns:
            Dictionary of all values in namespace
        """
        with self._lock:
            # Clean expired entries
            self._clean_expired()
            return self._data.get(namespace, {}).copy()
    
    def add_agent_history(self, agent_name: str, action: str, result: Any) -> None:
        """
        Add entry to agent history.
        
        Args:
            agent_name: Name of agent
            action: Action performed
            result: Result or summary
        """
        with self._lock:
            if "agent_history" not in self._data:
                self._data["agent_history"] = []
            
            self._data["agent_history"].append({
                "agent": agent_name,
                "action": action,
                "result": str(result)[:200] if result else None,  # Truncate long results
                "timestamp": datetime.now().isoformat()
            })
    
    def get_agent_history(self) -> list:
        """Get agent execution history."""
        with self._lock:
            return self._data.get("agent_history", []).copy()
    
    def _clean_expired(self) -> None:
        """Clean expired entries from all namespaces."""
        now = datetime.now()
        expired_keys = []
        
        for full_key, timestamp in self._timestamps.items():
            if now - timestamp > self._ttl:
                expired_keys.append(full_key)
        
        for full_key in expired_keys:
            namespace, key = full_key.split(".", 1)
            self._data.get(namespace, {}).pop(key, None)
            self._timestamps.pop(full_key, None)
    
    def clear(self, namespace: Optional[str] = None) -> None:
        """
        Clear context data.
        
        Args:
            namespace: Optional namespace to clear (clears all if None)
        """
        with self._lock:
            if namespace:
                self._data.pop(namespace, None)
                # Remove timestamps for this namespace
                keys_to_remove = [k for k in self._timestamps.keys() if k.startswith(f"{namespace}.")]
                for k in keys_to_remove:
                    self._timestamps.pop(k, None)
            else:
                self._data.clear()
                self._timestamps.clear()
                # Reinitialize
                self.__init__(self._ttl.total_seconds())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        with self._lock:
            self._clean_expired()
            return json.loads(json.dumps(self._data, default=str))
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load context from dictionary."""
        with self._lock:
            self._data = data
            # Reset all timestamps to now
            now = datetime.now()
            for namespace in self._data.keys():
                if isinstance(self._data[namespace], dict):
                    for key in self._data[namespace].keys():
                        full_key = f"{namespace}.{key}"
                        self._timestamps[full_key] = now

