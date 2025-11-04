"""
Cost Tracker for Agentic Swarm

Tracks API calls, LLM usage, and costs for optimization.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import threading


@dataclass
class CostEntry:
    """Individual cost entry."""
    timestamp: datetime
    agent_name: str
    operation: str
    provider: str
    model: str
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0


class CostTracker:
    """
    Tracks costs for swarm operations.
    
    Monitors:
    - LLM API calls (tokens, cost)
    - External API calls (NewsAPI, etc.)
    - Execution time per agent
    """
    
    # LLM provider costs (per 1K tokens)
    # These are approximate - adjust based on actual provider pricing
    PROVIDER_COSTS = {
        "ollama": {
            "input": 0.0,  # Free (local)
            "output": 0.0
        },
        "deepseek_cloud": {
            "input": 0.0001,  # $0.0001 per 1K tokens
            "output": 0.0002
        },
        "grok": {
            "input": 0.001,  # $0.001 per 1K tokens
            "output": 0.003
        }
    }
    
    def __init__(self):
        """Initialize cost tracker."""
        self.entries: List[CostEntry] = []
        self._lock = threading.RLock()
        self._start_time = datetime.now()
    
    def log_llm_call(
        self,
        agent_name: str,
        provider: str,
        model: str,
        tokens_input: int,
        tokens_output: int,
        duration_seconds: float
    ):
        """
        Log an LLM API call.
        
        Args:
            agent_name: Name of agent making the call
            provider: LLM provider (ollama, deepseek_cloud, grok)
            model: Model name
            tokens_input: Input tokens
            tokens_output: Output tokens
            duration_seconds: Call duration
        """
        with self._lock:
            # Calculate cost
            cost = 0.0
            if provider in self.PROVIDER_COSTS:
                cost = (
                    (tokens_input / 1000.0) * self.PROVIDER_COSTS[provider]["input"] +
                    (tokens_output / 1000.0) * self.PROVIDER_COSTS[provider]["output"]
                )
            
            entry = CostEntry(
                timestamp=datetime.now(),
                agent_name=agent_name,
                operation="llm_call",
                provider=provider,
                model=model,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost_usd=cost,
                duration_seconds=duration_seconds
            )
            
            self.entries.append(entry)
    
    def log_api_call(
        self,
        agent_name: str,
        operation: str,
        cost_usd: float = 0.0,
        duration_seconds: float = 0.0
    ):
        """
        Log an external API call.
        
        Args:
            agent_name: Name of agent making the call
            operation: Operation type (newsapi, etc.)
            cost_usd: Cost in USD
            duration_seconds: Call duration
        """
        with self._lock:
            entry = CostEntry(
                timestamp=datetime.now(),
                agent_name=agent_name,
                operation=operation,
                provider="external",
                model="",
                cost_usd=cost_usd,
                duration_seconds=duration_seconds
            )
            
            self.entries.append(entry)
    
    def get_total_cost(self, hours: Optional[int] = None) -> float:
        """
        Get total cost.
        
        Args:
            hours: Optional time window in hours (None = all time)
        
        Returns:
            Total cost in USD
        """
        with self._lock:
            cutoff = None
            if hours:
                cutoff = datetime.now() - timedelta(hours=hours)
            
            total = 0.0
            for entry in self.entries:
                if cutoff is None or entry.timestamp >= cutoff:
                    total += entry.cost_usd
            
            return total
    
    def get_cost_by_agent(self, hours: Optional[int] = None) -> Dict[str, float]:
        """
        Get cost breakdown by agent.
        
        Args:
            hours: Optional time window in hours
        
        Returns:
            Dict of agent_name -> total_cost
        """
        with self._lock:
            cutoff = None
            if hours:
                cutoff = datetime.now() - timedelta(hours=hours)
            
            costs = {}
            for entry in self.entries:
                if cutoff is None or entry.timestamp >= cutoff:
                    costs[entry.agent_name] = costs.get(entry.agent_name, 0.0) + entry.cost_usd
            
            return costs
    
    def get_cost_by_provider(self, hours: Optional[int] = None) -> Dict[str, float]:
        """
        Get cost breakdown by provider.
        
        Args:
            hours: Optional time window in hours
        
        Returns:
            Dict of provider -> total_cost
        """
        with self._lock:
            cutoff = None
            if hours:
                cutoff = datetime.now() - timedelta(hours=hours)
            
            costs = {}
            for entry in self.entries:
                if cutoff is None or entry.timestamp >= cutoff:
                    costs[entry.provider] = costs.get(entry.provider, 0.0) + entry.cost_usd
            
            return costs
    
    def get_api_call_count(self, hours: Optional[int] = None) -> int:
        """Get total API call count."""
        with self._lock:
            cutoff = None
            if hours:
                cutoff = datetime.now() - timedelta(hours=hours)
            
            count = 0
            for entry in self.entries:
                if cutoff is None or entry.timestamp >= cutoff:
                    count += 1
            
            return count
    
    def get_statistics(self, hours: Optional[int] = None) -> Dict:
        """
        Get comprehensive statistics.
        
        Args:
            hours: Optional time window in hours
        
        Returns:
            Dict with statistics
        """
        with self._lock:
            cutoff = None
            if hours:
                cutoff = datetime.now() - timedelta(hours=hours)
            
            filtered_entries = [
                e for e in self.entries
                if cutoff is None or e.timestamp >= cutoff
            ]
            
            if not filtered_entries:
                return {
                    "total_cost": 0.0,
                    "total_calls": 0,
                    "avg_cost_per_call": 0.0,
                    "total_duration": 0.0,
                    "avg_duration": 0.0
                }
            
            total_cost = sum(e.cost_usd for e in filtered_entries)
            total_duration = sum(e.duration_seconds for e in filtered_entries)
            
            return {
                "total_cost": total_cost,
                "total_calls": len(filtered_entries),
                "avg_cost_per_call": total_cost / len(filtered_entries),
                "total_duration": total_duration,
                "avg_duration": total_duration / len(filtered_entries),
                "cost_by_agent": self.get_cost_by_agent(hours),
                "cost_by_provider": self.get_cost_by_provider(hours),
                "time_window_hours": hours
            }
    
    def clear(self):
        """Clear all entries."""
        with self._lock:
            self.entries.clear()
            self._start_time = datetime.now()

