"""
Configuration Loader for Agentic Swarm

Loads and validates swarm configuration from YAML files.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml


class SwarmConfigLoader:
    """Loads and validates swarm configuration."""
    
    @staticmethod
    def load_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load swarm configuration from main config dict.
        
        Args:
            config: Main configuration dictionary
        
        Returns:
            Swarm configuration dictionary with defaults
        """
        swarm_config = config.get("agentic_swarm", {})
        
        # Set defaults
        defaults = {
            "enabled": True,
            "provider": "ollama",
            "max_handoffs": 10,
            "max_iterations": 15,
            "execution_timeout": 20.0,
            "node_timeout": 5.0,
            "cache_ttl": 300,
            "market_research": {
                "instruments": ["ES", "NQ", "RTY", "YM"],
                "correlation_window": 20,
                "divergence_threshold": 0.1
            },
            "sentiment": {
                "sources": ["newsapi"],
                "newsapi_key": None,
                "sentiment_window": 3600
            },
            "elliott_wave": {
                "enabled": True,
                "instrument": "ES",
                "timeframes": [1, 5, 15],
                "lookback_bars": 400,
                "min_bars": 120,
                "min_confidence": 0.55,
                "swing_threshold": 0.003,
                "position_multiplier": 0.6,
                "max_position_size": 0.8
            },
            "analyst": {
                "deep_reasoning": True,
                "conflict_detection": True
            },
            "recommendation": {
                "risk_integration": True,
                "position_sizing": True
            }
        }
        
        # Merge with defaults
        merged = defaults.copy()
        merged.update(swarm_config)
        
        # Deep merge nested configs
        for key in ["market_research", "sentiment", "elliott_wave", "analyst", "recommendation"]:
            if key in swarm_config:
                merged[key] = {**defaults.get(key, {}), **swarm_config[key]}
        
        return merged
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate swarm configuration.
        
        Args:
            config: Configuration to validate
        
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(config, dict):
            return False, "Configuration must be a dictionary"
        
        required_fields = ["max_handoffs", "max_iterations", "execution_timeout"]
        for field in required_fields:
            if field not in config:
                return False, f"Missing required field: {field}"
        
        # Validate timeouts
        if config["execution_timeout"] <= 0:
            return False, "execution_timeout must be positive"
        
        if config.get("node_timeout", 0) <= 0:
            return False, "node_timeout must be positive"
        
        if config["execution_timeout"] < config.get("node_timeout", 0):
            return False, "execution_timeout must be >= node_timeout"
        
        # Validate agent configs
        for agent_key in ["market_research", "sentiment", "analyst", "recommendation"]:
            if agent_key in config and not isinstance(config[agent_key], dict):
                return False, f"{agent_key} config must be a dictionary"
        
        return True, None

