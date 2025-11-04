"""
Base Agent Wrapper for Strands Agents

Provides common functionality for all swarm agents.
"""

from typing import Dict, Any, Optional, List
import os
from strands import Agent
from strands.models.anthropic import AnthropicModel
from src.reasoning_engine import ReasoningEngine
from src.agentic_swarm.shared_context import SharedContext


class BaseSwarmAgent:
    """
    Base class for all swarm agents.
    
    Provides:
    - LLM provider integration
    - Shared context access
    - Common utilities
    """
    
    def __init__(
        self,
        name: str,
        system_prompt: str,
        shared_context: SharedContext,
        reasoning_engine: Optional[ReasoningEngine] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base agent.
        
        Args:
            name: Agent name
            system_prompt: System prompt for agent
            shared_context: Shared context instance
            reasoning_engine: Optional reasoning engine (uses default if None)
            config: Optional configuration
        """
        self.name = name
        self.shared_context = shared_context
        self.config = config or {}
        
        # Initialize reasoning engine if not provided
        if reasoning_engine is None:
            # Get provider config from shared context or config
            provider_config = self.config.get("reasoning", {})
            import os
            api_key = provider_config.get("api_key") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("GROK_API_KEY")
            
            self.reasoning_engine = ReasoningEngine(
                provider_type=provider_config.get("provider", "ollama"),
                model=provider_config.get("model", "deepseek-r1:8b"),
                api_key=api_key,
                base_url=provider_config.get("base_url"),
                timeout=int(provider_config.get("timeout", 2.0) * 60)
            )
        else:
            self.reasoning_engine = reasoning_engine
        
        # Get model configuration from config
        provider_config = self.config.get("reasoning", {})
        model_name = provider_config.get("model", "claude-sonnet-4-20250514")
        
        # Get API key from environment or config
        api_key = provider_config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        
        # Allow skipping API key check ONLY in test/mock mode (for unit tests)
        # WARNING: This should NEVER be True in production
        skip_api_check = provider_config.get("skip_api_check", False)
        
        # Safety check: Warn if skip_api_check is used outside of test environment
        if skip_api_check:
            import sys
            is_test_env = "pytest" in sys.modules or "unittest" in sys.modules or "test" in sys.argv[0].lower()
            if not is_test_env:
                import warnings
                warnings.warn(
                    "skip_api_check is enabled but not in test environment! "
                    "This should only be used for unit tests. API calls will fail.",
                    UserWarning,
                    stacklevel=2
                )
        
        if not api_key and not skip_api_check:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment or config. "
                "Set ANTHROPIC_API_KEY environment variable or provide in config."
            )
        
        # Use dummy key ONLY for mocked tests (will fail on real API calls)
        if not api_key:
            if skip_api_check:
                api_key = "test-api-key"  # Dummy key for mocked tests
            else:
                raise ValueError("API key is required for production use")
        
        # Create Strands Anthropic model
        # Note: AnthropicModel uses client_args for API key and model_id for model selection
        anthropic_model = AnthropicModel(
            client_args={
                "api_key": api_key,
            },
            model_id=model_name,
            max_tokens=provider_config.get("max_tokens", 4096),
            params={
                "temperature": provider_config.get("temperature", 0.7),
            }
        )
        
        # Create Strands Agent with Anthropic model
        self.agent = Agent(
            name=name,
            system_prompt=system_prompt,
            model=anthropic_model
        )
        
        # Agent description for swarm coordination
        self.description = self.config.get("description", "")
    
    def get_context_summary(self) -> str:
        """
        Get formatted context summary for handoff messages.
        
        Returns:
            Formatted context string
        """
        history = self.shared_context.get_agent_history()
        market_data = self.shared_context.get_all("market_data")
        research = self.shared_context.get_all("research_findings")
        sentiment = self.shared_context.get_all("sentiment_scores")
        
        summary = f"Shared Context:\n"
        summary += f"- Market Data: {len(market_data)} instruments\n"
        summary += f"- Research Findings: {len(research)} items\n"
        summary += f"- Sentiment Scores: {len(sentiment)} items\n"
        
        if history:
            summary += f"\nAgent History:\n"
            for entry in history[-5:]:  # Last 5 entries
                summary += f"- {entry['agent']}: {entry['action']}\n"
        
        return summary
    
    def log_action(self, action: str, result: Any) -> None:
        """
        Log agent action to shared context.
        
        Args:
            action: Action performed
            result: Result or summary
        """
        self.shared_context.add_agent_history(self.name, action, result)
    
    def get_agent(self) -> Agent:
        """Get the underlying Strands Agent instance."""
        return self.agent

