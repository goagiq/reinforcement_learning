"""
Swarm Orchestrator

Orchestrates the multi-agent swarm for market analysis.
Coordinates execution of all agents with proper handoff logic.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from src.agentic_swarm.shared_context import SharedContext
from src.agentic_swarm.config_loader import SwarmConfigLoader
from src.agentic_swarm.agents import (
    MarketResearchAgent,
    SentimentAgent,
    ContrarianAgent,
    AnalystAgent,
    RecommendationAgent
)
from src.data_sources.market_data import MarketDataProvider
from src.data_sources.sentiment_sources import SentimentDataProvider
from src.data_sources.cache import DataCache
from src.reasoning_engine import ReasoningEngine
from src.risk_manager import RiskManager
from src.agentic_swarm.cost_tracker import CostTracker


class SwarmOrchestrator:
    """
    Orchestrates the agentic swarm for market analysis.
    
    Coordinates:
    - Market Research Agent (correlation analysis)
    - Sentiment Agent (sentiment analysis)
    - Analyst Agent (synthesis + deep reasoning)
    - Recommendation Agent (final decision)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        reasoning_engine: Optional[ReasoningEngine] = None,
        risk_manager: Optional[RiskManager] = None
    ):
        """
        Initialize swarm orchestrator.
        
        Args:
            config: Main configuration dictionary
            reasoning_engine: Optional reasoning engine (creates default if None)
            risk_manager: Optional risk manager (creates default if None)
        """
        self.config = config
        self.swarm_config = SwarmConfigLoader.load_from_config(config)
        
        # Validate config
        is_valid, error = SwarmConfigLoader.validate_config(self.swarm_config)
        if not is_valid:
            raise ValueError(f"Invalid swarm configuration: {error}")
        
        # Initialize shared context
        self.shared_context = SharedContext(
            ttl_seconds=self.swarm_config.get("cache_ttl", 300)
        )
        
        # Initialize data cache
        self.data_cache = DataCache(default_ttl=self.swarm_config.get("cache_ttl", 300))
        
        # Initialize cost tracker
        self.cost_tracker = CostTracker()
        
        # Initialize data providers
        self._init_data_providers()
        
        # Initialize reasoning engine if not provided
        if reasoning_engine is None:
            reasoning_config = config.get("reasoning", {})
            import os
            api_key = reasoning_config.get("api_key") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("GROK_API_KEY")
            
            self.reasoning_engine = ReasoningEngine(
                provider_type=reasoning_config.get("provider", "ollama"),
                model=reasoning_config.get("model", "deepseek-r1:8b"),
                api_key=api_key,
                base_url=reasoning_config.get("base_url"),
                timeout=int(reasoning_config.get("timeout", 2.0) * 60),
                keep_alive=reasoning_config.get("keep_alive", "10m")  # Keep model pre-loaded
            )
        else:
            self.reasoning_engine = reasoning_engine
        
        # Initialize risk manager if not provided
        if risk_manager is None:
            from src.risk_manager import RiskManager
            risk_config = config.get("risk_management", {})
            self.risk_manager = RiskManager(risk_config)
        else:
            self.risk_manager = risk_manager
        
        # Initialize agents
        self._init_agents()
        
        # Status tracking
        self.last_execution_time: Optional[float] = None
        self.last_result: Optional[Dict[str, Any]] = None
        self.execution_count = 0
    
    def _init_data_providers(self):
        """Initialize data providers."""
        # Market data provider
        market_config = {
            "data_path": self.config.get("data_path", "data"),
            "instruments": self.swarm_config.get("market_research", {}).get("instruments", ["ES", "NQ", "RTY", "YM"]),
            "timeframes": self.config.get("timeframes", [1, 5, 15])
        }
        self.market_data_provider = MarketDataProvider(market_config, cache=self.data_cache)
        
        # Sentiment data provider
        sentiment_config = self.swarm_config.get("sentiment", {})
        self.sentiment_provider = SentimentDataProvider(sentiment_config, cache=self.data_cache)
    
    def _init_agents(self):
        """Initialize all agents."""
        # Get reasoning config to pass to agents (includes keep_alive for Ollama)
        reasoning_config = self.config.get("reasoning", {})
        
        # Market Research Agent
        market_research_config = self.swarm_config.get("market_research", {})
        market_research_config["reasoning"] = reasoning_config  # Include reasoning config
        self.market_research_agent = MarketResearchAgent(
            shared_context=self.shared_context,
            market_data_provider=self.market_data_provider,
            reasoning_engine=self.reasoning_engine,
            config=market_research_config
        )
        
        # Sentiment Agent
        sentiment_config = self.swarm_config.get("sentiment", {})
        sentiment_config["reasoning"] = reasoning_config  # Include reasoning config
        self.sentiment_agent = SentimentAgent(
            shared_context=self.shared_context,
            sentiment_provider=self.sentiment_provider,
            reasoning_engine=self.reasoning_engine,
            config=sentiment_config
        )
        
        # Contrarian Agent (runs parallel with MarketResearch and Sentiment)
        contrarian_config = self.swarm_config.get("contrarian", {})
        contrarian_config["reasoning"] = reasoning_config  # Include reasoning config
        self.contrarian_agent = ContrarianAgent(
            shared_context=self.shared_context,
            market_data_provider=self.market_data_provider,
            reasoning_engine=self.reasoning_engine,
            config=contrarian_config
        )
        
        # Analyst Agent
        analyst_config = self.swarm_config.get("analyst", {})
        analyst_config["reasoning"] = reasoning_config  # Include reasoning config
        self.analyst_agent = AnalystAgent(
            shared_context=self.shared_context,
            reasoning_engine=self.reasoning_engine,
            config=analyst_config
        )
        
        # Recommendation Agent
        recommendation_config = self.swarm_config.get("recommendation", {})
        recommendation_config["reasoning"] = reasoning_config  # Include reasoning config
        self.recommendation_agent = RecommendationAgent(
            shared_context=self.shared_context,
            risk_manager=self.risk_manager,
            reasoning_engine=self.reasoning_engine,
            config=recommendation_config
        )
        
        # Store agent list
        self.agents = [
            self.market_research_agent,
            self.sentiment_agent,
            self.contrarian_agent,
            self.analyst_agent,
            self.recommendation_agent
        ]
    
    async def analyze(
        self,
        market_data: Dict[str, Any],
        rl_recommendation: Optional[Dict[str, Any]] = None,
        current_position: float = 0.0,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run swarm analysis asynchronously.
        
        Execution flow:
        1. Market Research Agent (parallel with Sentiment Agent)
        2. Sentiment Agent (parallel with Market Research Agent)
        3. Analyst Agent (waits for both above)
        4. Recommendation Agent (waits for Analyst)
        
        Args:
            market_data: Current market data
            rl_recommendation: Optional RL agent recommendation for context
            current_position: Current position size
            timeout: Execution timeout (uses config default if None)
        
        Returns:
            Dict with swarm analysis results
        """
        start_time = datetime.now()
        timeout = timeout or self.swarm_config.get("execution_timeout", 20.0)
        self.execution_count += 1
        
        try:
            # Store market data in shared context
            self.shared_context.set("current_market_data", market_data, "market_data")
            if rl_recommendation:
                self.shared_context.set("rl_recommendation", rl_recommendation, "general")
            
            # Create market state for agents
            market_state = {
                "price_data": market_data.get("price_data", {}),
                "volume_data": market_data.get("volume_data", {}),
                "indicators": market_data.get("indicators", {}),
                "market_regime": market_data.get("market_regime", "unknown"),
                "timestamp": market_data.get("timestamp", datetime.now().isoformat())
            }
            
            # Execute swarm with timeout
            try:
                result = await asyncio.wait_for(
                    self._execute_swarm(market_state, rl_recommendation, current_position),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                return {
                    "status": "timeout",
                    "error": f"Swarm execution exceeded timeout of {timeout}s",
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "timestamp": datetime.now().isoformat()
                }
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            result["execution_time"] = execution_time
            result["execution_count"] = self.execution_count
            
            self.last_execution_time = execution_time
            self.last_result = result
            
            return result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            self.last_result = error_result
            return error_result
    
    async def _execute_swarm(
        self,
        market_state: Dict[str, Any],
        rl_recommendation: Optional[Dict[str, Any]],
        current_position: float
    ) -> Dict[str, Any]:
        """
        Execute swarm agents in sequence with proper handoffs.
        
        Args:
            market_state: Current market state
            rl_recommendation: RL recommendation
            current_position: Current position
        
        Returns:
            Final result with all agent outputs
        """
        # Phase 1: Parallel execution of Market Research, Sentiment, and Contrarian agents
        research_task = asyncio.create_task(
            asyncio.to_thread(self.market_research_agent.analyze, market_state)
        )
        sentiment_task = asyncio.create_task(
            asyncio.to_thread(self.sentiment_agent.analyze, market_state)
        )
        contrarian_task = asyncio.create_task(
            asyncio.to_thread(self.contrarian_agent.analyze, market_state)
        )
        
        # Wait for all three to complete
        research_findings, sentiment_findings, contrarian_findings = await asyncio.gather(
            research_task,
            sentiment_task,
            contrarian_task
        )
        
        # Phase 2: Analyst Agent (synthesizes research + sentiment)
        analyst_task = asyncio.create_task(
            asyncio.to_thread(
                self.analyst_agent.analyze,
                market_state,
                rl_recommendation
            )
        )
        analyst_analysis = await analyst_task
        
        # Phase 3: Recommendation Agent (final decision)
        recommendation_task = asyncio.create_task(
            asyncio.to_thread(
                self.recommendation_agent.recommend,
                market_state,
                rl_recommendation,
                current_position
            )
        )
        final_recommendation = await recommendation_task
        
        # Compile final result
        result = {
            "status": "success",
            "recommendation": final_recommendation,
            "analysis": {
                "research": research_findings,
                "sentiment": sentiment_findings,
                "contrarian": contrarian_findings,
                "analyst": analyst_analysis
            },
            "shared_context": {
                "agent_history": self.shared_context.get_agent_history(),
                "context_summary": self.shared_context.to_dict()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def analyze_sync(
        self,
        market_data: Dict[str, Any],
        rl_recommendation: Optional[Dict[str, Any]] = None,
        current_position: float = 0.0,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run swarm analysis synchronously (wrapper for async).
        
        Args:
            market_data: Current market data
            rl_recommendation: Optional RL agent recommendation
            current_position: Current position size
            timeout: Execution timeout
        
        Returns:
            Dict with swarm analysis results
        """
        # Use asyncio.run() for proper event loop handling
        # This creates a new event loop and runs the async function
        return asyncio.run(
            self.analyze(market_data, rl_recommendation, current_position, timeout)
        )
    
    def is_enabled(self) -> bool:
        """Check if swarm is enabled."""
        return self.swarm_config.get("enabled", True)
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "enabled": self.is_enabled(),
            "agents_initialized": len(self.agents) > 0,
            "agent_count": len(self.agents),
            "execution_count": self.execution_count,
            "last_execution_time": self.last_execution_time,
            "last_result_status": self.last_result.get("status") if self.last_result else None,
            "shared_context_stats": self.shared_context.get_all("metadata"),
            "cache_stats": self.data_cache.get_stats(),
            "cost_statistics": self.cost_tracker.get_statistics(hours=1)
        }
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self.data_cache.clear()
        self.shared_context.clear()
    
    def get_agent_history(self) -> list:
        """Get agent execution history."""
        return self.shared_context.get_agent_history()
