"""
Swarm Orchestrator

Orchestrates the multi-agent swarm for market analysis.
Coordinates execution of all agents with proper handoff logic.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import json
import threading
import time
from pathlib import Path
from src.agentic_swarm.shared_context import SharedContext
from src.agentic_swarm.config_loader import SwarmConfigLoader
from src.agentic_swarm.agents import (
    MarketResearchAgent,
    SentimentAgent,
    ContrarianAgent,
    ElliottWaveAgent,
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
            
            # Kong Gateway configuration
            use_kong = reasoning_config.get("use_kong", False)
            kong_api_key = reasoning_config.get("kong_api_key") or os.getenv("KONG_API_KEY")
            
            self.reasoning_engine = ReasoningEngine(
                provider_type=reasoning_config.get("provider", "ollama"),
                model=reasoning_config.get("model", "deepseek-r1:8b"),
                api_key=api_key,
                base_url=reasoning_config.get("base_url"),
                timeout=int(reasoning_config.get("timeout", 2.0) * 60),
                keep_alive=reasoning_config.get("keep_alive", "10m"),  # Keep model pre-loaded
                use_kong=use_kong,
                kong_api_key=kong_api_key
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
        
        # Adaptive Learning Agent periodic execution
        self.adaptive_learning_task: Optional[asyncio.Task] = None
        self.adaptive_learning_running = False
    
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
        contrarian_config = dict(self.swarm_config.get("contrarian", {}))
        contrarian_config["reasoning"] = reasoning_config  # Include reasoning config
        override = self._load_contrarian_enabled_setting()
        if override is not None:
            contrarian_config["enabled"] = override
        self.contrarian_enabled = contrarian_config.get("enabled", True)
        
        if self.contrarian_enabled:
            self.contrarian_agent = ContrarianAgent(
                shared_context=self.shared_context,
                market_data_provider=self.market_data_provider,
                reasoning_engine=self.reasoning_engine,
                config=contrarian_config
            )
        else:
            self.contrarian_agent = None
        
        # Elliott Wave Agent
        elliott_config = dict(self.swarm_config.get("elliott_wave", {}))
        elliott_config["environment_defaults"] = {
            "instrument": self.config.get("environment", {}).get("instrument", "ES"),
            "timeframes": self.config.get("environment", {}).get("timeframes", [1, 5, 15]),
        }
        self.elliott_wave_enabled = elliott_config.get("enabled", True)
        
        if self.elliott_wave_enabled:
            self.elliott_wave_agent = ElliottWaveAgent(
                shared_context=self.shared_context,
                market_data_provider=self.market_data_provider,
                config=elliott_config
            )
        else:
            self.elliott_wave_agent = None
        
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
        
        # Adaptive Learning Agent (runs independently, continuously)
        adaptive_learning_config = self.swarm_config.get("adaptive_learning", {})
        adaptive_learning_config["reasoning"] = reasoning_config  # Include reasoning config
        self.adaptive_learning_enabled = adaptive_learning_config.get("enabled", False)
        
        if self.adaptive_learning_enabled:
            from src.agentic_swarm.agents.adaptive_learning_agent import AdaptiveLearningAgent
            self.adaptive_learning_agent = AdaptiveLearningAgent(
                shared_context=self.shared_context,
                reasoning_engine=self.reasoning_engine,
                config=adaptive_learning_config
            )
            print("[OK] Adaptive Learning Agent enabled (runs independently)")
        else:
            self.adaptive_learning_agent = None
        
        # Store agent list (main workflow agents only - adaptive learning runs separately)
        self.agents = [
            self.market_research_agent,
            self.sentiment_agent,
        ]
        if self.contrarian_agent:
            self.agents.append(self.contrarian_agent)
        if self.elliott_wave_agent:
            self.agents.append(self.elliott_wave_agent)
        self.agents.extend([
            self.analyst_agent,
            self.recommendation_agent
        ])
    
    def _load_contrarian_enabled_setting(self) -> Optional[bool]:
        """Load contrarian enabled flag from settings.json if present."""
        settings_file = Path("settings.json")
        if not settings_file.exists():
            return None
        
        try:
            with open(settings_file, "r") as f:
                settings = json.load(f)
            if "contrarian_enabled" in settings:
                return bool(settings["contrarian_enabled"])
        except Exception:
            pass
        
        return None

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
        parallel_tasks = [
            (
                "research",
                asyncio.create_task(
                    asyncio.to_thread(self.market_research_agent.analyze, market_state)
                ),
            ),
            (
                "sentiment",
                asyncio.create_task(
                    asyncio.to_thread(self.sentiment_agent.analyze, market_state)
                ),
            ),
        ]
        
        if self.contrarian_agent:
            parallel_tasks.append(
                (
                    "contrarian",
                    asyncio.create_task(
                        asyncio.to_thread(self.contrarian_agent.analyze, market_state)
                    ),
                )
            )
        else:
            self.shared_context.set("contrarian_analysis", None, "contrarian_signals")
        
        if self.elliott_wave_agent:
            parallel_tasks.append(
                (
                    "elliott_wave",
                    asyncio.create_task(
                        asyncio.to_thread(self.elliott_wave_agent.analyze, market_state)
                    ),
                )
            )
        else:
            self.shared_context.set("elliott_wave_analysis", None, "analysis_results")
        
        task_results = await asyncio.gather(*[task for _, task in parallel_tasks])
        task_map = {name: result for (name, _), result in zip(parallel_tasks, task_results)}
        
        research_findings = task_map.get("research")
        sentiment_findings = task_map.get("sentiment")
        contrarian_findings = task_map.get(
            "contrarian",
            {
                "status": "disabled",
                "message": "Contrarian agent disabled via settings",
                "timestamp": datetime.now().isoformat()
            }
        )
        elliott_findings = task_map.get(
            "elliott_wave",
            {
                "status": "disabled",
                "message": "Elliott Wave agent disabled via settings",
                "timestamp": datetime.now().isoformat()
            }
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
                "elliott_wave": elliott_findings,
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
    
    def start_adaptive_learning(self, performance_data_provider):
        """
        Start periodic adaptive learning analysis.
        
        Args:
            performance_data_provider: Callable that returns performance data dict
        """
        if not self.adaptive_learning_enabled or not self.adaptive_learning_agent:
            return
        
        self.adaptive_learning_running = True
        self.performance_data_provider = performance_data_provider
        
        # Start background thread for periodic analysis
        def run_adaptive_learning():
            while self.adaptive_learning_running:
                try:
                    # Get performance data
                    performance_data = performance_data_provider()
                    
                    if performance_data:
                        # Run analysis
                        result = self.adaptive_learning_agent.analyze(
                            market_state={},  # Can be enhanced to get from shared context
                            performance_data=performance_data
                        )
                        
                        if result.get("status") == "success":
                            recommendations = result.get("recommendations", {})
                            if recommendations.get("type") != "NO_CHANGE":
                                print(f"\n[ADAPTIVE LEARNING] Recommendation: {recommendations['type']}")
                                print(f"  Reasoning: {recommendations.get('reasoning', 'N/A')}")
                                print(f"  Parameters: {recommendations.get('parameters', {})}")
                                print(f"  Confidence: {recommendations.get('confidence', 0.0):.2f}")
                                print(f"  ⚠️  Requires manual approval\n")
                    
                    # Wait for next analysis (configurable frequency)
                    analysis_frequency = self.adaptive_learning_agent.analysis_frequency
                    time.sleep(analysis_frequency)
                    
                except Exception as e:
                    print(f"[ERROR] Adaptive learning analysis failed: {e}")
                    time.sleep(60)  # Wait 1 minute before retry
        
        # Start thread
        self.adaptive_learning_thread = threading.Thread(
            target=run_adaptive_learning,
            daemon=True,
            name="AdaptiveLearning"
        )
        self.adaptive_learning_thread.start()
        print("[OK] Adaptive Learning Agent started (periodic analysis)")
    
    def stop_adaptive_learning(self):
        """Stop periodic adaptive learning analysis"""
        self.adaptive_learning_running = False
        if hasattr(self, 'adaptive_learning_thread'):
            self.adaptive_learning_thread.join(timeout=5.0)
        print("[OK] Adaptive Learning Agent stopped")
    
    def get_adaptive_learning_recommendations(self) -> Optional[Dict[str, Any]]:
        """Get latest adaptive learning recommendations from shared context"""
        if not self.adaptive_learning_enabled:
            return None
        
        return self.shared_context.get("adaptive_learning_analysis", namespace="adaptive_learning")
    
    def apply_adaptive_learning_recommendation(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply an approved adaptive learning recommendation"""
        if not self.adaptive_learning_enabled or not self.adaptive_learning_agent:
            return {"status": "error", "message": "Adaptive learning not enabled"}
        
        return self.adaptive_learning_agent.apply_recommendation(recommendation)
