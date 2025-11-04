"""
Market Research Agent

Analyzes correlation between ES, NQ, RTY, YM futures to understand market dynamics
and provide correlation-based insights for trading decisions.
"""

from typing import Dict, Any, Optional, List
from strands import Agent
from src.agentic_swarm.base_agent import BaseSwarmAgent
from src.agentic_swarm.shared_context import SharedContext
from src.data_sources.market_data import MarketDataProvider
from src.reasoning_engine import ReasoningEngine


class MarketResearchAgent(BaseSwarmAgent):
    """
    Market Research Agent for correlation analysis.
    
    Responsibilities:
    - Calculate correlation matrices between instruments
    - Identify divergence/convergence patterns
    - Analyze volume relationships
    - Detect regime changes
    """
    
    def __init__(
        self,
        shared_context: SharedContext,
        market_data_provider: MarketDataProvider,
        reasoning_engine: Optional[ReasoningEngine] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Market Research Agent.
        
        Args:
            shared_context: Shared context instance
            market_data_provider: Market data provider instance
            reasoning_engine: Optional reasoning engine
            config: Optional configuration
        """
        config = config or {}
        system_prompt = """You are a market research specialist with expertise in futures market correlation analysis.

Your role is to:
1. Analyze correlation patterns between ES, NQ, RTY, and YM futures
2. Identify divergence and convergence signals
3. Detect market regime changes
4. Provide correlation-based insights for trading decisions

You have access to:
- Historical OHLCV data for all instruments
- Correlation calculation tools
- Divergence detection tools
- Statistical analysis capabilities

When analyzing:
- Focus on recent correlation patterns (last 20-100 bars)
- Identify when instruments are moving together vs diverging
- Note any significant correlation changes that might indicate regime shifts
- Provide clear, actionable insights

Format your analysis clearly with:
- Correlation scores (0.0 to 1.0)
- Divergence/convergence signals
- Confidence levels
- Key findings"""
        
        super().__init__(
            name="market_research",
            system_prompt=system_prompt,
            shared_context=shared_context,
            reasoning_engine=reasoning_engine,
            config=config
        )
        
        self.market_data_provider = market_data_provider
        self.instruments = config.get("instruments", ["ES", "NQ", "RTY", "YM"])
        self.correlation_window = config.get("correlation_window", 20)
        self.divergence_threshold = config.get("divergence_threshold", 0.1)
        
        # Add description for swarm coordination
        self.description = "Analyzes correlation patterns between futures instruments (ES, NQ, RTY, YM) to identify market dynamics and trading opportunities."
        
        # Create tools for the agent
        self._setup_tools()
    
    def _setup_tools(self):
        """Set up tools for the agent."""
        # Tools will be added to the Strands Agent
        # For now, we'll create Python functions that can be called
        
        def calculate_correlation(
            instrument1: str,
            instrument2: str,
            timeframe: int = 5
        ) -> Dict[str, Any]:
            """Calculate rolling correlation between two instruments."""
            try:
                corr = self.market_data_provider.get_rolling_correlation(
                    instrument1, instrument2, timeframe, self.correlation_window
                )
                
                if corr is None:
                    return {
                        "error": f"Could not calculate correlation between {instrument1} and {instrument2}",
                        "correlation": None
                    }
                
                # Store in shared context
                key = f"{instrument1}_{instrument2}_{timeframe}"
                self.shared_context.set(key, {
                    "correlation": corr,
                    "instrument1": instrument1,
                    "instrument2": instrument2,
                    "timeframe": timeframe
                }, "research_findings")
                
                return {
                    "correlation": corr,
                    "instrument1": instrument1,
                    "instrument2": instrument2,
                    "timeframe": timeframe,
                    "interpretation": self._interpret_correlation(corr)
                }
            except Exception as e:
                return {"error": str(e), "correlation": None}
        
        def get_correlation_matrix(timeframe: int = 5) -> Dict[str, Any]:
            """Get full correlation matrix for all instruments."""
            try:
                matrix = self.market_data_provider.get_correlation_matrix(timeframe, self.correlation_window)
                
                if matrix is None:
                    return {"error": "Could not calculate correlation matrix", "matrix": None}
                
                # Convert to dict for serialization
                matrix_dict = matrix.to_dict()
                
                # Store in shared context
                self.shared_context.set("correlation_matrix", matrix_dict, "research_findings")
                
                # Analyze matrix
                analysis = self._analyze_correlation_matrix(matrix)
                
                return {
                    "matrix": matrix_dict,
                    "analysis": analysis,
                    "timeframe": timeframe
                }
            except Exception as e:
                return {"error": str(e), "matrix": None}
        
        def detect_divergence(
            base_instrument: str = "ES",
            comparison_instruments: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """Detect divergence/convergence signals."""
            try:
                signal = self.market_data_provider.get_divergence_signal(
                    base_instrument,
                    comparison_instruments or [i for i in self.instruments if i != base_instrument],
                    timeframe=5,
                    threshold=self.divergence_threshold
                )
                
                # Store in shared context
                self.shared_context.set("divergence_signal", signal, "research_findings")
                
                return signal
            except Exception as e:
                return {"error": str(e), "signal": None}
        
        # Store tools for agent use
        # Note: In Phase 3, these will be properly registered with Strands Agent
        self.tools = {
            "calculate_correlation": calculate_correlation,
            "get_correlation_matrix": get_correlation_matrix,
            "detect_divergence": detect_divergence
        }
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation value."""
        abs_corr = abs(correlation)
        if abs_corr > 0.9:
            return "Very high correlation - instruments move together strongly"
        elif abs_corr > 0.7:
            return "High correlation - instruments generally move together"
        elif abs_corr > 0.5:
            return "Moderate correlation - some relationship exists"
        elif abs_corr > 0.3:
            return "Low correlation - limited relationship"
        else:
            return "Very low correlation - instruments move independently"
    
    def _analyze_correlation_matrix(self, matrix) -> Dict[str, Any]:
        """Analyze correlation matrix and extract insights."""
        import numpy as np
        
        # Convert to numpy array
        values = matrix.values
        np.fill_diagonal(values, np.nan)  # Ignore diagonal
        
        # Find strongest correlations
        max_corr = np.nanmax(values)
        min_corr = np.nanmin(values)
        avg_corr = np.nanmean(values)
        
        # Find pairs with strongest correlation
        max_idx = np.unravel_index(np.nanargmax(values), values.shape)
        min_idx = np.unravel_index(np.nanargmin(values), values.shape)
        
        instruments = matrix.index.tolist()
        
        return {
            "average_correlation": float(avg_corr),
            "max_correlation": float(max_corr),
            "min_correlation": float(min_corr),
            "strongest_pair": {
                "instruments": [instruments[max_idx[0]], instruments[max_idx[1]]],
                "correlation": float(max_corr)
            },
            "weakest_pair": {
                "instruments": [instruments[min_idx[0]], instruments[min_idx[1]]],
                "correlation": float(min_corr)
            },
            "market_regime": self._determine_regime(avg_corr)
        }
    
    def _determine_regime(self, avg_correlation: float) -> str:
        """Determine market regime based on average correlation."""
        if avg_correlation > 0.8:
            return "Highly correlated - all instruments moving together"
        elif avg_correlation > 0.6:
            return "Moderately correlated - general market trend"
        elif avg_correlation > 0.4:
            return "Mixed signals - some divergence present"
        else:
            return "Low correlation - independent movements, possible regime change"
    
    def analyze(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform market research analysis.
        
        Args:
            market_state: Current market state
        
        Returns:
            Dict with research findings
        """
        try:
            # Calculate correlation matrix
            correlation_matrix_result = self.tools["get_correlation_matrix"](timeframe=5)
            
            # Detect divergence
            divergence_result = self.tools["detect_divergence"]()
            
            # Store results
            findings = {
                "correlation_matrix": correlation_matrix_result,
                "divergence_signal": divergence_result,
                "timestamp": market_state.get("timestamp"),
                "instruments": self.instruments
            }
            
            self.shared_context.set("market_research_findings", findings, "research_findings")
            self.log_action("analyze_correlation", f"Analyzed {len(self.instruments)} instruments")
            
            return findings
            
        except Exception as e:
            error_result = {
                "error": str(e),
                "timestamp": market_state.get("timestamp")
            }
            self.log_action("analyze_correlation", f"Error: {str(e)}")
            return error_result

