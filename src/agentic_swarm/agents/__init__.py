"""
Agentic Swarm Agents

Specialized agents for market analysis:
- Market Research Agent: Correlation analysis
- Sentiment Agent: Market sentiment gathering
- Contrarian Agent: Greed/fear detection (Warren Buffett philosophy)
- Analyst Agent: Deep reasoning and synthesis
- Recommendation Agent: Final trading recommendations
"""

from src.agentic_swarm.agents.market_research_agent import MarketResearchAgent
from src.agentic_swarm.agents.sentiment_agent import SentimentAgent
from src.agentic_swarm.agents.contrarian_agent import ContrarianAgent
from src.agentic_swarm.agents.analyst_agent import AnalystAgent
from src.agentic_swarm.agents.recommendation_agent import RecommendationAgent

__all__ = [
    "MarketResearchAgent",
    "SentimentAgent",
    "ContrarianAgent",
    "AnalystAgent",
    "RecommendationAgent"
]

