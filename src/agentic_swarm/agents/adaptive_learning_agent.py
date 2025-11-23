"""
Adaptive Learning Agent

Monitors trading performance over time and suggests parameter adjustments
for continuous learning and optimization. Works independently from training.

Key Features:
- Analyzes historical performance data
- Suggests R:R threshold adjustments
- Suggests quality filter adjustments
- Can recommend pausing/resuming trading
- Uses LLM reasoning (configurable)
- Requires manual approval for adjustments
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd

from src.agentic_swarm.base_agent import BaseSwarmAgent
from src.reasoning_engine import ReasoningEngine


class AdaptiveLearningAgent(BaseSwarmAgent):
    """
    Adaptive Learning Agent for continuous performance monitoring and optimization.
    
    This agent:
    - Monitors trading performance over time
    - Analyzes win rates, R:R ratios, trade quality
    - Suggests parameter adjustments (R:R thresholds, quality filters)
    - Can recommend pausing/resuming trading based on performance
    - All recommendations require manual approval
    """
    
    def __init__(
        self,
        shared_context,
        reasoning_engine: Optional[ReasoningEngine] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Adaptive Learning Agent.
        
        Args:
            shared_context: Shared context instance
            reasoning_engine: Optional reasoning engine (uses default if None)
            config: Optional configuration
        """
        system_prompt = self._get_system_prompt()
        
        super().__init__(
            name="AdaptiveLearningAgent",
            system_prompt=system_prompt,
            shared_context=shared_context,
            reasoning_engine=reasoning_engine,
            config=config
        )
        
        self.config = config or {}
        self.use_llm_reasoning = self.config.get("use_llm_reasoning", True)
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.last_analysis_time: Optional[datetime] = None
        self.analysis_frequency = self.config.get("analysis_frequency", 300)  # Every 5 minutes (300 seconds)
        
        # Adjustment thresholds
        self.min_trades_for_analysis = self.config.get("min_trades_for_analysis", 20)  # Need at least 20 trades
        self.min_analysis_window = self.config.get("min_analysis_window", 3600)  # 1 hour minimum
        
        # Performance thresholds
        self.min_win_rate = self.config.get("min_win_rate", 0.35)  # 35% minimum
        self.min_rr_ratio = self.config.get("min_rr_ratio", 1.3)  # 1.3:1 minimum
        self.max_drawdown_threshold = self.config.get("max_drawdown_threshold", 0.10)  # 10% max drawdown
        
        # Data storage paths
        self.performance_log_path = Path("logs/adaptive_learning/performance_history.jsonl")
        self.performance_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Current parameter state
        self.current_rr_threshold = self.config.get("initial_rr_threshold", 1.5)
        self.current_min_confidence = self.config.get("initial_min_confidence", 0.15)
        self.current_min_quality = self.config.get("initial_min_quality", 0.4)
        self.trading_paused = False
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for adaptive learning agent"""
        return """You are an Adaptive Learning Agent that monitors trading performance and suggests parameter adjustments for continuous optimization.

Your role:
1. Analyze historical trading performance data
2. Identify patterns in win rates, risk/reward ratios, and trade quality
3. Suggest parameter adjustments (R:R thresholds, quality filters)
4. Recommend pausing/resuming trading if performance degrades
5. Provide clear reasoning for all recommendations

Key metrics to monitor:
- Win rate (should be >= 35% minimum, target 60%+)
- Risk/Reward ratio (avg_win / avg_loss, should be >= 1.3:1 minimum, target 2.0:1+)
- Trade count per period (should be 0.5-1.0 trades/hour ideally)
- Maximum drawdown (should be < 10%)
- Average win vs average loss

Recommendation types:
1. ADJUST_RR_THRESHOLD: Adjust minimum risk/reward ratio threshold
2. ADJUST_QUALITY_FILTERS: Adjust min_action_confidence or min_quality_score
3. PAUSE_TRADING: Recommend pausing trading due to poor performance
4. RESUME_TRADING: Recommend resuming trading after pause
5. NO_CHANGE: No adjustments needed

Always provide:
- Clear reasoning based on data
- Specific parameter values
- Expected impact of adjustments
- Confidence level in recommendation"""
    
    def analyze(
        self,
        market_state: Dict[str, Any],
        performance_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze trading performance and suggest adjustments.
        
        Args:
            market_state: Current market state (from shared context)
            performance_data: Optional performance data (if not provided, reads from shared context)
        
        Returns:
            Dict with analysis results and recommendations
        """
        start_time = datetime.now()
        
        try:
            # Check if enough time has passed since last analysis
            if self.last_analysis_time:
                time_since_last = (start_time - self.last_analysis_time).total_seconds()
                if time_since_last < self.analysis_frequency:
                    # Too soon, return cached result if available
                    return self._get_cached_result()
            
            # Get performance data
            if performance_data is None:
                performance_data = self._get_performance_data()
            
            # Check if we have enough data
            if not self._has_sufficient_data(performance_data):
                return {
                    "status": "insufficient_data",
                    "message": f"Need at least {self.min_trades_for_analysis} trades and {self.min_analysis_window}s of data",
                    "current_trades": performance_data.get("total_trades", 0),
                    "timestamp": start_time.isoformat()
                }
            
            # Analyze performance
            analysis = self._analyze_performance(performance_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(analysis, performance_data)
            
            # Use LLM reasoning if enabled
            if self.use_llm_reasoning and recommendations:
                reasoning = self._generate_llm_reasoning(analysis, recommendations, performance_data)
                recommendations["reasoning"] = reasoning
            
            # Store in history
            self._store_analysis(analysis, recommendations, start_time)
            
            # Update last analysis time
            self.last_analysis_time = start_time
            
            result = {
                "status": "success",
                "agent": "AdaptiveLearningAgent",
                "analysis": analysis,
                "recommendations": recommendations,
                "timestamp": start_time.isoformat(),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
            
            # Store in shared context
            self.shared_context.set("adaptive_learning_analysis", result, "adaptive_learning")
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "agent": "AdaptiveLearningAgent",
                "error": str(e),
                "timestamp": start_time.isoformat(),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    def _get_performance_data(self) -> Dict[str, Any]:
        """Get performance data from shared context or logs"""
        # Try to get from shared context first
        performance_data = self.shared_context.get("trading_performance")
        if performance_data:
            return performance_data
        
        # Otherwise, read from logs or API
        # This would need to be implemented based on your logging system
        # For now, return empty dict
        return {}
    
    def _has_sufficient_data(self, performance_data: Dict[str, Any]) -> bool:
        """Check if we have sufficient data for analysis"""
        total_trades = performance_data.get("total_trades", 0)
        time_window = performance_data.get("time_window_seconds", 0)
        
        return total_trades >= self.min_trades_for_analysis and time_window >= self.min_analysis_window
    
    def _analyze_performance(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trading performance metrics"""
        total_trades = performance_data.get("total_trades", 0)
        winning_trades = performance_data.get("winning_trades", 0)
        losing_trades = performance_data.get("losing_trades", 0)
        avg_win = performance_data.get("avg_win", 0.0)
        avg_loss = performance_data.get("avg_loss", 0.0)
        win_rate = winning_trades / max(1, total_trades)
        rr_ratio = avg_win / max(0.01, avg_loss) if avg_loss > 0 else 0.0
        max_drawdown = performance_data.get("max_drawdown", 0.0)
        trades_per_hour = performance_data.get("trades_per_hour", 0.0)
        
        # Calculate performance scores
        win_rate_score = 1.0 if win_rate >= 0.60 else (0.5 if win_rate >= 0.35 else 0.0)
        rr_score = 1.0 if rr_ratio >= 2.0 else (0.5 if rr_ratio >= 1.3 else 0.0)
        drawdown_score = 1.0 if max_drawdown < 0.05 else (0.5 if max_drawdown < 0.10 else 0.0)
        trade_frequency_score = 1.0 if 0.5 <= trades_per_hour <= 1.0 else 0.5
        
        overall_score = (win_rate_score + rr_score + drawdown_score + trade_frequency_score) / 4.0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "rr_ratio": rr_ratio,
            "max_drawdown": max_drawdown,
            "trades_per_hour": trades_per_hour,
            "win_rate_score": win_rate_score,
            "rr_score": rr_score,
            "drawdown_score": drawdown_score,
            "trade_frequency_score": trade_frequency_score,
            "overall_score": overall_score,
            "is_profitable": win_rate > (avg_loss / max(0.01, avg_win + avg_loss)) if avg_win > 0 and avg_loss > 0 else False
        }
    
    def _generate_recommendations(
        self,
        analysis: Dict[str, Any],
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate parameter adjustment recommendations"""
        recommendations = {
            "type": "NO_CHANGE",
            "parameters": {},
            "reasoning": "",
            "confidence": 0.0,
            "requires_approval": True
        }
        
        win_rate = analysis["win_rate"]
        rr_ratio = analysis["rr_ratio"]
        max_drawdown = analysis["max_drawdown"]
        trades_per_hour = analysis["trades_per_hour"]
        overall_score = analysis["overall_score"]
        
        # Check if trading should be paused
        if overall_score < 0.3 or max_drawdown > self.max_drawdown_threshold:
            recommendations["type"] = "PAUSE_TRADING"
            recommendations["reasoning"] = f"Poor performance detected: overall_score={overall_score:.2f}, max_drawdown={max_drawdown:.1%}"
            recommendations["confidence"] = 0.9
            return recommendations
        
        # Check if trading should be resumed (if currently paused)
        if self.trading_paused and overall_score > 0.6:
            recommendations["type"] = "RESUME_TRADING"
            recommendations["reasoning"] = f"Performance improved: overall_score={overall_score:.2f}"
            recommendations["confidence"] = 0.8
            return recommendations
        
        # Adjust R:R threshold if needed
        if rr_ratio < 1.3:
            # Poor R:R - tighten threshold
            new_rr_threshold = min(2.5, self.current_rr_threshold + 0.1)
            if new_rr_threshold != self.current_rr_threshold:
                recommendations["type"] = "ADJUST_RR_THRESHOLD"
                recommendations["parameters"]["min_risk_reward_ratio"] = new_rr_threshold
                recommendations["reasoning"] = f"Poor R:R ratio ({rr_ratio:.2f}:1) - tightening threshold from {self.current_rr_threshold:.2f} to {new_rr_threshold:.2f}"
                recommendations["confidence"] = 0.8
        elif rr_ratio >= 2.0 and self.current_rr_threshold > 1.3:
            # Good R:R - can relax slightly
            new_rr_threshold = max(1.3, self.current_rr_threshold - 0.05)
            if new_rr_threshold != self.current_rr_threshold:
                recommendations["type"] = "ADJUST_RR_THRESHOLD"
                recommendations["parameters"]["min_risk_reward_ratio"] = new_rr_threshold
                recommendations["reasoning"] = f"Good R:R ratio ({rr_ratio:.2f}:1) - relaxing threshold from {self.current_rr_threshold:.2f} to {new_rr_threshold:.2f}"
                recommendations["confidence"] = 0.7
        
        # Adjust quality filters if needed
        if trades_per_hour > 2.0:
            # Too many trades - tighten filters
            new_confidence = min(0.2, self.current_min_confidence + 0.01)
            new_quality = min(0.5, self.current_min_quality + 0.02)
            if new_confidence != self.current_min_confidence or new_quality != self.current_min_quality:
                recommendations["type"] = "ADJUST_QUALITY_FILTERS"
                recommendations["parameters"]["min_action_confidence"] = new_confidence
                recommendations["parameters"]["min_quality_score"] = new_quality
                recommendations["reasoning"] = f"Too many trades ({trades_per_hour:.2f}/hour) - tightening filters"
                recommendations["confidence"] = 0.8
        elif trades_per_hour < 0.3 and win_rate >= 0.35:
            # Too few trades but performance is OK - relax filters
            new_confidence = max(0.1, self.current_min_confidence - 0.01)
            new_quality = max(0.3, self.current_min_quality - 0.02)
            if new_confidence != self.current_min_confidence or new_quality != self.current_min_quality:
                recommendations["type"] = "ADJUST_QUALITY_FILTERS"
                recommendations["parameters"]["min_action_confidence"] = new_confidence
                recommendations["parameters"]["min_quality_score"] = new_quality
                recommendations["reasoning"] = f"Too few trades ({trades_per_hour:.2f}/hour) but performance OK - relaxing filters"
                recommendations["confidence"] = 0.7
        
        return recommendations
    
    def _generate_llm_reasoning(
        self,
        analysis: Dict[str, Any],
        recommendations: Dict[str, Any],
        performance_data: Dict[str, Any]
    ) -> str:
        """Generate LLM reasoning for recommendations"""
        prompt = f"""Analyze the following trading performance data and provide reasoning for the recommended adjustments:

Performance Analysis:
- Total Trades: {analysis['total_trades']}
- Win Rate: {analysis['win_rate']:.1%}
- Risk/Reward Ratio: {analysis['rr_ratio']:.2f}:1
- Average Win: ${analysis['avg_win']:.2f}
- Average Loss: ${analysis['avg_loss']:.2f}
- Max Drawdown: {analysis['max_drawdown']:.1%}
- Trades per Hour: {analysis['trades_per_hour']:.2f}
- Overall Score: {analysis['overall_score']:.2f}

Current Parameters:
- R:R Threshold: {self.current_rr_threshold:.2f}
- Min Confidence: {self.current_min_confidence:.3f}
- Min Quality Score: {self.current_min_quality:.3f}

Recommendation: {recommendations['type']}
Parameters: {recommendations.get('parameters', {})}

Provide clear, concise reasoning for this recommendation. Explain:
1. Why this adjustment is needed
2. Expected impact on performance
3. Any risks or considerations"""
        
        try:
            response = self.reasoning_engine.generate_reasoning(
                prompt=prompt,
                context={"analysis": analysis, "recommendations": recommendations}
            )
            return response.get("reasoning", recommendations.get("reasoning", ""))
        except Exception as e:
            # Fallback to basic reasoning if LLM fails
            return recommendations.get("reasoning", "LLM reasoning unavailable")
    
    def _store_analysis(
        self,
        analysis: Dict[str, Any],
        recommendations: Dict[str, Any],
        timestamp: datetime
    ):
        """Store analysis in history"""
        entry = {
            "timestamp": timestamp.isoformat(),
            "analysis": analysis,
            "recommendations": recommendations
        }
        
        self.performance_history.append(entry)
        
        # Keep only last 1000 entries in memory
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        # Append to log file
        try:
            with open(self.performance_log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            print(f"[WARN] Failed to write to performance log: {e}")
    
    def _get_cached_result(self) -> Dict[str, Any]:
        """Get cached result from shared context"""
        cached = self.shared_context.get("adaptive_learning_analysis")
        if cached:
            return cached
        
        return {
            "status": "cached",
            "message": "Analysis too recent, using cached result",
            "timestamp": datetime.now().isoformat()
        }
    
    def apply_recommendation(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a recommendation (after manual approval).
        
        Args:
            recommendation: Recommendation dict with type and parameters
        
        Returns:
            Dict with application result
        """
        rec_type = recommendation.get("type")
        parameters = recommendation.get("parameters", {})
        
        if rec_type == "ADJUST_RR_THRESHOLD":
            self.current_rr_threshold = parameters.get("min_risk_reward_ratio", self.current_rr_threshold)
            # Update in shared context
            self.shared_context.set("min_risk_reward_ratio", self.current_rr_threshold, "adaptive_learning")
        
        elif rec_type == "ADJUST_QUALITY_FILTERS":
            self.current_min_confidence = parameters.get("min_action_confidence", self.current_min_confidence)
            self.current_min_quality = parameters.get("min_quality_score", self.current_min_quality)
            # Update in shared context
            self.shared_context.set("quality_filters", {
                "min_action_confidence": self.current_min_confidence,
                "min_quality_score": self.current_min_quality
            }, "adaptive_learning")
        
        elif rec_type == "PAUSE_TRADING":
            self.trading_paused = True
            self.shared_context.set("trading_paused", True, "adaptive_learning")
        
        elif rec_type == "RESUME_TRADING":
            self.trading_paused = False
            self.shared_context.set("trading_paused", False, "adaptive_learning")
        
        return {
            "status": "applied",
            "type": rec_type,
            "parameters": parameters,
            "timestamp": datetime.now().isoformat()
        }

