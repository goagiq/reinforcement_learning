# Adaptive Learning Agent - Swarm Integration

## Overview

The **Adaptive Learning Agent** is a new swarm agent that monitors trading performance over time and suggests parameter adjustments for continuous learning and optimization. It works independently from the training system and provides recommendations that require manual approval.

## Key Features

✅ **Continuous Learning**: Runs continuously during live trading  
✅ **Historical Analysis**: Analyzes historical performance data (not real-time)  
✅ **Parameter Suggestions**: Suggests R:R threshold and quality filter adjustments  
✅ **Trading Control**: Can recommend pausing/resuming trading based on performance  
✅ **Swarm Integration**: Has access to swarm's shared context  
✅ **LLM Reasoning**: Uses LLM reasoning (configurable)  
✅ **Manual Approval**: All recommendations require manual approval  
✅ **Independent**: Works separately from training system  

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Live Trading System                        │
│                                                         │
│  ┌──────────────────────────────────────────────────┐ │
│  │         Swarm Orchestrator                        │ │
│  │                                                   │ │
│  │  Phase 1 (Parallel):                             │ │
│  │  ┌──────────────┐  ┌──────────┐                 │ │
│  │  │   Market     │  │ Sentiment│                 │ │
│  │  │   Research   │  │  Agent   │                 │ │
│  │  └──────┬───────┘  └────┬─────┘                 │ │
│  │         └────────┬───────┘                       │ │
│  │                  ↓                                │ │
│  │         Shared Context                            │ │
│  │                  ↓                                │ │
│  │  Phase 2 (Sequential):                            │ │
│  │  ┌──────────────────┐                            │ │
│  │  │  Analyst Agent   │                            │ │
│  │  └────────┬─────────┘                            │ │
│  │           ↓                                       │ │
│  │  ┌──────────────────┐                            │ │
│  │  │ Recommendation   │                            │ │
│  │  │     Agent        │                            │ │
│  │  └──────────────────┘                            │ │
│  └───────────────────────────────────────────────┘ │
│                                                         │
│  ┌──────────────────────────────────────────────────┐ │
│  │  Adaptive Learning Agent (Independent)            │ │
│  │  - Runs continuously (every 5 min)               │ │
│  │  - Analyzes historical performance               │ │
│  │  - Suggests parameter adjustments                │ │
│  │  - Can pause/resume trading                      │ │
│  │  - Requires manual approval                      │ │
│  └──────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## How It Works

### 1. Continuous Monitoring

The Adaptive Learning Agent runs continuously during live trading:
- **Analysis Frequency**: Every 5 minutes (configurable)
- **Data Requirements**: Needs at least 20 trades and 1 hour of data
- **Performance Metrics**: Analyzes win rate, R:R ratio, drawdown, trade frequency

### 2. Performance Analysis

The agent analyzes:
- **Win Rate**: Should be >= 35% minimum, target 60%+
- **Risk/Reward Ratio**: Should be >= 1.3:1 minimum, target 2.0:1+
- **Trade Frequency**: Should be 0.5-1.0 trades/hour ideally
- **Maximum Drawdown**: Should be < 10%
- **Average Win vs Average Loss**: Calculates profitability

### 3. Recommendation Types

The agent can recommend:

1. **ADJUST_RR_THRESHOLD**: Adjust minimum risk/reward ratio
   - Tighten if R:R < 1.3
   - Relax if R:R >= 2.0

2. **ADJUST_QUALITY_FILTERS**: Adjust min_action_confidence or min_quality_score
   - Tighten if trades/hour > 2.0
   - Relax if trades/hour < 0.3

3. **PAUSE_TRADING**: Recommend pausing trading
   - If overall_score < 0.3
   - If max_drawdown > 10%

4. **RESUME_TRADING**: Recommend resuming trading
   - If overall_score > 0.6 after pause

5. **NO_CHANGE**: No adjustments needed

### 4. LLM Reasoning (Configurable)

If enabled, the agent uses LLM reasoning to:
- Explain why adjustments are needed
- Describe expected impact
- Identify risks and considerations

### 5. Manual Approval

All recommendations require manual approval:
- Recommendations are stored in shared context
- UI can display recommendations for approval
- Once approved, `apply_recommendation()` is called
- Parameters are updated in shared context

## Integration with Swarm

### Shared Context Access

The agent reads/writes to shared context:
- **Reads**: `trading_performance` (historical performance data)
- **Writes**: `adaptive_learning_analysis` (analysis results)
- **Writes**: `min_risk_reward_ratio` (after approval)
- **Writes**: `quality_filters` (after approval)
- **Writes**: `trading_paused` (after approval)

### Workflow Integration

The agent runs **independently** from the main swarm workflow:
- Does NOT participate in trade decision making
- Does NOT influence individual trades
- Runs in background, analyzing performance
- Provides recommendations for manual review

## Configuration

### Agent Configuration

```yaml
agentic_swarm:
  adaptive_learning:
    enabled: true
    use_llm_reasoning: true  # Enable LLM reasoning (configurable)
    analysis_frequency: 300  # Every 5 minutes (seconds)
    min_trades_for_analysis: 20  # Minimum trades needed
    min_analysis_window: 3600  # 1 hour minimum (seconds)
    
    # Performance thresholds
    min_win_rate: 0.35  # 35% minimum
    min_rr_ratio: 1.3  # 1.3:1 minimum
    max_drawdown_threshold: 0.10  # 10% max drawdown
    
    # Initial parameters
    initial_rr_threshold: 1.5
    initial_min_confidence: 0.15
    initial_min_quality: 0.4
```

### Integration in Swarm Orchestrator

The agent is initialized separately and runs independently:

```python
# In SwarmOrchestrator.__init__()
if adaptive_learning_config.get("enabled", False):
    self.adaptive_learning_agent = AdaptiveLearningAgent(
        shared_context=self.shared_context,
        reasoning_engine=self.reasoning_engine,
        config=adaptive_learning_config
    )
```

## Usage Example

### 1. Initialize Agent

```python
from src.agentic_swarm.agents.adaptive_learning_agent import AdaptiveLearningAgent

agent = AdaptiveLearningAgent(
    shared_context=shared_context,
    reasoning_engine=reasoning_engine,
    config={
        "use_llm_reasoning": True,
        "analysis_frequency": 300,
        "min_trades_for_analysis": 20
    }
)
```

### 2. Run Analysis

```python
# Get performance data (from your trading system)
performance_data = {
    "total_trades": 50,
    "winning_trades": 30,
    "losing_trades": 20,
    "avg_win": 150.0,
    "avg_loss": 80.0,
    "max_drawdown": 0.05,
    "trades_per_hour": 0.8,
    "time_window_seconds": 7200  # 2 hours
}

# Run analysis
result = agent.analyze(
    market_state={},  # From shared context
    performance_data=performance_data
)

# Check recommendations
if result["status"] == "success":
    recommendations = result["recommendations"]
    if recommendations["type"] != "NO_CHANGE":
        # Display for manual approval
        print(f"Recommendation: {recommendations['type']}")
        print(f"Reasoning: {recommendations['reasoning']}")
        print(f"Parameters: {recommendations['parameters']}")
```

### 3. Apply Recommendation (After Approval)

```python
# After manual approval
if user_approved:
    application_result = agent.apply_recommendation(recommendations)
    print(f"Applied: {application_result['type']}")
```

## Benefits

1. **Continuous Optimization**: System adapts to changing market conditions
2. **Risk Management**: Can pause trading if performance degrades
3. **Performance Monitoring**: Tracks key metrics over time
4. **Manual Control**: All adjustments require approval
5. **Independent Operation**: Doesn't interfere with trade decisions
6. **LLM Reasoning**: Provides clear explanations (if enabled)

## Differences from Training AdaptiveTrainer

| Feature | Training AdaptiveTrainer | Adaptive Learning Agent |
|---------|-------------------------|------------------------|
| **When** | During training | During live trading |
| **Frequency** | Every 10k timesteps | Every 5 minutes |
| **Data Source** | Training episodes | Live trading history |
| **Adjustments** | Automatic | Manual approval required |
| **Scope** | Training parameters | Live trading parameters |
| **Integration** | Training loop | Swarm workflow |
| **LLM Reasoning** | No | Yes (configurable) |

## Next Steps

1. **Integrate with Live Trading System**: Add agent initialization and periodic analysis
2. **Performance Data Collection**: Implement data collection from live trading
3. **UI Integration**: Display recommendations for manual approval
4. **Testing**: Test with simulated trading data
5. **Monitoring**: Add logging and metrics tracking

---

## Summary

The Adaptive Learning Agent provides continuous learning and optimization during live trading, working independently from the main swarm workflow to monitor performance and suggest parameter adjustments that require manual approval.

