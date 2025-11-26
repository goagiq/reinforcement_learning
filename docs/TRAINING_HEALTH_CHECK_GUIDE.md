# Training Health Check Guide

## Overview

The `check_training_health.py` script provides a comprehensive analysis of your training progress with detailed explanations of what each metric means.

## Usage

```bash
python check_training_health.py
```

## What It Shows

### 1. Training Status
- Current status (running, idle, error)
- Progress percentage
- Episode and timestep counts

### 2. Trading Metrics (with explanations)
- **Total Trades**: Number of trades executed
  - Good: Increasing steadily
  - Bad: 0 or very low (agent too conservative)
  
- **Win Rate**: Percentage of profitable trades
  - Good: > 50%
  - Bad: < 30%
  
- **PnL (Profit/Loss)**: Current and average profitability
  - Good: Positive and increasing
  - Bad: Negative or decreasing
  
- **Risk/Reward Ratio**: Average win / Average loss
  - Good: > 2.0 (wins are 2x larger than losses)
  - Bad: < 1.0 (losses are larger than wins)
  
- **Max Drawdown**: Maximum equity drop
  - Good: < 5%
  - Bad: > 10%

### 3. Health Analysis
- Overall status (healthy, warning, critical)
- Identified issues
- Warnings
- Recommendations
- Strengths

## Frontend Integration

The Monitoring tab in the frontend automatically displays this data when training is active. It refreshes every 5 seconds to show real-time metrics.

## Understanding the Metrics

### Win Rate
- **What it means**: Percentage of trades that made money
- **Good**: > 50% means more wins than losses
- **Bad**: < 30% means losing more often than winning

### PnL (Profit and Loss)
- **What it means**: How much money you're making/losing
- **Current Episode PnL**: Profit/loss in the current episode
- **Mean PnL (Last 10)**: Average profit/loss over recent episodes
- **Good**: Positive and trending upward
- **Bad**: Negative or trending downward

### Risk/Reward Ratio
- **What it means**: Average win size / Average loss size
- **Good**: > 2.0 means wins are 2x larger than losses
- **Bad**: < 1.0 means losses are larger than wins

### Max Drawdown
- **What it means**: Maximum equity drop from peak
- **Good**: < 5% (low risk)
- **Bad**: > 10% (high risk)

### Sharpe Ratio
- **What it means**: Risk-adjusted return (higher is better)
- **Good**: > 1.0
- **Bad**: < 0.5

## Common Issues and Solutions

### Issue: No Trades Executed
- **Cause**: Agent is too conservative
- **Solution**: 
  - Increase `entropy_coef` in config
  - Lower `action_threshold`
  - Review quality filters

### Issue: Low Win Rate
- **Cause**: Model needs more training or reward function needs adjustment
- **Solution**:
  - Continue training
  - Review reward function alignment with PnL
  - Check if transaction costs are too high

### Issue: Negative PnL
- **Cause**: Model is losing money
- **Solution**:
  - Review reward function
  - Check transaction costs
  - May need parameter tuning

### Issue: Poor Risk/Reward Ratio
- **Cause**: Average losses are larger than wins
- **Solution**:
  - Tighten stop-loss
  - Improve entry timing
  - Review position sizing

## Monitoring Tab

The Monitoring tab in the frontend shows:
- Total P&L
- Sharpe Ratio
- Sortino Ratio
- Win Rate
- Profit Factor
- Max Drawdown
- Total Trades
- Average Trade

All metrics update automatically every 5 seconds when training is active.

## Next Steps

1. Run `check_training_health.py` regularly to monitor progress
2. Check the Monitoring tab in the frontend for real-time updates
3. Address any issues or warnings identified
4. Continue training and monitor improvements

