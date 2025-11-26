# Adaptive Stop Loss Implementation

## Overview

Adaptive stop-loss has been implemented to dynamically adjust stop-loss distance based on market volatility and trading performance. The system now automatically widens stops during high volatility periods and tightens them during low volatility, while also considering recent trading performance.

## Features

1. **Volatility-Based Adjustment**: Uses `VolatilityPredictor` to assess current market volatility and adjusts stop-loss distance accordingly
   - High volatility (>80th percentile): Widen stops by 50% (multiplier: 1.5x)
   - Above average volatility (>60th percentile): Widen stops by 25% (multiplier: 1.25x)
   - Low volatility (<30th percentile): Tighten stops by 15% (multiplier: 0.85x)
   - Normal volatility (30-60th percentile): Standard stops (multiplier: 1.0x)

2. **Performance-Based Adjustment**: Considers recent trading performance
   - Frequent stop hits in high volatility: Widen stops by 0.3%
   - Frequent stop hits in low volatility: Slight widening by 0.1%
   - Small losses in low volatility: Tighten stops by 0.2%

3. **Hard Safety Limits**: 
   - **Minimum stop loss**: 1.0% (hard floor - never goes below)
   - **Maximum stop loss**: 3.0% (hard ceiling - never exceeds)
   - **Base stop loss**: 1.5% (starting/default value)

4. **Adjustment Frequency**: Evaluated every 5,000 timesteps during training (same as other adaptive parameters)

## Configuration

### AdaptiveConfig (src/adaptive_trainer.py)

```python
@dataclass
class AdaptiveConfig:
    # Stop loss adjustment (NEW - adaptive based on volatility and performance)
    stop_loss_adjustment_enabled: bool = True
    min_stop_loss_pct: float = 0.01  # Hard minimum 1.0% (safety floor)
    max_stop_loss_pct: float = 0.03  # Maximum 3.0% (for high volatility)
    stop_loss_adjustment_rate: float = 0.002  # How much to adjust per step (0.2%)
    base_stop_loss_pct: float = 0.015  # Base/starting stop loss (1.5%)
```

## Implementation Details

### 1. Adaptive Trainer (src/adaptive_trainer.py)

- Added `current_stop_loss_pct` tracking to `AdaptiveTrainer.__init__()`
- Added stop loss adjustment logic in `_analyze_and_adjust()` method
- Volatility prediction uses `VolatilityPredictor` with evaluation test data
- Performance metrics (avg_loss, avg_win) from `ModelMetrics` inform adjustments
- Stop loss is saved to `logs/adaptive_training/current_reward_config.json`

### 2. Trading Environment (src/trading_env.py)

- Modified `TradingEnvironment.__init__()` to read adaptive stop loss from config file
- Falls back to default (2% from reward_config) if adaptive config not available
- Prints message when using adaptive stop loss

### 3. Volatility Predictor Integration

- Uses existing `VolatilityPredictor.get_adaptive_stop_loss_multiplier()` method
- Calculates volatility percentile from price data
- Returns multiplier between 0.7x and 2.0x based on volatility conditions

## How It Works

### During Training

1. Every 5,000 timesteps, `AdaptiveTrainer` evaluates the model
2. Gets volatility forecast from `VolatilityPredictor` using test data
3. Calculates volatility multiplier (0.7x - 2.0x)
4. Analyzes recent trading performance:
   - Checks if average loss is close to stop loss threshold (indicates frequent stop hits)
   - Adjusts based on volatility percentile
5. Combines adjustments:
   ```
   new_stop_loss = current_stop_loss * volatility_multiplier + performance_adjustment
   new_stop_loss = clamp(new_stop_loss, min=1.0%, max=3.0%)
   ```
6. Updates `current_reward_config.json` with new stop loss
7. Trading environment reads updated stop loss on next reset

### During Live Trading

- The adaptive stop loss from training is saved in the config file
- `AdaptiveLearningAgent` can also adjust stop loss during live trading (separate system)

## Example Adjustments

### Scenario 1: High Volatility + Frequent Stop Hits
- Current stop: 1.5%
- Volatility: 85th percentile → multiplier: 1.5x
- Performance: Frequent stops → +0.3%
- New stop: 1.5% * 1.5 + 0.3% = 2.55%

### Scenario 2: Low Volatility + Small Losses
- Current stop: 1.5%
- Volatility: 25th percentile → multiplier: 0.85x
- Performance: Small losses → -0.2%
- New stop: 1.5% * 0.85 - 0.2% = 1.075% (clamped to 1.0% minimum)

### Scenario 3: Normal Volatility
- Current stop: 1.5%
- Volatility: 50th percentile → multiplier: 1.0x
- Performance: No adjustment needed
- New stop: 1.5% * 1.0 = 1.5%

## Monitoring

### View Current Stop Loss

```bash
cat logs/adaptive_training/current_reward_config.json
```

### View Adjustment History

```bash
cat logs/adaptive_training/config_adjustments.jsonl | grep stop_loss
```

### Systems Tab

The Systems tab in the frontend shows:
- Last adjustment timestamp
- Current stop loss value
- Adjustment history with volatility percentile and multiplier

## Benefits

1. **Risk Management**: Automatically widens stops during volatile market conditions to avoid premature exits
2. **Profit Optimization**: Tightens stops during calm markets to protect profits better
3. **Performance-Driven**: Adjusts based on actual trading results, not just market conditions
4. **Safety**: Hard limits (1.0% - 3.0%) prevent excessive risk or overly tight stops
5. **Continuous Learning**: Adapts throughout training to optimize stop placement

## Future Enhancements

- Track stop loss hit frequency more precisely during training
- Consider ATR-based stop loss as alternative to percentage-based
- Adjust stop loss based on time-of-day (e.g., widen during news events)
- Consider position size when adjusting stop loss

## Notes

- Stop loss adjustments are evaluated every 5,000 timesteps (same as other adaptive parameters)
- Minimum change threshold: 0.1% (adjustments smaller than this are ignored)
- Stop loss is separate from the fixed 2% stop loss that was previously in place
- The adaptive system now replaces the fixed stop loss during training

