# Adaptive Profitability Integration - Summary

## ✅ Implementation Complete

The profitability fixes have been integrated into the adaptive learning system. The system will now automatically adjust parameters based on performance.

## What's Adaptive

### 1. **Risk/Reward Ratio Threshold** ✅
- **Adaptive**: Yes
- **Logic**: 
  - If `avg_win / avg_loss < 1.5`: Tighten threshold (increase from 1.5 → up to 2.5)
  - If `avg_win / avg_loss >= 2.0`: Relax threshold (decrease from 1.5 → down to 1.3)
- **Range**: 1.3 to 2.5
- **Adjustment Rate**: 0.1 per evaluation

### 2. **Quality Filter Thresholds** ✅
- **Adaptive**: Yes
- **Parameters**: `min_action_confidence`, `min_quality_score`
- **Logic**:
  - If `trades/episode > 2.0`: Tighten filters (increase thresholds)
  - If `trades/episode < 0.3`: Relax filters (decrease thresholds)
- **Ranges**:
  - `min_action_confidence`: 0.1 to 0.2
  - `min_quality_score`: 0.3 to 0.5
- **Adjustment Rate**: 0.01 per evaluation

### 3. **Stop Loss** ❌
- **Adaptive**: No (fixed at 2%)
- **Reason**: Hard risk management rule - safety mechanism

## How It Works

### 1. Adaptive Trainer Monitors Performance
- Every 10,000 timesteps (or on evaluation), checks:
  - Risk/reward ratio (`avg_win / avg_loss`)
  - Trade count per episode
  - Profitability status

### 2. Adjustments Made Automatically
- If losing money: Tightens R:R threshold and quality filters
- If too many trades: Tightens quality filters
- If no trades: Relaxes quality filters
- If very profitable: Relaxes R:R threshold slightly

### 3. Environment Reads Adaptive Values
- On each episode reset, environment reads from:
  - `logs/adaptive_training/current_reward_config.json`
- Uses adaptive values for:
  - `min_risk_reward_ratio`
  - `min_action_confidence`
  - `min_quality_score`

### 4. No Manual Intervention Required
- All adjustments happen automatically during training
- Logged to `logs/adaptive_training/config_adjustments.jsonl`
- Visible in console output with `[ADAPT]` prefix

## Configuration

### AdaptiveConfig (in `src/adaptive_trainer.py`)

```python
# Risk/reward ratio adjustment
rr_adjustment_enabled: bool = True
min_rr_threshold: float = 1.3  # Minimum R:R threshold
max_rr_threshold: float = 2.5  # Maximum R:R threshold
rr_adjustment_rate: float = 0.1  # How much to adjust per step

# Quality filter adjustment
quality_filter_adjustment_enabled: bool = True
min_action_confidence_range: Tuple[float, float] = (0.1, 0.2)
min_quality_score_range: Tuple[float, float] = (0.3, 0.5)
quality_adjustment_rate: float = 0.01  # How much to adjust per step
```

## Example Adjustments

### Scenario 1: Losing Money (Poor R:R)
```
Current R:R: 1.2:1 (avg_win=$50, avg_loss=$60)
Action: Tighten R:R threshold
  min_risk_reward_ratio: 1.5 → 1.6
Result: System rejects more trades until R:R improves
```

### Scenario 2: Too Many Trades
```
Trades per episode: 3.5
Action: Tighten quality filters
  min_action_confidence: 0.15 → 0.16
  min_quality_score: 0.4 → 0.42
Result: Fewer but higher quality trades
```

### Scenario 3: No Trades
```
Trades per episode: 0.1
Action: Relax quality filters
  min_action_confidence: 0.15 → 0.14
  min_quality_score: 0.4 → 0.38
Result: More trades allowed for exploration
```

### Scenario 4: Very Profitable
```
Current R:R: 2.5:1 (avg_win=$100, avg_loss=$40)
Action: Relax R:R threshold slightly
  min_risk_reward_ratio: 1.5 → 1.45
Result: Allows slightly more trades while maintaining profitability
```

## Benefits

1. **Automatic Optimization**: System adjusts itself based on actual performance
2. **Better Trade Quality**: Tightens filters when too many trades
3. **Better Exploration**: Relaxes filters when no trades
4. **Profitability Focus**: Adjusts R:R threshold based on actual win/loss ratios
5. **No Manual Intervention**: All adjustments happen automatically

## Files Modified

1. **`src/adaptive_trainer.py`**:
   - Added adaptive R:R threshold adjustment
   - Added adaptive quality filter adjustment
   - Extended `_update_reward_config()` to include new parameters
   - Initializes adaptive config file on startup

2. **`src/trading_env.py`**:
   - Reads adaptive parameters from config file on each reset
   - Uses adaptive values for R:R check and quality filters

## Next Steps

1. **Monitor Adjustments**: Watch console for `[ADAPT]` messages
2. **Check Logs**: Review `logs/adaptive_training/config_adjustments.jsonl`
3. **Verify Behavior**: Ensure adjustments are being applied correctly
4. **Tune Rates**: Adjust `rr_adjustment_rate` and `quality_adjustment_rate` if needed

---

## Summary

✅ **Risk/Reward Ratio**: Adaptive (1.3-2.5 range)
✅ **Quality Filters**: Adaptive (confidence 0.1-0.2, quality 0.3-0.5)
❌ **Stop Loss**: Fixed at 2% (safety mechanism)

The system will now automatically optimize itself for profitability!

