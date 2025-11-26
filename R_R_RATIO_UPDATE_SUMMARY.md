# Risk/Reward Ratio (R:R) Update Summary

## Current Performance
- **Actual R:R**: 0.73 (Average Win: $107.30, Average Loss: $147.83)
- **Win Rate**: 43.64%
- **Profit Factor**: 0.61 (unprofitable - should be > 1.0)
- **Issue**: Losses are 38% larger than wins, causing unprofitability

## Changes Made

### 1. Reduced Minimum R:R Requirement
**File**: `configs/train_config_adaptive.yaml`

- **Before**: `min_risk_reward_ratio: 1.8`
- **After**: `min_risk_reward_ratio: 1.2`
- **Reason**: More achievable while still targeting profitable trading (1.5+ for good performance)

### 2. Updated Quality Scorer R:R
**File**: `configs/train_config_adaptive.yaml`

- **Before**: `min_risk_reward_ratio: 1.5` (in decision_gate section)
- **After**: `min_risk_reward_ratio: 1.2` (consistent with reward config)

### 3. Made R:R Enforcement More Adaptive
**File**: `src/trading_env.py`

**Key Changes**:
- Increased minimum trades from 10 to 20 before enforcing R:R (more reliable estimate)
- Changed from strict enforcement (reject if below 1.05 * required) to adaptive:
  - Only reject if R:R < 0.7 (catastrophically poor)
  - Allows learning while preventing extremely bad trades
  - Reward function still penalizes poor R:R, encouraging improvement

**Before**:
```python
if self.state.trades_count > 10:
    required_rr_with_buffer = self.min_risk_reward_ratio * 1.05  # 1.26
    if risk_reward_ratio < required_rr_with_buffer:
        reject_trade()
```

**After**:
```python
if self.state.trades_count > 20:
    min_acceptable_rr = 0.7  # Only reject if catastrophically poor
    if risk_reward_ratio < min_acceptable_rr:
        reject_trade()
```

## Rationale

### Why Lower the Requirement to 1.2?
- Current actual R:R is 0.73 (very poor)
- Strict enforcement at 1.8 * 1.05 = 1.89 would reject ALL trades
- This prevents the agent from learning to improve R:R
- Setting target at 1.2 is more achievable while still profitable
- Target for good performance: 1.5+ R:R

### Why Adaptive Enforcement?
- **Problem**: Strict enforcement prevents learning when R:R is poor
- **Solution**: Only reject catastrophically bad trades (R:R < 0.7)
- **Benefit**: 
  - Agent can learn to improve R:R
  - Reward function still penalizes poor R:R
  - Prevents extremely bad trades from causing large losses

## Expected Impact

1. **Learning**: Agent can now take trades even with current poor R:R (0.73), allowing it to learn to improve
2. **Protection**: Still rejects catastrophically bad trades (R:R < 0.7)
3. **Improvement**: Reward function penalties will encourage R:R improvement toward 1.2+ target
4. **Gradual Improvement**: As R:R improves, more trades will be allowed

## Target Metrics

- **Minimum Acceptable R:R**: 0.7 (prevents catastrophic losses)
- **Target R:R**: 1.2 (minimum for profitability)
- **Good R:R**: 1.5+ (optimal performance)

## Monitoring

Track these metrics to validate improvements:
1. **Actual R:R** (should improve from 0.73 toward 1.2+)
2. **Win Rate** (should improve from 43.64% toward 50%+)
3. **Profit Factor** (should improve from 0.61 toward 1.0+)
4. **Average Win vs Average Loss** (should improve from $107/$148 toward more balanced)

