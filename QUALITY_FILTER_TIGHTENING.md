# Quality Filter Tightening - Focus on Quality Trades

## Problem Identified

**Issue**: After previous changes to reduce overtrading, the system is now producing MORE bad trades instead of filtering for quality trades.

**Current Performance**:
- Total Trades: 869
- Total P&L: -$26,075.12
- Win Rate: 44.76%
- Profit Factor: 0.57 (should be >1.0)
- Avg Trade: -$30.01

**Root Cause**: Quality filters were made TOO LENIENT to allow learning, resulting in poor-quality trades being executed.

---

## Fixes Applied

### 1. Increased Action Confidence Threshold

**Before**: `min_action_confidence: 0.2` (only 20% confidence required)
**After**: `min_action_confidence: 0.4` (require 40% confidence minimum)

**Impact**: Filters out weak signals, only executes trades with reasonable confidence.

### 2. Increased Quality Score Threshold

**Before**: `min_quality_score: 0.5` (bare minimum - 50/50)
**After**: `min_quality_score: 0.65` (require better quality trades)

**Impact**: Only trades with quality score >= 0.65 will be executed.

### 3. Enabled Positive Expected Value Requirement

**Before**: `require_positive_expected_value: false` (allows unprofitable trades)
**After**: `require_positive_expected_value: true` (only allow profitable trades)

**Impact**: Only trades with positive expected value (profitable) will be executed.

### 4. Increased DecisionGate Quality Score

**Before**: `min_quality_score: 0.6` (DecisionGate)
**After**: `min_quality_score: 0.7` (DecisionGate)

**Impact**: DecisionGate now requires higher quality scores for trade approval.

### 5. Increased DecisionGate Confidence Threshold

**Before**: `min_combined_confidence: 0.3` (training override)
**After**: `min_combined_confidence: 0.5` (training override)

**Impact**: Requires 50% combined confidence minimum, up from 30%.

### 6. Increased Risk/Reward Ratio Target

**Before**: `min_risk_reward_ratio: 1.2`
**After**: `min_risk_reward_ratio: 1.5`

**Impact**: Requires better risk/reward ratio, targeting 1.5:1 minimum.

---

## Expected Improvements

### Trade Quality
- ✅ **Higher confidence**: Only trades with >= 40% confidence
- ✅ **Better quality score**: Only trades with >= 0.65 quality score
- ✅ **Positive EV**: Only profitable trades (EV > 0)
- ✅ **Better R:R**: Targeting 1.5:1 minimum

### Performance Metrics
- **Expected**: Profit Factor > 1.0 (currently 0.57)
- **Expected**: Win Rate > 50% (currently 44.76%)
- **Expected**: Positive avg trade (currently -$30.01)
- **Expected**: Fewer trades, but higher quality

### Trade Count
- **Expected**: Trade count may decrease (quality over quantity)
- **Expected**: Better win rate due to filtering
- **Expected**: Positive P&L trend

---

## Files Modified

1. **`configs/train_config_adaptive.yaml`**:
   - `quality_filters.min_action_confidence`: 0.2 → 0.4
   - `quality_filters.min_quality_score`: 0.5 → 0.65
   - `quality_filters.require_positive_expected_value`: false → true
   - `decision_gate.quality_scorer.min_quality_score`: 0.6 → 0.7
   - `reward.min_risk_reward_ratio`: 1.2 → 1.5

2. **`src/train.py`**:
   - `min_combined_confidence` (training override): 0.3 → 0.5

---

## Verification

After restarting training, monitor:

1. **Trade Count**: Should decrease (quality over quantity)
2. **Win Rate**: Should increase toward 50%+
3. **Profit Factor**: Should improve toward >1.0
4. **Avg Trade**: Should become positive
5. **Total P&L**: Should show positive trend

---

## Summary

✅ **Tightened**: All quality filter thresholds increased
✅ **Enabled**: Positive expected value requirement
✅ **Focused**: On quality trades, not quantity
✅ **Expected**: Better performance metrics with fewer, higher-quality trades

The system will now be more selective, executing only trades that meet higher quality standards.

