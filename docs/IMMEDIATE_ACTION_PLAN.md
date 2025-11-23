# Immediate Action Plan - Fix Low Trade Count

## Problem Summary

**Current State**:
- 9 trades in 361 episodes (0.025 trades/episode)
- Expected: 0.5-1.0 trades/episode (180 trades)
- **Gap**: 172 trades missing

**Root Cause**: System is TOO CONSERVATIVE - filters are rejecting too many trades

---

## Immediate Fixes (Apply Now)

### Fix 1: Reduce Action Threshold
**File**: `configs/train_config_adaptive.yaml`
**Change**: `action_threshold: 0.05` → `action_threshold: 0.02`
**Impact**: Will allow more trades to pass through (2% vs 5% threshold)

### Fix 2: Reduce DecisionGate Confidence (Training)
**File**: `src/train.py` (already set to 0.5, but can reduce further)
**Change**: `min_combined_confidence: 0.5` → `min_combined_confidence: 0.3`
**Impact**: Will allow more trades during training

### Fix 3: Relax Quality Filters
**File**: `configs/train_config_adaptive.yaml`
**Changes**:
- `min_action_confidence: 0.15` → `min_action_confidence: 0.1`
- `min_quality_score: 0.4` → `min_quality_score: 0.3`
**Impact**: Will allow more trades while still filtering

### Fix 4: Disable Expected Value Check (Temporarily)
**File**: `configs/train_config_adaptive.yaml`
**Change**: `require_positive_expected_value: false` (already false, but verify)
**Impact**: EV calculation needs more data - disabling allows more trades early in training

---

## Expected Results After Fixes

- **Trade Count**: Should increase to 0.3-0.5 trades/episode
- **Win Rate**: May initially decrease slightly, but should improve as system learns
- **Learning**: Agent will have more trading experience to learn from

---

## Monitoring Plan

After applying fixes, monitor:
1. **Trade Count**: Should see 10-20 trades in next 50 episodes
2. **Win Rate**: Track if it improves or decreases
3. **Episode Length**: Should remain ~10,000 steps (no more 20-step episodes)
4. **Mean PnL**: Should remain positive

---

## Gradual Tightening Strategy

Once trade count increases:
1. **Week 1**: Monitor with relaxed thresholds
2. **Week 2**: Gradually increase `action_threshold` to 0.03
3. **Week 3**: Gradually increase `min_combined_confidence` to 0.4
4. **Week 4**: Gradually increase quality filter thresholds
5. **Target**: Reach 0.5-1.0 trades/episode with 60%+ win rate

