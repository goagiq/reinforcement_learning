# Model Evaluation Report - Post Retraining

**Generated**: 2025-11-22  
**Evaluation Date**: After Retraining from Scratch  
**Model Evaluated**: best_model.pt (from retraining with adaptive config)

---

## Executive Summary

### 🔴 **CRITICAL ISSUE: Model Saturation + Zero Trades**

The retrained model shows **severe performance degradation**:

- **Win Rate**: 4% (extremely low)
- **Trades per Episode**: 0.5 (only 10 trades in 20 episodes)
- **Total PnL**: -$13,659.62 (highly negative)
- **Sharpe Ratio**: -11.64 (catastrophic)
- **Sortino Ratio**: -151.35 (catastrophic)
- **Profit Factor**: 0.00 (no profits)

### Status: ❌ **MODEL FAILED - REQUIRES IMMEDIATE ATTENTION**

---

## Detailed Findings

### 1. Model Saturation (CRITICAL)

**Diagnostic Results:**
- Model outputs **1.0 (maximum action)** at **every single step**
- 100% action saturation - no diversity in decision-making
- This is the **same saturation issue** we saw before retraining

**Evidence:**
```
Step 0-99: Action value: 1.000000 (all steps identical)
Total actions: 100
Trades executed: 0
```

### 2. Zero Trade Execution (CRITICAL)

**Problem:**
- Actions are above threshold (1.0 >> 0.015 threshold)
- But **NO trades are being executed** (0 trades in diagnostic episodes)
- Quality filters are blocking ALL trades

**Root Cause Analysis:**
1. **Action Confidence**: ✅ Passes (1.0 > 0.20 min_action_confidence)
2. **Quality Score**: ❓ Likely failing (needs investigation)
3. **Expected Value**: ❓ May be None or negative (no trade history)
4. **Risk/Reward Ratio**: ❓ May be failing (no trade history to calculate)

### 3. Backtest Performance

**Metrics from 20-episode backtest:**
- **Mean Reward**: -1.64
- **Mean PnL**: -$682.98 per episode
- **Total PnL**: -$13,659.62
- **Mean Trades**: 0.50 per episode (10 total trades)
- **Mean Win Rate**: 4% (1 winning trade out of 10)
- **Sharpe Ratio**: -11.64
- **Sortino Ratio**: -151.35
- **Profit Factor**: 0.00

---

## Root Cause Analysis

### Why Did Retraining Fail?

1. **Early Saturation**: Model saturated very early in training (likely before 10k steps)
   - Despite increased `entropy_coef` (0.15) and action diversity rewards
   - Model learned to output maximum action value consistently

2. **Quality Filters Too Strict**: 
   - `min_action_confidence: 0.20` - Model outputs 1.0, so this passes
   - `min_quality_score: 0.50` - Likely failing due to no trade history
   - `min_risk_reward_ratio: 2.0` - Cannot calculate without trade history

3. **Catch-22 Situation**:
   - Model needs trade history to calculate quality metrics
   - But quality filters block trades when there's no history
   - Result: Zero trades executed, no learning possible

4. **Adaptive Training May Not Have Triggered**:
   - Adaptive trainer evaluates every 5,000 steps
   - If model saturated before first evaluation, adjustments never happened

---

## Comparison with Previous Model

| Metric | Previous Model | Retrained Model | Change |
|--------|---------------|-----------------|--------|
| Win Rate | 0% | 4% | ⬆️ Slight improvement |
| Trades/Episode | 1.0 | 0.5 | ⬇️ Worse (fewer trades) |
| Total PnL | -$15,938 | -$13,659 | ⬆️ Slightly better |
| Sharpe Ratio | Invalid | -11.64 | ⬇️ Worse (now measurable, but negative) |
| Action Saturation | Yes (1.0) | Yes (1.0) | ❌ Same issue |

**Conclusion**: Retraining did not fix the saturation issue. Performance is still poor.

---

## Recommendations

### Immediate Actions (Priority 1)

1. **Fix Quality Filter Logic**:
   - Allow trades when `expected_value` is `None` (insufficient history)
   - Reduce `min_quality_score` threshold for initial trades (0.3 instead of 0.5)
   - Temporarily disable `require_positive_expected_value` for first 50 trades

2. **Investigate Why Model Saturated**:
   - Check training logs for when saturation occurred
   - Review adaptive training adjustments (did they trigger?)
   - Check if entropy coefficient was actually applied during training

3. **Diagnose Quality Score Calculation**:
   - Add logging to see why quality score is failing
   - Check if quality score calculation needs trade history
   - Verify quality score formula is correct

### Short-Term Actions (Priority 2)

4. **Further Increase Exploration**:
   - Increase `entropy_coef` to 0.20-0.25 (from 0.15)
   - Increase `action_diversity_bonus` to 0.02 (from 0.01)
   - Increase `constant_action_penalty` to 0.10 (from 0.05)

5. **Relax Initial Quality Filters**:
   - Start with `min_quality_score: 0.3` for first 100 trades
   - Gradually increase to 0.5 after 100 trades
   - Use adaptive quality filter adjustment

6. **Add Early Saturation Detection**:
   - Monitor action diversity during training
   - If saturation detected before 10k steps, automatically increase entropy
   - Add alert when action variance drops below threshold

### Long-Term Actions (Priority 3)

7. **Review Training Data**:
   - Verify data quality and diversity
   - Check if data has sufficient market conditions
   - Ensure data covers different market regimes

8. **Consider Architecture Changes**:
   - Add action noise during training (Ornstein-Uhlenbeck process)
   - Use different activation functions (tanh instead of sigmoid)
   - Consider adding action regularization to prevent saturation

9. **Implement Gradual Quality Filter Introduction**:
   - Start with no quality filters for first 1,000 trades
   - Gradually introduce filters as model learns
   - Use adaptive thresholds based on model performance

---

## Next Steps

1. ✅ **Diagnostic Complete**: Model saturation confirmed
2. ⏳ **Fix Quality Filters**: Allow initial trades to pass through
3. ⏳ **Retrain with Fixed Filters**: Start new training run
4. ⏳ **Monitor Early Training**: Watch for saturation in first 10k steps
5. ⏳ **Evaluate After Fixes**: Run comprehensive evaluation

---

## Files to Review

- `src/trading_env.py` - Quality filter logic (lines 703-750)
- `configs/train_config_adaptive.yaml` - Quality filter thresholds
- Training logs - Check when saturation occurred
- Adaptive training logs - Check if adjustments were made

---

**Report Status**: 🔴 **CRITICAL - IMMEDIATE ACTION REQUIRED**
