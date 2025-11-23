# Metrics Regression Analysis

## Date: Current Training Session

## Summary

**CRITICAL REGRESSION DETECTED**: Mean PnL (Last 10 Episodes) dropped from **+$745.01** to **-$641.22** (a drop of $1,386.23).

## Metrics Comparison

### Previous Check (Episode 376)
- **Mean PnL (Last 10)**: +$745.01 ✅
- **Mean Win Rate (Last 10)**: 40.0% ✅ (Above breakeven)
- **Overall Win Rate**: 33.3%
- **Total Trades**: 6
- **Current Episode Trades**: 4

### Current Check (Episode 377)
- **Mean PnL (Last 10)**: -$641.22 ❌
- **Mean Win Rate (Last 10)**: 32.5% ❌ (Below breakeven 34%)
- **Overall Win Rate**: 27.3%
- **Total Trades**: 11
- **Current Episode Trades**: 4
- **Latest Episode Length**: 40 steps (very short)

## Key Findings

### 1. Metrics Verification ✅
- Win rate calculation is **correct**: 3 wins / 11 trades = 27.3%
- Trade count increased: 6 → 11 trades (more activity)
- But quality decreased: Win rate dropped significantly

### 2. Regression Analysis
- **Mean PnL**: Dropped by $1,386.23 (from positive to negative)
- **Mean Win Rate**: Dropped by 7.5% (from 40.0% to 32.5%)
- **Overall Win Rate**: Dropped by 6.0% (from 33.3% to 27.3%)

### 3. Persistent Issues
- **Episode Length**: Still very short (40 steps vs 9,980 mean)
  - This suggests episodes are terminating early
  - Could be exceptions, data boundary issues, or termination logic

### 4. Adaptive Trainer Status
- Last adjustments were at timestep 4,190,000 (before our fix)
- Current reward config still shows relaxed values:
  - `min_action_confidence`: 0.08
  - `min_quality_score`: 0.25
- **No recent filter tightening detected**

## Possible Root Causes

### 1. Short Episodes (40 steps)
- Episodes terminating very early
- May be causing:
  - Insufficient learning per episode
  - Poor trade quality
  - Incomplete market cycles

### 2. Poor Trade Quality
- More trades (11 vs 6) but lower win rate (27.3% vs 33.3%)
- Suggests model is taking lower-quality trades
- Risk/reward ratio may be poor

### 3. Model Overfitting or Poor Generalization
- Model may have learned patterns that don't generalize
- Recent episodes may be in different market conditions

### 4. Risk/Reward Ratio Filter
- May be rejecting good trades
- Or allowing poor trades through

### 5. Consecutive Loss Limit
- May be pausing trading too often
- Causing missed opportunities

## Immediate Actions Needed

### 1. Investigate Short Episodes
- [ ] Check backend logs for exceptions
- [ ] Verify data boundary checks
- [ ] Review episode termination logic
- [ ] Check if `IndexError` or other exceptions are occurring

### 2. Review Trade Quality
- [ ] Check average win/loss amounts
- [ ] Verify risk/reward ratio is being calculated correctly
- [ ] Review if stop loss is being enforced
- [ ] Check if quality filters are working

### 3. Monitor Adaptive Trainer
- [ ] Verify adaptive trainer isn't making bad adjustments
- [ ] Check if evaluation episodes are still causing issues
- [ ] Review recent performance snapshots

### 4. Consider Adjustments
- [ ] Temporarily disable consecutive loss limit
- [ ] Relax risk/reward ratio filter
- [ ] Increase exploration (entropy_coef)
- [ ] Check if action threshold is appropriate

## Recommendations

### Short Term
1. **Investigate episode termination** - This is the most critical issue
2. **Check backend logs** for exceptions or errors
3. **Review recent trade details** to understand why quality dropped

### Medium Term
1. **Fix episode termination logic** if issues found
2. **Adjust filters** if trade quality is poor
3. **Monitor for stability** after fixes

### Long Term
1. **Improve episode termination logic** to prevent early termination
2. **Enhance quality filters** to better identify good trades
3. **Review adaptive trainer logic** to prevent false adjustments

## Next Steps

1. Check backend console/logs for exceptions
2. Review episode termination logic in `src/trading_env.py`
3. Verify data boundary checks are working
4. Consider temporarily disabling consecutive loss limit
5. Monitor for improvement after fixes

---

**Status**: 🔴 **REGRESSION DETECTED - Investigation Required**

