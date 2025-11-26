# Model Performance Assessment

## Executive Summary

**Status: ⚠️ NEEDS IMPROVEMENT**

The model is showing mixed results with some positive trends but overall negative performance.

---

## Key Metrics

### Overall Performance (All 3,523 Trades)

| Metric | Value | Status |
|--------|-------|--------|
| **Total Trades** | 3,523 | - |
| **Win Rate** | 45.44% | ✅ GOOD (target: >45%) |
| **Total PnL** | -$113,436.19 | ❌ LOSING |
| **Average Trade** | -$32.20 | ❌ NEGATIVE |
| **Profit Factor** | 0.61 | ❌ POOR (target: >1.0) |
| **Sharpe Ratio** | -2.89 | ❌ VERY POOR |
| **Max Drawdown** | $115,071.11 | ❌ LARGE |

### Risk/Reward Analysis

| Metric | Value | Assessment |
|--------|-------|------------|
| **Average Win** | $111.97 | ✅ Reasonable |
| **Average Loss** | -$152.29 | ❌ Too large |
| **Risk/Reward Ratio** | 0.74 | ❌ POOR (target: >1.5) |

**Problem:** Losses are 36% larger than wins, which explains the negative PnL despite decent win rate.

---

## Recent Performance (Last 50 Trades)

| Metric | Value | Trend |
|--------|-------|-------|
| **Win Rate** | 46.00% | ✅ Slightly improved |
| **Total PnL** | -$828.87 | ⚠️ Still negative |
| **Average PnL** | -$16.58 | ⚠️ Better than overall (-$32.20) |

**Observation:** Recent performance is **improving** - average loss per trade is smaller.

---

## Equity Curve Analysis

| Metric | Value | Status |
|--------|-------|--------|
| **Current Equity** | $101,484.37 | - |
| **Peak Equity** | $101,648.42 | - |
| **Current Drawdown** | 0.16% | ✅ MINIMAL |
| **Trend (Last 10 vs Previous 10)** | +$215.32 | ✅ POSITIVE |

**Positive Sign:** Equity curve shows **recent upward trend** and minimal drawdown.

---

## Forecast Features Performance

| Metric | Value | Status |
|--------|-------|--------|
| **Forecast Enabled** | Yes | ✅ Active |
| **Regime Enabled** | Yes | ✅ Active |
| **State Dim Match** | No | ⚠️ Mismatch (900 vs 908) |
| **Trades with Forecasts** | 1,000 | - |
| **Win Rate** | 45.00% | ✅ Good |
| **Total PnL** | -$32,398.35 | ❌ Negative |
| **Profit Factor** | 0.70 | ⚠️ Better than overall (0.61) |

**Observation:** Forecast features show **slightly better** profit factor (0.70 vs 0.61), suggesting they may be helping.

---

## Critical Issues

### 1. **Profit Factor Too Low (0.61)**
- **Target:** >1.0
- **Current:** 0.61
- **Impact:** Even with 45% win rate, losses outweigh wins

### 2. **Risk/Reward Ratio Poor (0.74)**
- **Target:** >1.5
- **Current:** 0.74
- **Problem:** Average loss ($152) is much larger than average win ($112)

### 3. **Large Cumulative Losses**
- Total PnL: -$113,436
- This is from **all historical trades** (including before checkpoint resume)
- Recent trades show improvement

### 4. **State Dimension Mismatch**
- Checkpoint: 900 dimensions
- Config: 908 dimensions (with forecast features)
- System should handle this with transfer learning, but worth monitoring

---

## Positive Signs

### ✅ Win Rate is Good
- 45.44% win rate is above the 45% target
- Recent trades show 46% win rate (improving)

### ✅ Recent Performance Improving
- Last 50 trades: -$16.58 avg (vs -$32.20 overall)
- Equity curve trending upward
- Minimal current drawdown (0.16%)

### ✅ Forecast Features Helping
- Profit factor with forecasts: 0.70 (vs 0.61 overall)
- Suggests forecast features are providing value

---

## Recommendations

### Immediate Actions

1. **Monitor Recent Trades Only**
   - Use `since` parameter to filter trades after checkpoint resume
   - This will show true performance of current training run
   - Old trades from before checkpoint are skewing the numbers

2. **Focus on Risk Management**
   - Average loss ($152) is too large
   - Consider:
     - Tighter stop-losses
     - Better position sizing
     - Risk/reward filters

3. **Verify Transfer Learning**
   - Check if state dimension transfer (900→908) is working correctly
   - Monitor if forecast features are being used effectively

### Training Adjustments

1. **Improve Risk/Reward**
   - Target: Average win should be 1.5x average loss
   - Current: Average win is only 0.74x average loss
   - Adjust reward function to penalize large losses more

2. **Continue Training**
   - Recent trends are positive
   - Model may need more training to learn better risk management
   - Monitor next 100-200 trades for improvement

3. **Review Stop-Loss Logic**
   - Current stop-loss may be too wide
   - Consider adaptive stops based on volatility

---

## Next Steps

1. **Filter Performance by Timestamp**
   - Get timestamp when training resumed from checkpoint
   - Use `/api/monitoring/performance?since=TIMESTAMP` to see only new trades
   - This will give accurate picture of current training performance

2. **Monitor Next 100 Trades**
   - Track if recent improvement continues
   - Watch for:
     - Win rate maintaining >45%
     - Average loss decreasing
     - Profit factor improving toward 1.0

3. **Review Training Configuration**
   - Check if reward function properly penalizes large losses
   - Verify stop-loss settings
   - Ensure forecast features are working correctly

---

## Conclusion

**Current Status:** ⚠️ **MIXED - Improving but needs work**

The model shows:
- ✅ **Good win rate** (45%+)
- ✅ **Recent improvement** (smaller losses, upward equity trend)
- ❌ **Poor profit factor** (0.61 - losses too large)
- ❌ **Negative cumulative PnL** (but includes old trades)

**Key Insight:** The model is winning often enough, but losses are too large. Focus on **risk management** and **reducing average loss size**.

**Recommendation:** Continue training and monitor recent trades (after checkpoint resume) separately from historical trades to get accurate performance picture.

