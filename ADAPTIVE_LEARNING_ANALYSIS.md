# Adaptive Learning Analysis Report

**Generated**: 2025-11-25  
**Current Timestep**: 80,000  
**Last Adjustment**: Timestep 70,000

---

## ✅ Adaptive Learning Status: **ACTIVE**

### Summary
- **Total Adjustments**: 325
- **Last Adjustment**: Timestep 70,000 (10,000 timesteps ago)
- **Evaluation Frequency**: Every 5,000 timesteps
- **Status**: ⚠️ **No adjustment in last 10,000 timesteps** (should have had 2 evaluations)

---

## 📊 Current Adaptive Parameters

### Active Settings
- **R:R Ratio**: 2.0 (not being adjusted - 0 adjustments)
- **Entropy Coef**: 0.025 (13 adjustments made)
- **Inaction Penalty**: 0.0001 (2 adjustments made)
- **Min Action Confidence**: 0.218 (102 adjustments made)
- **Min Quality Score**: 0.535 (102 adjustments made)
- **Stop Loss**: 1.0% (volatility-adjusted)

---

## 🔍 Adjustment Analysis

### Adjustment Types
1. **Quality Filters**: 102 adjustments (most common)
   - Adjusting `min_action_confidence` and `min_quality_score`
   - Reason: "Too many trades" - system is tightening filters
   - Recent: Lowered from 0.25 → 0.2 (confidence) and 0.6 → 0.5 (quality)

2. **Entropy Coef**: 13 adjustments
   - Adjusting exploration vs exploitation
   - Recent: Increased from 0.055 → 0.085 (increasing exploration)
   - Reason: "Policy converged + negative trend - increasing exploration"

3. **R:R Ratio**: 0 adjustments
   - **Not being adjusted** - may need review

4. **Inaction Penalty**: 2 adjustments
   - Minimal adjustments

### Recent Adjustment Pattern
- **Last 10 adjustments** all at timesteps: 30k, 40k, 50k, 60k, 70k
- **Pattern**: Adjustments happening every 10,000 timesteps
- **Issue**: Should be every 5,000 timesteps (eval frequency)

---

## ⚠️ Issues Identified

### 1. **Missing Evaluations**
- **Problem**: No adjustment since timestep 70,000
- **Expected**: Should have had evaluations at 75,000 and 80,000
- **Possible Causes**:
  - Evaluations running but not triggering adjustments
  - Evaluation conditions not met
  - Training paused or stuck

### 2. **R:R Ratio Not Adjusting**
- **Problem**: R:R ratio has 0 adjustments (stuck at 2.0)
- **Impact**: May be too high, preventing profitable trades
- **Action**: Review R:R adjustment logic

### 3. **Quality Filters Being Loosened**
- **Observation**: Recent adjustments lowered quality thresholds
- **Impact**: May be allowing too many low-quality trades
- **Current**: min_action_confidence: 0.2, min_quality_score: 0.5

---

## 📈 Adjustment Trends

### Entropy Coef Evolution
- Started at: 0.025 (base config)
- Recent: 0.085 (increased for exploration)
- Trend: **Increasing** (system wants more exploration)

### Quality Filters Evolution
- Confidence: 0.25 → 0.2 (lowered - allowing more trades)
- Quality Score: 0.6 → 0.5 (lowered - allowing more trades)
- Trend: **Loosening** (system responding to "too many trades" by tightening, but then loosening again)

### Stop Loss
- Current: 1.0% (volatility-adjusted)
- Trend: **Decreasing** (from 1.3% → 1.0%)

---

## 💡 Recommendations

### Immediate Actions

1. **Investigate Missing Evaluations**
   - Check if `should_evaluate()` is being called
   - Verify evaluation conditions are met
   - Check if evaluations are running but not triggering adjustments

2. **Review R:R Ratio Adjustment**
   - R:R ratio is not being adjusted (0 adjustments)
   - Current value (2.0) may be too high
   - Consider manual adjustment or review adjustment logic

3. **Monitor Quality Filter Impact**
   - Quality filters are being adjusted frequently (102 times)
   - Recent trend: Loosening filters
   - Monitor if this improves win rate

4. **Check Evaluation Frequency**
   - Config says every 5,000 timesteps
   - Actual adjustments every 10,000 timesteps
   - Verify evaluation is actually running

### Long-term Improvements

1. **Review Adjustment Logic**
   - Why are evaluations not triggering adjustments?
   - Check evaluation conditions and thresholds

2. **R:R Ratio Adjustment**
   - Enable or review R:R ratio adjustment logic
   - Current 2.0 may be preventing profitable trades

3. **Quality Filter Strategy**
   - System is loosening filters (may be counterproductive)
   - Consider tightening filters to improve win rate

---

## 📝 Notes

- Adaptive learning is **active and making adjustments**
- **325 total adjustments** shows system is responsive
- **Quality filters** are the most frequently adjusted parameter
- **R:R ratio** needs attention (not being adjusted)
- **Missing evaluations** at 75k and 80k need investigation

---

**Status**: ⚠️ **Active but Missing Recent Evaluations** - System is working but evaluations may not be triggering

