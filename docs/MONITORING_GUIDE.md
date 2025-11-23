# Training Monitoring Guide

## Current Status: IMPROVING ✅

### Key Positive Indicators (as of latest check)
- **Mean PnL (Last 10 Episodes)**: +$745.01 (POSITIVE - was -$2,104.25)
- **Mean Win Rate (Last 10 Episodes)**: 40.0% (Above breakeven ~34%)
- **Overall Win Rate**: 33.3% (Improved from 16.7%)
- **Current Episode Trades**: 4 trades (Trades are happening)

### What to Monitor

#### 1. **Mean Metrics (Last 10 Episodes)** - Most Important
   - **Mean PnL**: Should stay positive or continue improving
   - **Mean Win Rate**: Should stay above 34% (breakeven)
   - **Mean Equity**: Should trend upward
   - These reflect recent performance and are more reliable than overall metrics

#### 2. **Trade Count**
   - **Current Episode Trades**: Should see 2-8 trades per episode
   - **Total Trades**: May appear low due to cumulative tracking, but current episode activity is what matters
   - If current episode trades drop to 0-1 consistently, filters may be too strict

#### 3. **Win Rate Trends**
   - **Overall Win Rate**: Should gradually improve as more profitable episodes complete
   - **Current Win Rate**: Should fluctuate but trend toward 35-45%
   - Watch for sustained periods below 30% (concerning)

#### 4. **Profitability**
   - **Current PnL**: May fluctuate, but should trend positive over time
   - **Mean PnL (Last 10)**: This is the key metric - should stay positive
   - **Equity**: Should trend upward over time

#### 5. **Episode Length**
   - Should be close to max_episode_steps (10,000)
   - If episodes are consistently short (<100 steps), there may be exceptions or data issues

#### 6. **Rewards**
   - **Latest Reward**: May be negative, but should improve over time
   - **Mean Reward (Last 10)**: Should trend toward positive or less negative

### Red Flags to Watch For

1. **Mean PnL (Last 10) drops below -$500**
   - Indicates recent episodes are losing money
   - May need to adjust filters or parameters

2. **Mean Win Rate (Last 10) drops below 30%**
   - Below breakeven, system is losing money
   - May need to tighten quality filters

3. **Current Episode Trades consistently 0-1**
   - Filters may be too strict
   - May need to relax action_threshold or quality filters

4. **Episodes consistently very short (<100 steps)**
   - May indicate exceptions or data boundary issues
   - Check backend logs

5. **Mean Reward (Last 10) becomes very negative (<-2.0)**
   - Agent is not learning effectively
   - May need to adjust learning rate or exploration

### Green Flags (Good Signs)

1. **Mean PnL (Last 10) stays positive or improving**
2. **Mean Win Rate (Last 10) stays above 35%**
3. **Current Episode Trades: 2-8 per episode**
4. **Overall metrics gradually improving**
5. **Episode lengths close to max_episode_steps**

### Recent Fixes Applied

1. **Adaptive Trainer Threshold**: Increased from 2.0 to 10.0 trades/episode
   - Prevents false filter tightening based on evaluation data
   - Evaluation episodes may show different behavior than training

2. **Relaxed Filter Settings**:
   - `action_threshold`: 0.01 (1%)
   - `min_action_confidence`: 0.08
   - `min_quality_score`: 0.25
   - `max_consecutive_losses`: 10

3. **Auto-Resume Logic**: Trading pauses auto-resume after 100 steps
   - Prevents episodes from getting stuck in paused state

### Next Steps

1. **Continue monitoring** - Current trend is positive
2. **Let training complete** - Overall metrics will improve as more profitable episodes accumulate
3. **Check backend logs** if metrics suddenly worsen
4. **Review adaptive trainer logs** if trade count drops significantly

### Expected Timeline

- **Short term (next 50-100 episodes)**: Mean metrics should stabilize or improve
- **Medium term (next 200-300 episodes)**: Overall metrics should catch up to mean metrics
- **Long term (500+ episodes)**: System should show consistent profitability

---

**Last Updated**: After adaptive trainer fix
**Status**: Monitoring - Recent performance is positive ✅

