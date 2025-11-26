# Fixes Summary - Zero Metrics + Continuing Losses

## ✅ Fix 1: Mean (Last 10 Episodes) Zeros - FIXED

**Root Cause:** Episode metrics lists reset when resuming from checkpoint

**Solution Applied:**
- Added database fallback to calculate mean metrics from `trading_journal.db`
- Reads last 10 episodes from database and calculates means
- Provides historical context even after checkpoint resume

**File:** `src/api_server.py`

## ✅ Fix 2: Continuing Losses - COMPREHENSIVE FIXES

### A. Per-Trade R:R Tracking ✅

**Problem:** Aggregate R:R is lagging indicator - agent doesn't see immediate connection

**Solution:**
- Track R:R for each trade at exit (not just aggregate)
- Calculate: R:R = (exit_price - entry_price) / (entry_price - stop_loss_price)
- Store in `recent_trades_rr` list for immediate feedback

**Implementation:**
- Added `self.recent_trades_rr = []` in `__init__`
- Calculate per-trade R:R when trade exits (stop loss, position closed, position reversed)

### B. Per-Trade R:R Penalty ✅

**Problem:** Trades exiting before achieving target R:R

**Solution:**
- Penalize trades that exit before target R:R (2.0:1)
- Penalty: 30% of reward (scaled by how far below target)
- Immediate negative feedback for poor trade management

**Implementation:**
```python
if last_trade_rr > 0 and last_trade_rr < required_rr:
    per_trade_rr_penalty = 0.30 * (required_rr - last_trade_rr) / required_rr
    reward -= per_trade_rr_penalty
```

### C. Per-Trade R:R Bonus ✅

**Solution:**
- Reward trades that achieve good R:R (>= 2.0:1)
- Bonus: Up to 20% for achieving target R:R or better
- Immediate positive feedback for good trade management

### D. Strengthened Aggregate R:R Penalty ✅

**Problem:** Aggregate penalty (30%) wasn't strong enough

**Solution:**
- Increased maximum penalty from 30% to 50%
- More aggressive penalty for poor aggregate R:R

### E. Reward Function Logging ✅

**Solution:**
- Log reward components every 500 steps
- Shows aggregate R:R, penalties, bonuses
- Helps verify reward function is working

## Expected Impact

### Metrics Should Improve:
- **R:R:** 0.71:1 → Target: 2.0:1
- **Average Win:** $89.96 → Target: $254
- **Expected Value:** -$28.67 → Target: +$45.76

### Agent Should Learn:
- Hold winners longer (to achieve 2.0:1 R:R)
- Exit at target R:R (not too early)
- Improve per-trade R:R over time

## Next Steps

1. **Restart backend** (to apply mean metrics fix)
2. **Restart training** (to apply per-trade R:R tracking)
3. **Monitor reward debug logs** (verify penalties/bonuses)
4. **Watch R:R improve** (should trend toward 2.0:1)

## Files Modified

1. `src/api_server.py` - Mean metrics database fallback
2. `src/trading_env.py` - Per-trade R:R tracking + penalties/bonuses

