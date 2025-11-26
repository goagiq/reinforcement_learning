# Fix Toggle Switch and Rapid P&L Decrease - Root Causes Found

## Issue 1: Toggle Switch Not Filtering Correctly

**Problem:** "Current Session Only" toggle doesn't show correct data.

**Root Cause:** Need to verify timestamp format and SQL query filtering.

**Status:** Backend code looks correct - will add debug logging.

## Issue 2: Rapid P&L Decrease - CRITICAL

**Analysis Results:**
- **100 trades in last hour**
- **Gross P&L:** $68.29 ($0.68 per trade)
- **Commission:** $2,651.24 ($26.51 per trade)
- **Net P&L:** -$2,582.94
- **Commission is 39x larger than gross profit!**

### Why Commission is So High

Commission calculation:
```python
commission_cost = abs(position_change) * initial_capital * commission_rate
```

For $26.51 commission with rate 0.0001 (0.01%):
- position_change = $26.51 / ($100,000 * 0.0001) = 2.65 (265%!)

**This is impossible!** Position change can't be > 1.0 (100%).

### Possible Causes:

1. **Commission charged twice** (entry + exit)
   - Entry: position_change = 0.5 → commission = $5
   - Exit: position_change = 0.5 → commission = $5
   - Total: $10 per round trip
   - But we're seeing $26.51, so maybe 2.65x charged?

2. **Commission rate is wrong**
   - Config shows 0.0001, but maybe 0.0003 is being used?
   - $26.51 / ($100,000 * 0.0003) = 0.88 (88% position change) - **more reasonable**

3. **Position change calculated incorrectly**
   - Maybe cumulative instead of difference?
   - Or includes unrealized PnL somehow?

### Next Steps:

1. Check if commission is charged on entry AND exit
2. Verify which commission_rate is actually being used
3. Check position_change calculation
4. Add logging to see commission per trade

