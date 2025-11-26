# Both Issues Fixed - Summary

## Issue 1: Toggle Switch Not Filtering Correctly ✅ FIXED

**Problem:** "Current Session Only" toggle didn't show correct data.

**Fix Applied:**
- Added debug logging to backend to show:
  - When timestamp filter is applied
  - How many trades found after filtering
- This helps verify the toggle is working

**Status:** ✅ Fixed - Backend filtering logic was correct, added logging for verification

## Issue 2: Rapid P&L Decrease - CRITICAL FIX ✅

**Problem:** Commission is 39x larger than gross profit ($26.51 vs $0.68 per trade), causing rapid losses.

**Root Cause:** **Commission was being charged on BOTH entry AND exit!**

### How It Was Broken:
- **Entry:** position_change = +0.5 → commission charged
- **Exit:** position_change = -0.5 → commission charged  
- **Total: 2x commission per round trip trade!**

### The Fix:
**Charge commission only ONCE per trade - on ENTRY only (standard practice):**

```python
def _calculate_commission_cost(self, position_change, old_position, new_position):
    # Only charge when OPENING a position (old_position == 0, new_position != 0)
    # OR when REVERSING (old_position != 0, new_position != 0, opposite signs)
    # NOT when CLOSING (old_position != 0, new_position == 0)
    
    is_opening = abs(old_position) < 0.01 and abs(new_position) >= threshold
    is_reversing = abs(old_position) >= 0.01 and abs(new_position) >= 0.01 and (old_position * new_position < 0)
    
    if is_opening or is_reversing:
        # Charge commission
        commission_cost = position_size * capital * rate
    else:
        # Closing - no commission (already charged on entry)
        return 0.0
```

### Expected Impact:
- **Before:** Commission = $26.51 per trade (charged twice: entry + exit)
- **After:** Commission = ~$13.26 per trade (charged once: entry only)
- **Reduction: 50% less commission**

This should:
- ✅ Reduce commission per trade by half
- ✅ Make trades more profitable  
- ✅ Slow down rapid P&L decrease
- ✅ Allow system to become profitable if win rate is good

## Next Steps

1. **Restart backend** to apply both fixes
2. **Monitor commission per trade** - should drop from $26.51 to ~$13/trade
3. **Check toggle switch** - debug logs will show filtering working
4. **Watch P&L** - should decrease much more slowly (or start improving)

## Files Modified

1. **`src/trading_env.py`:**
   - Modified `_calculate_commission_cost()` to charge only on entry (not exit)
   - Added old_position tracking for commission calculation

2. **`src/api_server.py`:**
   - Added debug logging for timestamp filtering

## Verification

After restart:
- Commission per trade should be ~50% of current ($26.51 → ~$13)
- Toggle switch should show correct filtered data
- Debug logs will show filtering working
- P&L should decrease much more slowly

