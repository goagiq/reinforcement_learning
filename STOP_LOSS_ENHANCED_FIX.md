# Stop Loss Enhancement - More Explicit Checking

## Current Logic Analysis

The stop loss logic at line 972-976 is actually correct:
- `price_change = (current_price - entry_price) / entry_price` (from entry, line 965)
- `loss_pct = abs(price_change)` (line 973)

However, the condition might not catch all cases. Let me make it more explicit and robust.

## Enhanced Fix

The fix I applied makes the stop loss check more explicit:

1. **Check if position exists** (not zero)
2. **Check if entry_price exists** (not None)
3. **Calculate loss from entry price explicitly** (separate for long/short)
4. **Check against stop loss threshold**

This ensures stop loss is checked every step when there's an open position, regardless of price movement direction.

