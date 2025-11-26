# Stop Loss Bug Fixed

## Problem Identified
Stop loss was checking the **wrong value** - it was using price change from previous step instead of loss from entry price.

## Root Cause
```python
# OLD (WRONG):
if stop_loss_enabled and (self.state.position * price_change) < 0:
    loss_pct = abs(price_change)  # This is change from LAST STEP, not from ENTRY!
```

**Example:**
- Entry price: $100
- Current price: $97.50 (2.5% loss from entry)
- Previous price: $97.60
- `price_change = (97.50 - 97.60) / 97.60 = -0.1%`
- `loss_pct = 0.1%` (WRONG - this is from last step, not entry!)
- Stop loss threshold: 2.5%
- `0.1% >= 2.5%` → FALSE → stop loss doesn't trigger!

But the actual loss from entry is 2.5%, which should trigger!

## The Fix
Now checks loss from **entry price**:

```python
# NEW (CORRECT):
if stop_loss_enabled and self.state.position != 0 and self.state.entry_price is not None:
    # Calculate loss from ENTRY price
    if self.state.position > 0:  # Long position
        loss_pct = (self.state.entry_price - current_price) / self.state.entry_price
    else:  # Short position
        loss_pct = (current_price - self.state.entry_price) / self.state.entry_price
    
    if loss_pct >= self.stop_loss_pct:
        # Stop loss hit - force close position
```

## Expected Impact
- Stop loss will now properly trigger at 2.5% loss from entry
- Prevents positions from accumulating large losses (like -$55K in one episode)
- Max drawdown should match actual losses
- Current PnL should not exceed stop loss threshold

## Next Steps
1. Restart backend to apply fix
2. Monitor if stop loss triggers correctly
3. Verify that large losses are prevented
