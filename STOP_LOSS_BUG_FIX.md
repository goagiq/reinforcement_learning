# Stop Loss Not Working - Bug Found

## Problem
Training Progress shows very high negative PnL (-$55,085.82 for episode 146), but Max Drawdown is only 1.0%. This suggests stop loss is not being enforced properly.

## Root Cause

Looking at line 972-976 in `trading_env.py`:

```python
stop_loss_enabled = True
if stop_loss_enabled and (self.state.position * price_change) < 0:  # Position is losing
    loss_pct = abs(price_change)
    
    # If loss exceeds stop loss, force close position
    if loss_pct >= self.stop_loss_pct:
```

**THE BUG:** `price_change` is the price change from the **previous step**, not the loss from **entry price**!

- `price_change = (current_price - prev_price) / prev_price` (calculated earlier)
- This is the change from last step, NOT the loss from entry
- Stop loss should check: `loss_from_entry = abs(current_price - entry_price) / entry_price`

So if:
- Entry price: $100
- Current price: $97.50 (2.5% loss)
- Previous price: $97.60

The code checks:
- `price_change = (97.50 - 97.60) / 97.60 = -0.1%` (from last step)
- `loss_pct = 0.1%` (WRONG - too small!)
- Stop loss threshold: 2.5%
- `0.1% >= 2.5%` → FALSE → stop loss doesn't trigger!

But the actual loss from entry is 2.5%, which should trigger the stop loss!

## The Fix

Check loss from **entry price**, not from previous step:

```python
if stop_loss_enabled and self.state.position != 0 and self.state.entry_price is not None:
    # Calculate loss from ENTRY price (not from previous step)
    if self.state.position > 0:  # Long position
        loss_pct = (self.state.entry_price - current_price) / self.state.entry_price
    else:  # Short position
        loss_pct = (current_price - self.state.entry_price) / self.state.entry_price
    
    # If loss exceeds stop loss, force close position
    if loss_pct >= self.stop_loss_pct:
        # Stop loss hit - force close position
```

This ensures stop loss is checked against the entry price, not the previous step's price.

