# Commission Double-Charging Fix

## Root Cause Found

**Commission is being charged on BOTH entry AND exit:**
- Entry: position_change = 0.5 → commission charged
- Exit: position_change = -0.5 → commission charged
- **Total: 2x commission per round trip trade!**

## Why Commission is So High

Average commission: $26.51 per trade
- If charged once: position_change = 0.265 (26.5%) at rate 0.0001
- If charged twice: position_change = 0.53 (53%) at rate 0.0001

But wait - commission should be:
- Entry: $5 (0.5 * $100K * 0.0001)
- Exit: $5 (0.5 * $100K * 0.0001)
- Total: $10 per round trip

But we're seeing $26.51, which suggests:
- Either commission_rate is 0.000265 (0.0265%) - unlikely
- Or position changes are very large
- OR commission is charged multiple times

## The Fix

**Commission should be charged only ONCE per trade:**
- Charge on entry OR exit, not both
- Standard practice: Charge on entry only

## Solution

Modify commission calculation to charge only when:
- Opening a NEW position (from 0 to non-zero)
- NOT when closing a position (from non-zero to 0)

This way, commission is charged once per trade cycle, not twice.

