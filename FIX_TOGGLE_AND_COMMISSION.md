# Fix Toggle Switch and Rapid P&L Decrease

## Issue 1: Toggle Switch Not Showing Correct Data

**Problem:** The toggle switch for "Current Session Only" doesn't filter correctly.

**Root Cause:** Need to verify:
1. Is `checkpointResumeTimestamp` being fetched correctly?
2. Is the timestamp format correct when passed to backend?
3. Is the SQL query comparing timestamps correctly?

**Fix Needed:**
- Verify timestamp format (ISO string)
- Check SQL timestamp comparison
- Add logging to see what timestamp is being used

## Issue 2: Rapid P&L Decrease

**Problem:** Total P&L decreases rapidly (loses money fast).

**Root Cause from Analysis:**
- **Average Commission:** $26.51 per trade
- **Average Gross P&L:** $0.68 per trade
- **Commission is 39x larger than gross profit!**

This means:
- Every trade loses money because commission > gross profit
- Commission dwarfs any profits
- System can't be profitable at this rate

## Why Commission is So High

Commission calculation:
```python
commission_cost = abs(position_change) * initial_capital * commission_rate
```

For $26.51 commission:
- If commission_rate = 0.0001 (0.01%): position_change = 2.65 (265%) - **impossible!**
- If commission_rate = 0.0003 (0.03%): position_change = 0.88 (88%) - **possible but high**

**Problem:** Position changes might be too large, OR commission is being charged incorrectly.

## Solutions

### 1. Fix Toggle Switch
- Verify timestamp is in correct format
- Check SQL query filters correctly
- Add debug logging

### 2. Fix Commission Calculation
- Check if commission is being double-charged (entry + exit)
- Verify commission_rate is correct
- Check if position_change calculation is wrong
- Consider reducing commission_rate if too high

### 3. Reduce Overtrading
- Increase action_threshold to reduce trade frequency
- Let winners run longer (less frequent exits)

