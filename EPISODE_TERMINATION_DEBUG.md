# Episode Termination Debug - Enhanced Fix

## Issue
Episodes are still terminating in 1 step during actual training, even though the test passes.

## Root Cause Analysis

The termination check in `step()` happens AFTER `self.current_step += 1`, so:
1. Episode starts at position X
2. First step: `current_step` becomes X+1
3. Check: `remaining_data = len(data) - safe_step`
4. If `remaining_data < lookback_bars + 1`, terminate

## Enhanced Fixes Applied

### 1. Termination Check (src/trading_env.py, line ~1994)
- Changed from `remaining_data <= self.lookback_bars` to `remaining_data < (self.lookback_bars + 1)`
- Added buffer: need `lookback_bars + 1` remaining (current bar + lookback for next state)
- Enhanced error logging to identify when episodes terminate on step 1-2

### 2. Reset Logic (src/trading_env.py, line ~1301)
- Added `min_required_remaining = self.lookback_bars + 1` check
- If data is too short, calculate `earliest_safe_start` to maximize episode length
- Better error messages when data is insufficient

### 3. Debug Logging
- Added `[ERROR]` messages when episodes terminate on step <= lookback_bars + 2
- Logs data_len, lookback, max_steps for debugging
- Enhanced `[WARN]` messages with more context

## Next Steps

1. **Check training console logs** for:
   - `[ERROR] Episode terminated on step X!`
   - `[WARN] Episode terminating early:`
   - `[ERROR] Episode reset: Data too short!`

2. **Verify data length** in training:
   - Check if data is being loaded correctly
   - Verify `max_episode_steps` vs actual data length

3. **Check for exceptions**:
   - Look for `[ERROR] Exception in env.step()` messages
   - These would cause immediate termination

4. **DecisionGate impact**:
   - If DecisionGate is enabled, check if it's causing issues
   - Verify `should_execute()` isn't rejecting all actions

## Testing

Run `test_episode_termination.py` - should pass with episodes running 100+ steps.

If test passes but training fails, the issue is likely:
- Different data being used in training
- Configuration differences (max_steps, lookback_bars)
- Exceptions being caught silently
- DecisionGate filtering all actions

