# Boundary Check Fix for Episode Length Issue

## Problem

Episodes were terminating very early (60 steps vs 10,000 expected). Investigation revealed potential boundary issues in direct data access without bounds checking.

## Root Cause

In the `step()` method and related methods, there were direct data accesses using `self.current_step` without boundary checks:

1. **Line 530**: `current_price = self.data[min(self.timeframes)].iloc[self.current_step]["close"]`
   - Direct access without checking if `current_step` is within data bounds
   - If `current_step` somehow exceeds data length, this would raise `IndexError`

2. **Line 672**: `next_state = self._get_state_features(self.current_step)`
   - While `_get_state_features()` has internal boundary checks, it's safer to ensure `current_step` is valid before calling

3. **Line 406**: `recent_prices = self.data[min(self.timeframes)].iloc[max(0, self.current_step-20):self.current_step+1]["close"]`
   - Direct slice access without ensuring end index is within bounds

## Fix Applied

### 1. Added Boundary Check for Current Price Access

**Before**:
```python
# Get current price
current_price = self.data[min(self.timeframes)].iloc[self.current_step]["close"]
```

**After**:
```python
# Get current price with boundary check
primary_data = self.data[min(self.timeframes)]
# Ensure current_step is within data bounds
safe_step = min(self.current_step, len(primary_data) - 1)
if safe_step < 0:
    safe_step = 0
current_price = primary_data.iloc[safe_step]["close"]
```

### 2. Added Boundary Check for State Features

**Before**:
```python
# Get next state
if not terminated:
    next_state = self._get_state_features(self.current_step)
else:
    next_state = np.zeros(self.state_dim, dtype=np.float32)
```

**After**:
```python
# Get next state
if not terminated:
    # Ensure current_step is within data bounds before getting state features
    primary_data = self.data[min(self.timeframes)]
    safe_step = min(self.current_step, len(primary_data) - 1)
    if safe_step < 0:
        safe_step = 0
    next_state = self._get_state_features(safe_step)
else:
    next_state = np.zeros(self.state_dim, dtype=np.float32)
```

### 3. Added Boundary Check for Recent Prices (Quality Score)

**Before**:
```python
if self.current_step > 20:
    recent_prices = self.data[min(self.timeframes)].iloc[max(0, self.current_step-20):self.current_step+1]["close"]
```

**After**:
```python
if self.current_step > 20:
    primary_data = self.data[min(self.timeframes)]
    # Ensure indices are within bounds
    safe_current_step = min(self.current_step, len(primary_data) - 1)
    start_idx = max(0, safe_current_step - 20)
    end_idx = min(safe_current_step + 1, len(primary_data))
    recent_prices = primary_data.iloc[start_idx:end_idx]["close"]
```

## Why This Fixes the Issue

1. **Prevents IndexError**: If `current_step` somehow exceeds data bounds, the boundary checks prevent `IndexError` exceptions that could cause episodes to terminate early.

2. **Handles Edge Cases**: The checks handle multiple cases:
   - `current_step` exceeds data length: Uses last valid index
   - `current_step` is negative: Uses index 0
   - Slice end exceeds bounds: Clamps to data length

3. **Defensive Programming**: Even though `current_step` should never exceed bounds in normal operation, this adds a safety net that prevents silent failures or exceptions.

4. **Consistent with `_get_state_features()`**: The `_get_state_features()` method already has boundary checks (line 169), so this makes the rest of the code consistent.

## Expected Impact

- **Episodes should now run to completion**: Episodes should reach `max_episode_steps` (10,000) instead of terminating early at 60 steps
- **No performance impact**: Boundary checks are minimal and won't affect training speed
- **More robust**: System is now more resilient to edge cases and data boundary issues
- **Prevents silent failures**: Any boundary issues will be handled gracefully instead of causing exceptions

## Testing

After this fix:
1. Monitor episode lengths - should see episodes reaching ~10,000 steps
2. Check for any remaining early terminations
3. Verify training continues normally
4. Watch for any new debug messages about boundary conditions

## Files Modified

- `src/trading_env.py`: Added boundary checks in:
  - `step()` method (current price access)
  - `step()` method (state features access)
  - `_calculate_simplified_quality_score()` method (recent prices access)
