# Episode Termination Fix

## Issue
Episodes were terminating very early (40 steps) instead of running for the full 10,000 steps.

## Root Cause Analysis

### Potential Causes Identified:
1. **IndexError in `_extract_timeframe_features`**: Line 262 could cause IndexError if boundary checks fail
2. **Data boundary issues**: Episodes might terminate if `current_step` exceeds data bounds
3. **Uncaught exceptions**: Exceptions in `step()` or `_get_state_features()` might not be handled gracefully

## Fixes Applied

### 1. Enhanced Boundary Check in `_extract_timeframe_features` (Line 261-266)
**Before:**
```python
if current_idx >= 20:
    avg_volume = full_data["volume"].iloc[current_idx-20:current_idx].mean()
```

**After:**
```python
if current_idx >= 20:
    # CRITICAL FIX: Add boundary check to prevent IndexError
    start_vol_idx = max(0, current_idx - 20)
    end_vol_idx = min(current_idx, len(full_data))
    if end_vol_idx > start_vol_idx:
        avg_volume = full_data["volume"].iloc[start_vol_idx:end_vol_idx].mean()
    else:
        avg_volume = window["volume"].iloc[-1] if len(window) > 0 else 1.0
```

### 2. Data Boundary Check Before State Feature Extraction (Line 799-820)
**Added:**
- Check if `safe_step >= len(primary_data) - self.lookback_bars` before calling `_get_state_features()`
- If too close to data end, terminate episode gracefully with warning
- Added try/except around `_get_state_features()` to catch IndexError/KeyError

### 3. Exception Handling in Training Loop (Line 672-685)
**Added:**
- Try/except block around `env.step()` to catch exceptions
- On exception, terminate episode gracefully with error logging
- Prevents training from crashing on unexpected errors

## Expected Behavior After Fix

1. **Episodes should run for full 10,000 steps** (or until data ends)
2. **Early termination warnings** will be logged if data bounds are exceeded
3. **Exceptions will be caught and logged** instead of crashing training
4. **Graceful degradation** - episodes terminate cleanly on errors

## Monitoring

After applying fixes, monitor:
- Episode lengths should be close to 10,000 steps
- Check backend logs for any "[WARNING]" or "[ERROR]" messages
- Verify no IndexError or KeyError exceptions are occurring

## Next Steps

1. Restart training to apply fixes
2. Monitor episode lengths - should see ~10,000 steps
3. Check backend logs for warnings/errors
4. Verify metrics improve with longer episodes

