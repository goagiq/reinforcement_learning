# Forecast Caching Bug Fix

## Problem Identified

After implementing forecast caching to speed up training with Chronos, trade performance degraded significantly:
- **Win rate dropped**: From ~45% to 35% in recent trades
- **Losses widened**: Multiple large losses (>$500) occurring
- **Total PnL**: -$3,605 in last 100 trades

## Root Cause

The caching implementation had a critical bug:

1. **Incorrect forecast calculation**: When recalculating forecasts, we were calculating for `current_step` but caching under `cached_step`
2. **Stale forecasts**: Forecasts were being reused for too many steps (20 steps = very stale)
3. **No cache reset**: Cache persisted across episodes, potentially causing cross-episode contamination

### Example of the Bug:
- Step 50: Calculate forecast for step 50, cache it under key 40 (if cache_steps=20)
- Step 51: Use cached forecast from step 50 (stale by 1 step)
- Step 60: Calculate forecast for step 60, cache it under key 60
- Step 61: Use cached forecast from step 60 (stale by 1 step)

The problem: We were calculating forecasts for the wrong step, leading to inconsistent state features.

## Fixes Applied

### 1. Fixed Cache Calculation Logic
**Before:**
```python
forecast = self.predict(price_data, current_step)  # Wrong step!
self._cache[cached_step] = forecast
```

**After:**
```python
forecast = self.predict(price_data, cached_step)  # Correct step
self._cache[cached_step] = forecast
```

Now all steps in the same cache interval use the forecast calculated at the start of that interval, ensuring consistency.

### 2. Reduced Cache Staleness
- **Before**: `cache_steps: 20` (forecasts updated every 20 steps)
- **After**: `cache_steps: 5` (forecasts updated every 5 steps)
- **Impact**: 4x fresher forecasts while still providing ~5x speedup

### 3. Added Cache Reset on Episode Reset
- Cache is now cleared at the start of each episode
- Prevents cross-episode contamination
- Ensures each episode starts with fresh forecasts

## Expected Improvements

After this fix:
1. **Forecast accuracy**: Forecasts are now calculated correctly for the cached step
2. **Reduced staleness**: Forecasts update 4x more frequently (every 5 steps vs 20)
3. **Episode isolation**: Each episode starts with a clean cache
4. **Performance**: Should see improved win rate and reduced large losses

## Configuration

The cache steps can be adjusted in `configs/train_config_adaptive.yaml`:

```yaml
environment:
  reward:
    forecast_cache_steps: 5  # Adjust based on needs
```

**Recommendations:**
- **Chronos enabled**: Use 5-10 steps (good balance of speed and freshness)
- **Simple predictor**: Use 3-5 steps (simple predictor is fast, can update more often)
- **Maximum speed**: Use 10-20 steps (but may reduce forecast quality)

## Monitoring

After restarting training, monitor:
1. **Win rate**: Should improve back to previous levels (~45%+)
2. **Large losses**: Should decrease (fewer >$500 losses)
3. **Total PnL**: Should trend positive
4. **Training speed**: Should still be faster than without caching (~5x speedup)

## Next Steps

1. **Restart training** to apply the fix
2. **Monitor performance** over next 100-200 trades
3. **Adjust cache_steps** if needed (if still slow, increase; if forecasts seem stale, decrease)
4. **Consider disabling Chronos** during training if performance is still poor (use simple predictor instead)

## Technical Details

The cache now works as follows:
- Step 0-4: Use forecast calculated at step 0
- Step 5-9: Use forecast calculated at step 5
- Step 10-14: Use forecast calculated at step 10
- etc.

This ensures:
- All steps in the same interval use the same forecast (consistency)
- Forecasts are calculated for the correct step (accuracy)
- Cache is cleared between episodes (isolation)

