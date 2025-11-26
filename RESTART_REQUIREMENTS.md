# Backend Restart Requirements

## Changes Made

### 1. Configuration File Changes (`configs/train_config_adaptive.yaml`)
- ✅ Added `max_daily_loss: 0.05` to risk_management
- ✅ Changed `max_consecutive_losses: 10` to `5`

### 2. Python Code Changes
- ✅ `src/api_server.py`: Fixed training progress metrics display
- ✅ `src/trading_env.py`: Changed R:R enforcement floor from 0.7 to 1.0

## Do You Need to Restart?

### ✅ **YES, restart the backend if:**

1. **You want the API server changes to take effect immediately:**
   - Training Progress panel metrics fix (zero values issue)
   - This requires a backend restart to load new code

2. **Training is NOT currently running:**
   - Safe to restart anytime
   - Next training session will use all new changes

### ⚠️ **Current Training Session:**

**If training IS currently running:**
- Current session uses OLD config values (max_consecutive_losses: 10, etc.)
- Current session uses OLD code (R:R floor: 0.7, etc.)
- **Backend restart alone won't affect running training**
- **To use new settings, you need to:**
  1. Stop current training
  2. Restart backend (optional, but recommended)
  3. Start new training session

## How Config Loading Works

### Configuration Files:
- ✅ Loaded **when training starts** (not on backend startup)
- ✅ Changes are picked up automatically on next training session
- ⚠️ Currently running training won't see config changes

### Python Code:
- ✅ `api_server.py`: Loaded on backend startup → **needs restart**
- ✅ `trading_env.py`: Loaded when environment is created (training start) → **auto-updates on next training**

## Recommendation

**Restart the backend now if:**
- ✅ Training is NOT running (safe to restart)
- ✅ You want the Training Progress metrics fix immediately
- ✅ You want to be ready for next training session

**Wait to restart if:**
- ⚠️ Training is currently running and you want to let it finish
- ⚠️ You want to continue current session with old settings

**After restart:**
- ✅ Backend will use new `api_server.py` code (metrics fix)
- ✅ Next training session will load new config values
- ✅ Next training session will use new `trading_env.py` code (R:R floor)

## Quick Check: Is Training Running?

Check the Systems tab or Training tab to see if training is active:
- If "Training System" shows "running" → Training is active
- If "Training System" shows "stopped" → Safe to restart

