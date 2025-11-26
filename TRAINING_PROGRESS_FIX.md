# Training Progress Display Fix

## Issue

The Training Progress panel was showing all zeros for "Current Episode" metrics:
- Current Episode Trades: 0
- Current PnL: $0.00
- Current Equity: $0.00
- Current Win Rate: 0.0%
- Max Drawdown: 0.0%
- Mean (Last 10 Episodes): All 0.00

## Root Cause

The API endpoint was only showing current episode metrics if `has_active_episode` was true, which required `current_episode_length > 0`. However:
- When training just starts, `current_episode_length` might be 0
- The metrics are updated every step from `step_info`, but weren't being displayed

## Fix Applied

Modified `/api/training/status` endpoint in `src/api_server.py`:

1. **Always read current episode metrics** from trainer attributes (updated every step)
2. **Fallback to environment state** if trainer attributes are 0 but training is active
3. **Show latest completed episode** if no current episode data is available

### Changes:
- Removed dependency on `has_active_episode` flag for displaying current metrics
- Always read from `trainer.current_episode_*` attributes first
- Added fallback to read directly from `trainer.env` if attributes are 0
- Shows 0 only when truly no data is available (training not started)

## Expected Result

Now the Training Progress panel should show:
- Current Episode metrics updated in real-time as training progresses
- Mean (Last 10 Episodes) calculated from completed episodes
- All zeros only when training hasn't started yet

