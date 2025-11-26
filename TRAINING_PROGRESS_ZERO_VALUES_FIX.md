# Training Progress Zero Values Fix

## Issue

Training Progress panel was showing all zeros:
- Current Episode Trades: 0
- Current PnL: $0.00
- Current Equity: $0.00
- Current Win Rate: 0.0%
- Max Drawdown: 0.0%
- Mean (Last 10 Episodes): All 0.00

## Root Cause

The API endpoint had complex conditional logic that was preventing current episode metrics from being displayed when:
- Training just started (`current_episode_length` might be 0)
- No completed episodes yet
- Metrics exist in trainer attributes but weren't being read

## Fix Applied

Modified `/api/training/status` endpoint in `src/api_server.py` to use a simpler priority-based approach:

### Priority 1: Read from Trainer Attributes (Primary Source)
- Always reads `trainer.current_episode_*` attributes first
- These are updated every step from `step_info` in the training loop
- Should always reflect the current state

### Priority 2: Fallback to Environment State
- If trainer attributes are 0 but training is active (`timestep > 0`)
- Reads directly from `trainer.env` state
- Handles cases where step_info hasn't updated trainer attributes yet

### Priority 3: Latest Completed Episode
- If still no values, shows latest completed episode metrics
- Only if there are completed episodes from current session

### Priority 4: Show Zeros
- Only shows zeros when training truly hasn't started

## Expected Result

Now the Training Progress panel should:
- ✅ Show current episode metrics in real-time as training progresses
- ✅ Update values every step (from step_info)
- ✅ Show 0 only when training hasn't started or no trades yet
- ✅ Mean (Last 10 Episodes) calculates from completed episodes

## Testing

After restarting training:
1. Check that "Current Episode" metrics show actual values (not all zeros)
2. Verify metrics update as training progresses
3. Confirm Mean (Last 10 Episodes) populates after episodes complete

