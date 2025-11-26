# Timestamp Filter Fix - Always Filter Old Trades

## Problem Identified

**Issue**: When starting fresh training (without checkpoint), the `checkpoint_resume_timestamp` was `None`, which caused ALL old trades from previous sessions to be displayed in the Performance Monitoring dashboard.

**Result**: User sees -$84K P&L immediately when starting training, even though they haven't made any new trades yet.

**Root Cause**:
- `checkpoint_resume_timestamp` was only set when resuming from a checkpoint
- When `None`, the API returns ALL trades from the journal
- Old unprofitable trades from previous sessions are included

---

## Fix Applied

### Changed Logic: Always Set Timestamp When Training Starts

**Before**:
```python
checkpoint_resume_timestamp = None
if checkpoint_path_to_use:
    checkpoint_resume_timestamp = datetime.now().isoformat()  # Only set if checkpoint exists
```

**After**:
```python
# ALWAYS set training start timestamp when training begins
# This ensures the monitoring panel filters out old trades from previous sessions
training_start_timestamp = datetime.now().isoformat()  # Always set
if checkpoint_path_to_use:
    print(f"[INFO] Resuming from checkpoint: {checkpoint_path_to_use}")
print(f"[INFO] Training start timestamp: {training_start_timestamp} (will filter trades before this time)")
```

### Files Modified

1. **`src/api_server.py`** (lines 1298-1311):
   - Changed to ALWAYS set timestamp when training starts
   - Renamed variable to `training_start_timestamp` for clarity
   - Updated logging messages

2. **`src/api_server.py`** (lines 1354-1366):
   - Changed to ALWAYS set timestamp when training starts
   - Ensures placeholder entry also has timestamp

3. **`frontend/src/components/MonitoringPanel.jsx`** (line 266):
   - Updated label from "Filtered since checkpoint resume" to "Filtered since training start"

---

## How It Works Now

### When Training Starts (Fresh or Resume)

1. **Timestamp is ALWAYS set** to current time (`datetime.now().isoformat()`)
2. **Stored in** `active_systems["training"]["checkpoint_resume_timestamp"]`
3. **Frontend reads** timestamp from `/api/training/status`
4. **API filters** trades using `WHERE timestamp >= ?` query
5. **Result**: Only trades from AFTER training start are shown

### Example Flow

```
1. User starts training (fresh, no checkpoint)
   → timestamp = "2025-11-25T15:20:00.123456"
   → Stored in active_systems

2. Frontend loads Performance Monitoring
   → Fetches timestamp from /api/training/status
   → Gets: "2025-11-25T15:20:00.123456"

3. Frontend requests performance data
   → GET /api/monitoring/performance?since=2025-11-25T15:20:00.123456

4. Backend filters trades
   → SELECT * FROM trades WHERE timestamp >= '2025-11-25T15:20:00.123456'
   → Returns only NEW trades from this training session

5. Dashboard shows
   → Total Trades: 0 (initially)
   → Total P&L: $0.00 (initially)
   → "Filtered since training start" badge shown
```

---

## Expected Behavior After Fix

### Fresh Training Start
- ✅ **Total Trades**: 0 (initially, before any new trades)
- ✅ **Total P&L**: $0.00 (initially)
- ✅ **Badge**: "Filtered since training start" shown
- ✅ **Old trades**: Hidden (filtered out)

### Resume from Checkpoint
- ✅ **Total Trades**: Only trades since resume timestamp
- ✅ **Total P&L**: Only P&L from new trades
- ✅ **Badge**: "Filtered since training start" shown
- ✅ **Old trades**: Hidden (filtered out)

### During Training
- ✅ **New trades**: Appear as they're executed
- ✅ **P&L**: Updates with only new trades
- ✅ **Historical**: Old trades remain hidden

---

## Verification

To verify the fix works:

1. **Check API endpoint**:
   ```bash
   curl http://localhost:8200/api/training/status | grep checkpoint_resume_timestamp
   ```
   Should return a timestamp (not null)

2. **Check filtered trades**:
   ```bash
   curl "http://localhost:8200/api/monitoring/performance?since=2025-11-25T15:20:00"
   ```
   Should return only trades after that time

3. **Check dashboard**:
   - Should show "Filtered since training start" badge
   - Should show 0 trades initially (if no new trades yet)
   - Should show $0.00 P&L initially (if no new trades yet)

---

## Database State

Current journal state:
- **Total trades**: 10,198
- **Total P&L**: -$314,225.74
- **Date range**: 2025-11-24T08:34:21 to 2025-11-25T15:17:16

After fix:
- When training starts, timestamp filter will exclude all 10,198 old trades
- Dashboard will show 0 trades, $0.00 P&L initially
- As new trades execute, they'll appear in the dashboard

---

## Summary

✅ **Fixed**: Timestamp is now ALWAYS set when training starts
✅ **Result**: Old trades are filtered out, dashboard shows only new trades
✅ **User Experience**: No more confusion about seeing -$84K when starting fresh training

## Important Note for Checkpoint Resume

When resuming from checkpoint (like checkpoint_1,000,000.pt), the timestamp filter will:
- ✅ Filter out ALL trades from the previous training session that created the checkpoint
- ✅ Only show NEW trades executed after resuming training
- ✅ Display "Filtered since training start" badge in the dashboard

**If you're still seeing old trades after resuming:**
1. **Restart the API server** to apply the fix
2. **Refresh the frontend** to get the new timestamp
3. The timestamp is set when training starts, so any trades logged BEFORE that time will be filtered

**Current status:**
- Total trades in DB: 11,248 (from previous sessions)
- When you resume training, these should ALL be filtered out
- Only new trades after the resume timestamp will be shown

