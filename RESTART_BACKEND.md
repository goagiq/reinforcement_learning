# How to Restart Backend Server

## Quick Restart (Non-Interrupting)

The backend server needs to be restarted to apply the 404 error handling fix.

### Option 1: Restart via stop_ui.py + start_ui.py (Recommended)

1. **Stop the backend** (keeps training data safe):
   ```bash
   python stop_ui.py
   ```
   This will:
   - Stop the backend API server
   - Stop the frontend dev server
   - **Training will pause** (but checkpoints are saved)

2. **Start again**:
   ```bash
   python start_ui.py
   ```
   This will:
   - Start the backend API server with new code
   - Start the frontend dev server
   - **Training can be resumed** from the last checkpoint

### Option 2: Manual Restart (If using separate terminals)

If you started services manually:

1. **Find the backend process**:
   ```bash
   # On Windows (PowerShell)
   Get-Process python | Where-Object {$_.CommandLine -like "*api_server*"}
   
   # Or check the terminal running uvicorn
   ```

2. **Stop the backend**:
   - Press `Ctrl+C` in the terminal running the backend
   - Or kill the process

3. **Restart the backend**:
   ```bash
   uvicorn src.api_server:app --host 0.0.0.0 --port 8200 --reload
   ```

### Option 3: Wait for Next Checkpoint (Safest)

If training is close to a checkpoint save:

1. **Check current progress**:
   ```bash
   curl http://localhost:8200/api/training/status
   ```

2. **Wait for checkpoint** (saves every 10,000 steps)

3. **Then restart** using Option 1

## What Happens to Training?

- ✅ **Checkpoints are safe** - All saved checkpoints remain
- ✅ **Training can resume** - Use the latest checkpoint to resume
- ⚠️ **Current training session** - Will pause, but can be resumed

## Verify Fix After Restart

After restarting, test the fix:

```bash
# This should return JSON error, not 404
curl "http://localhost:8200/api/models/checkpoint/info?checkpoint_path=models/checkpoint_10000.pt"
```

**Expected response** (HTTP 200 with error JSON):
```json
{
  "error": "Checkpoint not found",
  "path": "models/checkpoint_10000.pt",
  "exists": false,
  "message": "Checkpoint file not found: models/checkpoint_10000.pt"
}
```

**NOT** a 404 error!

## Resume Training After Restart

If training was interrupted:

1. **Go to Training tab** in the UI
2. **Select latest checkpoint** (should auto-select)
3. **Click "Start Training"**
4. Training will resume from the last checkpoint

---

**Note**: The 404 error fix is non-critical for training. You can continue training without restarting, but the 404 errors will continue to appear in logs until you restart.

