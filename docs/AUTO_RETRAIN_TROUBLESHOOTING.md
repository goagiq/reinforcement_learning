# Auto-Retrain Troubleshooting Guide

## Why Training Didn't Trigger

If you dropped new files in the NT8 export directory but training didn't start automatically, check these:

### 1. Check Monitor Status

First, verify the monitor is running:

```bash
# Check via API
curl http://localhost:8200/api/settings/auto-retrain-status
```

**Response fields:**
- `total_files_in_directory`: Total CSV/TXT files in the directory (e.g., 70+)
- `files_detected`: **New files detected since monitor started** (starts at 0, only increments on new files)
- `known_files_count`: Files already processed and cached (won't trigger again)
- `monitor_running`: Whether the file watcher is active
- `training_job_running`: Whether a training job is actually running

**Important:** `files_detected` is **NOT** the total file count. It only counts NEW files detected since the monitor started. Existing files won't trigger retraining - only newly created/modified files will.

**If monitor shows "not_started" or "disabled":**
- Restart the API server (`start_ui.py`)
- Verify `settings.json` has `auto_retrain_enabled: true`
- Verify `nt8_data_path` is correct

### 2. File Cache Issue

The monitor uses a cache (`logs/known_files_cache.json`) to avoid reprocessing the same files. If you drop files that were already processed, they won't trigger.

**File identification:** Files are identified by `{filename}_{filesize}`. So:
- Same filename + same size = Already processed (won't trigger)
- Same filename + different size = New version (will trigger)
- Different filename = New file (will trigger)

**Solution:** If files aren't triggering but should:

1. **Clear the cache** (if files were mistakenly marked as processed):
   ```bash
   # Delete the cache file
   rm logs/known_files_cache.json
   # Or just delete specific entries if you edit the JSON
   ```

2. **Check which files are in cache**:
   ```bash
   cat logs/known_files_cache.json
   ```

### 3. Debounce Delay

The monitor waits **30 seconds** after the last file change before triggering. This prevents multiple triggers when:
- Multiple files are dropped at once
- Files are being written/copied

**What happens:**
1. File detected → Wait 30 seconds
2. If more files arrive → Reset timer, wait another 30 seconds
3. After 30 seconds of no new files → Trigger training

**Check logs for:**
```
📁 New file detected: filename.txt
🚀 Triggering retrain for N new file(s)
```

### 4. Training Already Running

If training is already in progress, new file detection will:
- Log the detection
- Queue the retraining (not implemented yet)
- NOT trigger immediately

**Check training status:**
```bash
curl http://localhost:8200/api/training/status
```

### 5. File Format/Type

The monitor only watches `.csv` and `.txt` files. Other file types are ignored.

**Your files should match:**
- Extension: `.csv` or `.txt`
- Location: The path specified in `settings.json` → `nt8_data_path`

### 6. Check Backend Logs

Look for these messages in the backend console/logs:

**Success:**
```
📁 New data detected: N file(s)
  - filename1.txt
  - filename2.txt
🚀 Auto-triggering retraining with new data...
✅ Auto-retraining triggered successfully
```

**Skipped (already known):**
```
📁 New file detected: filename.txt
# (No trigger message - file already in cache)
```

**Training in progress:**
```
📁 New data detected: N file(s)
⚠️  Training already in progress. New data detected but will not retrain yet.
```

### 7. Manual Testing

To test if auto-retrain is working:

1. **Clear the cache:**
   ```bash
   # Via API (recommended - also clears in-memory cache)
   curl -X POST http://localhost:8200/api/settings/auto-retrain/clear-cache
   
   # Or manually delete the file
   rm logs/known_files_cache.json
   ```

2. **Drop a test file:**
   - Create a new CSV/TXT file in the export directory
   - Wait 35 seconds (30s debounce + buffer)

3. **Check logs:**
   - Should see "📁 New data detected"
   - Should see "🚀 Auto-triggering retraining"
   - Training should start in the UI

### 7a. Use Diagnostic Endpoints

**Get detailed diagnostics:**
```bash
curl http://localhost:8200/api/settings/auto-retrain/diagnostics
```

This shows:
- Monitor status and configuration
- Total files in directory
- Cache contents
- Recent files detected
- Pending files waiting for debounce

**Clear cache via API:**
```bash
curl -X POST http://localhost:8200/api/settings/auto-retrain/clear-cache
```

**Manually trigger training** (bypasses file detection, uses all files in directory):
```bash
curl -X POST http://localhost:8200/api/settings/auto-retrain/trigger-manual
```
This will immediately trigger training with all CSV/TXT files found, even if they were already processed.

**⚠️ If manual trigger returns success but training doesn't start:**

1. **Check backend console logs** - Look for:
   - `🚀 Manual retrain triggered - found N file(s)`
   - `🚀 Manual retrain - calling start_training() directly`
   - `📤 TRAINING START REQUEST RECEIVED`
   - `✅✅✅ ASYNC TRAINING FUNCTION CALLED ✅✅✅`
   - Any error messages starting with `❌`

2. **Check training status:**
   ```bash
   curl http://localhost:8200/api/training/status
   ```
   - If status is "idle": Training never started (check backend logs for errors)
   - If status is "starting": Training is initializing (may take time with many files)
   - If status shows an error: Check the error message

3. **Common issues:**
   - **Stale training entry**: Previous training completed but wasn't cleaned up. The manual trigger now auto-cleans this.
   - **Training already running**: Check `curl http://localhost:8200/api/training/status` - if it shows training in progress, you need to stop it first.
   - **Initialization timeout**: With 70+ files, data loading may take 30-60 seconds. Check backend logs for progress messages.
   - **Silent failure**: Check backend console for Python exceptions or errors during `_train()` execution.

### 8. Force Trigger (If Needed)

If auto-retrain isn't working, you can manually trigger training:

```bash
# Via API
curl -X POST http://localhost:8200/api/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "device": "cuda",
    "config_path": "configs/train_config.yaml",
    "checkpoint_path": "models/best_model.pt"
  }'

# Or use the UI: Training tab → Resume Training button
```

## Quick Diagnostic Commands

```bash
# 1. Check monitor status (shows total files, detected files, etc.)
curl http://localhost:8200/api/settings/auto-retrain-status

# 2. Get detailed diagnostics (files, cache, pending, etc.)
curl http://localhost:8200/api/settings/auto-retrain/diagnostics

# 3. Check if training is running
curl http://localhost:8200/api/training/status

# 4. Clear cache (via API - recommended)
curl -X POST http://localhost:8200/api/settings/auto-retrain/clear-cache

# 5. Manually trigger training (uses all files, bypasses detection)
curl -X POST http://localhost:8200/api/settings/auto-retrain/trigger-manual

# 6. View cache file (what files were already processed)
cat logs/known_files_cache.json

# 7. Check settings
cat settings.json | grep -E "auto_retrain|nt8_data_path"
```

## Expected Behavior

When working correctly:

1. **File dropped** → Detected immediately
2. **Wait 30 seconds** (debounce)
3. **Training starts** → Auto-resumes from latest checkpoint
4. **UI notification** → Shows "Auto-retraining triggered"
5. **Training tab** → Shows training progress

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| `files_detected` shows 0 but directory has 70+ files | `files_detected` only counts NEW files since monitor started, not existing files | This is normal - only new files trigger training. Use `total_files_in_directory` to see all files. |
| No detection | Monitor not started | Restart API server |
| Detection but no training | Callback not implemented (old code) | Update to latest code |
| Files detected but skipped | Already in cache | Use `POST /api/settings/auto-retrain/clear-cache` or manually trigger |
| Training doesn't start | Training already running | Wait for current training to finish |
| Wrong path | Incorrect `nt8_data_path` | Update `settings.json` |
| Need to retrain with existing files | Files already processed | Use `POST /api/settings/auto-retrain/trigger-manual` to force retrain |

