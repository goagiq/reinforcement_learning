# Restart and Resume Training Guide

**Quick reference for restarting frontend/backend and resuming training**

---

## 🚀 Quick Steps

### 1. Stop Current Processes

**If using `start.sh`:**
```bash
# Press Ctrl+C in the terminal running start.sh
# This stops the backend API server
```

**If using `start_ui.py`:**
```bash
# Option 1: Press Ctrl+C (stops both frontend and backend)
# Option 2: Use stop script
python stop_ui.py
```

**If processes are stuck:**
```bash
# Windows PowerShell
Get-Process | Where-Object {$_.ProcessName -like "*python*" -or $_.ProcessName -like "*node*"} | Stop-Process -Force

# Or manually kill processes on ports
# Backend (port 8200)
netstat -ano | findstr :8200
taskkill /PID <PID> /F

# Frontend (port 3200)
netstat -ano | findstr :3200
taskkill /PID <PID> /F
```

---

### 2. Restart Frontend and Backend

**Recommended: Use `start_ui.py` (starts both):**
```bash
python start_ui.py
```

This will:
- ✅ Start backend API server (port 8200)
- ✅ Start frontend dev server (port 3200)
- ✅ Optionally start Kong Gateway (if Docker is running)

**Or manually:**
```bash
# Terminal 1: Backend
python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8200

# Terminal 2: Frontend
cd frontend
npm run dev
```

**Or use `start.sh` (backend only):**
```bash
./start.sh
# Then start frontend separately:
cd frontend && npm run dev
```

---

### 3. Resume Training

**Option 1: Automatic Resume (Easiest) ⭐**

```bash
python resume_training.py
```

This automatically:
- ✅ Finds your latest checkpoint
- ✅ Loads the training config
- ✅ Resumes from exactly where you left off

**With options:**
```bash
# Force GPU training
python resume_training.py --device cuda

# Use specific config
python resume_training.py --config configs/train_config_adaptive.yaml --device cuda

# Just check what checkpoint exists (don't resume)
python resume_training.py --check-only
```

**Option 2: Via Web UI**

1. Open http://localhost:3200
2. Go to **Training** panel
3. Click **"Start Training"**
4. In the form:
   - Select device (CUDA/CPU)
   - Select config file
   - **Important:** Check "Resume from checkpoint"
   - Select checkpoint (or leave blank for latest)
5. Click **"Start Training"**

**Option 3: Via API**

```bash
# Find latest checkpoint
ls -lh models/checkpoint_*.pt

# Resume from specific checkpoint
curl -X POST http://localhost:8200/api/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "device": "cuda",
    "config_path": "configs/train_config_adaptive.yaml",
    "checkpoint_path": "models/checkpoint_30000.pt"
  }'
```

**Option 4: Via CLI**

```bash
python src/train.py \
  --config configs/train_config_adaptive.yaml \
  --device cuda \
  --checkpoint models/checkpoint_30000.pt
```

---

## 📋 Complete Example Workflow

```bash
# 1. Stop everything
# (Press Ctrl+C in terminal running start.sh or start_ui.py)

# 2. Restart frontend and backend
python start_ui.py

# 3. Wait for servers to start (about 5-10 seconds)
# Check: http://localhost:3200 should load

# 4. Resume training (automatic - finds latest checkpoint)
python resume_training.py --device cuda

# OR resume via web UI:
# - Open http://localhost:3200
# - Go to Training panel
# - Click "Start Training"
# - Check "Resume from checkpoint"
# - Click "Start Training"
```

---

## 🔍 Finding Your Latest Checkpoint

**Check what checkpoints exist:**
```bash
# List all checkpoints
ls -lh models/checkpoint_*.pt

# Or via API
curl http://localhost:8200/api/models/list
```

**Checkpoints are saved:**
- Every **10,000 timesteps** automatically
- As `models/checkpoint_10000.pt`, `checkpoint_20000.pt`, etc.
- Latest checkpoint is usually the one with highest number

**Example output:**
```
checkpoint_10000.pt   (saved at 10k timesteps)
checkpoint_20000.pt   (saved at 20k timesteps)
checkpoint_30000.pt   (saved at 30k timesteps)  ← Latest
best_model.pt         (best performing model)
```

---

## ✅ Verification

**After restarting, verify:**

1. **Backend is running:**
   ```bash
   curl http://localhost:8200/
   # Should return: {"message":"NT8 RL Trading System API","version":"1.0.0"}
   ```

2. **Frontend is running:**
   - Open http://localhost:3200 in browser
   - Should see the web UI

3. **Training resumed:**
   - Check Training panel in web UI
   - Should show training status
   - Or check logs for "Resuming from checkpoint" message

---

## 🎯 What Gets Preserved

When you resume from checkpoint, you get:
- ✅ Neural network weights (Actor & Critic)
- ✅ Optimizer states
- ✅ Current timestep (continues from checkpoint)
- ✅ Episode number
- ✅ Episode rewards history
- ✅ Episode lengths history

**Result:** Training continues exactly where it left off - no data loss!

---

## ⚠️ Troubleshooting

### Port Already in Use

**Error:** `Address already in use` or `Port 8200 is in use`

**Solution:**
```bash
# Windows
netstat -ano | findstr :8200
taskkill /PID <PID> /F

# Or use stop script
python stop_ui.py
```

### No Checkpoints Found

**Error:** `No checkpoints found`

**Solution:**
- Check if training has run for at least 10,000 timesteps
- Checkpoints are saved every 10k timesteps
- If training just started, there may not be a checkpoint yet
- You can start fresh training (without checkpoint)

### Training Doesn't Resume

**Check:**
1. Is checkpoint path correct?
2. Does checkpoint file exist?
3. Check training logs for errors
4. Verify config file matches the one used to create checkpoint

---

## 📝 Summary

**To restart and resume:**

1. **Stop:** Ctrl+C or `python stop_ui.py`
2. **Restart:** `python start_ui.py`
3. **Resume:** `python resume_training.py --device cuda`

**That's it!** Training will continue from your latest checkpoint.

---

**Last Updated:** Current  
**Status:** ✅ Ready to Use

