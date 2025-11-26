# Start Script Recommendations for Priority 1 Messages

**Issue**: Priority 1 initialization messages may not appear due to output buffering  
**Solution**: Add `PYTHONUNBUFFERED=1` environment variable

---

## ✅ Recommended Addition

### For Shell Scripts (start.sh, start_ui.sh, etc.)

Add this at the top of your script, before starting Python:

```bash
#!/bin/bash

# Enable unbuffered Python output (ensures Priority 1 messages appear immediately)
export PYTHONUNBUFFERED=1

# Your existing commands...
python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8200
```

### For start_ui.py (Python Script)

If you're using `start_ui.py`, you can modify it to set the environment variable:

**Option 1: Set in subprocess.Popen**
```python
backend_process = subprocess.Popen(
    backend_command,
    stdout=None,
    stderr=subprocess.STDOUT,
    env={**os.environ, "PYTHONUNBUFFERED": "1"}  # Add this
)
```

**Option 2: Set at script start**
```python
import os
os.environ["PYTHONUNBUFFERED"] = "1"
```

---

## 🔍 What PYTHONUNBUFFERED Does

- **Without it**: Python buffers stdout/stderr, messages may appear delayed or not at all
- **With it**: Python outputs immediately, ensuring Priority 1 messages appear right away

---

## 📋 Example start.sh Script

If you create a `start.sh` script, here's a complete example:

```bash
#!/bin/bash

# Enable unbuffered Python output
export PYTHONUNBUFFERED=1

# Set Python encoding (optional, helps with special characters)
export PYTHONIOENCODING=utf-8

# Start backend API server
echo "Starting backend API server..."
python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8200

# Or if using uv:
# uv run python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8200
```

---

## 🎯 For Windows (start.bat or PowerShell)

If you're on Windows and using a batch file:

```batch
@echo off
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8200
```

Or PowerShell:

```powershell
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8200
```

---

## ✅ Current Status

**Code changes already made:**
- ✅ Print statements use `[PRIORITY 1]` prefix (no emoji encoding issues)
- ✅ `sys.stdout.flush()` calls added to force output
- ✅ Messages should appear in console

**What you need:**
- ⚠️ Add `PYTHONUNBUFFERED=1` to your start script (if not already present)

---

## 🔍 How to Check

After adding `PYTHONUNBUFFERED=1`, when you start training you should see:

```
Creating trading environment...
  Max episode steps: 10000 (episodes will terminate at this limit)
  [PRIORITY 1] Slippage model: Enabled
  [PRIORITY 1] Market impact model: Enabled
  [PRIORITY 1] Execution quality tracker: Available
Creating PPO agent...
```

---

## 📝 Note

If you're using `start_ui.py`, the backend is started via `subprocess.Popen`. The environment variable should be set in the parent process (start_ui.py) or passed to the subprocess.

**Current start_ui.py behavior:**
- Output is shown in console (`stdout=None`)
- But `PYTHONUNBUFFERED` is not explicitly set
- Adding it will ensure immediate output

---

## 🚀 Quick Fix

**If you have a start.sh script**, just add this line at the top:

```bash
export PYTHONUNBUFFERED=1
```

**If you're using start_ui.py**, you can either:
1. Set it before running: `PYTHONUNBUFFERED=1 python start_ui.py`
2. Or modify start_ui.py to set it in the subprocess environment

