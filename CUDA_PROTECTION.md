# CUDA PyTorch Protection System

This document describes the safeguards in place to prevent CPU-only PyTorch from being installed.

## Problem

When running `uv sync` or `pip install`, CPU-only PyTorch (e.g., `torch==2.9.0+cpu`) can sometimes be installed instead of CUDA-enabled PyTorch (e.g., `torch==2.5.1+cu121`). This causes:
- Training to run on CPU (much slower)
- Frontend to show "CUDA not available"
- Poor performance

## Solution: Multi-Layer Protection

### 1. Configuration Safeguards (`pyproject.toml`)

- **CUDA Index**: `extra-index-url = ["https://download.pytorch.org/whl/cu121"]`
- **Prefer CUDA**: `index-strategy = "unsafe-best-match"` (prefers CUDA builds)
- **Version Constraints**: Pinned to CUDA-compatible versions (`torch>=2.5.1,<2.6.0`)

### 2. Pre-Sync Verification (`prevent_cpu_pytorch.py`)

**Before running `uv sync`, verify configuration:**

```bash
python prevent_cpu_pytorch.py
```

This checks:
- `pyproject.toml` has CUDA index configured
- `index-strategy` is set correctly
- Current PyTorch (if installed) is CUDA-enabled

### 3. Safe Sync Wrapper (`uv_sync_safe.ps1` / `uv_sync_safe.sh`)

**Use this instead of `uv sync`:**

```powershell
# Windows
.\uv_sync_safe.ps1

# Linux/Mac
./uv_sync_safe.sh
```

This script:
1. Verifies CUDA configuration
2. Runs `uv sync`
3. Checks if CUDA PyTorch is still installed
4. Auto-fixes if CPU-only version was installed

### 4. Auto-Fix Script (`ensure_cuda_pytorch.py`)

**If CPU-only PyTorch is detected, fix it:**

```bash
# Interactive (prompts for confirmation)
python ensure_cuda_pytorch.py

# Auto-fix (no prompts)
python ensure_cuda_pytorch.py --auto
```

This script:
- Detects CPU-only PyTorch
- Uninstalls CPU-only version
- Installs CUDA-enabled version from PyTorch index

### 5. Startup Checks (`start_ui.py`)

When starting the UI, `start_ui.py`:
- Checks CUDA availability using venv Python
- Warns if CPU-only PyTorch is detected
- Provides instructions to fix

### 6. Backend Verification (`src/api_server.py`)

The backend API:
- Reports CUDA status at `/api/system/cuda-status`
- Detects CPU-only builds
- Provides installation instructions

## Usage Workflow

### Recommended Workflow

1. **Before first setup:**
   ```bash
   python prevent_cpu_pytorch.py
   ```

2. **Install dependencies (use safe wrapper):**
   ```powershell
   .\uv_sync_safe.ps1
   ```

3. **If CPU-only PyTorch detected:**
   ```bash
   python ensure_cuda_pytorch.py --auto
   ```

4. **Verify:**
   ```bash
   python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Version:', torch.__version__)"
   ```

### Alternative: Manual Fix

If you've already run `uv sync` and got CPU-only PyTorch:

```bash
# Uninstall CPU-only
uv pip uninstall torch torchvision torchaudio

# Install CUDA version
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Verification

### Check PyTorch Version

```bash
# Using venv Python
.venv\Scripts\python.exe -c "import torch; print(torch.__version__)"

# Should show: 2.5.1+cu121 (or similar with +cu)
# NOT: 2.9.0+cpu
```

### Check CUDA Availability

```bash
.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### Check Configuration

```bash
python prevent_cpu_pytorch.py
```

## Troubleshooting

### Issue: `uv sync` installs CPU-only PyTorch

**Solution:**
1. Run `python ensure_cuda_pytorch.py --auto`
2. Use `uv_sync_safe.ps1` instead of `uv sync` in the future

### Issue: Frontend shows "CUDA not available"

**Solution:**
1. Check backend is using venv Python: `start_ui.py` should show venv Python path
2. Verify CUDA PyTorch: `python ensure_cuda_pytorch.py --auto`
3. Restart backend: `python stop_ui.py && python start_ui.py`

### Issue: Configuration check fails

**Solution:**
1. Verify `pyproject.toml` has:
   - `extra-index-url = ["https://download.pytorch.org/whl/cu121"]`
   - `index-strategy = "unsafe-best-match"`
2. Re-run: `python prevent_cpu_pytorch.py`

## Files

- `prevent_cpu_pytorch.py` - Pre-sync verification
- `ensure_cuda_pytorch.py` - Auto-fix CPU-only PyTorch
- `uv_sync_safe.ps1` - Safe wrapper for Windows
- `uv_sync_safe.sh` - Safe wrapper for Linux/Mac
- `setup_cuda_pytorch.ps1` - PowerShell setup script
- `pyproject.toml` - Project configuration with CUDA settings
- `start_ui.py` - Startup script with CUDA checks

## Best Practices

1. **Always use `uv_sync_safe.ps1`** instead of `uv sync`
2. **Run `prevent_cpu_pytorch.py`** before major dependency updates
3. **Check CUDA status** after `uv sync` or `pip install`
4. **Use `--auto` flag** for automated fixes in scripts
5. **Verify venv Python** is used (not system Python)

