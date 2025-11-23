# PyTorch CUDA Setup

## Overview

The project is configured to automatically install CUDA-enabled PyTorch builds instead of CPU-only versions.

## Configuration

### pyproject.toml

The `pyproject.toml` file is configured with:

```toml
[tool.uv]
# PyTorch CUDA index - CUDA 12.1 builds
extra-index-url = ["https://download.pytorch.org/whl/cu121"]
# Use unsafe-best-match to prefer CUDA builds from PyTorch index over CPU builds from PyPI
index-strategy = "unsafe-best-match"
```

This configuration:
- ✅ Adds PyTorch's CUDA 12.1 index as an extra source
- ✅ Uses `unsafe-best-match` strategy to prefer CUDA builds over CPU builds
- ✅ Automatically installs CUDA versions when running `uv sync`

### Dependencies

```toml
dependencies = [
    "torch>=2.5.0",      # CUDA builds available from PyTorch index (cu121)
    "torchvision>=0.20.0",  # CUDA builds available from PyTorch index
    "torchaudio>=2.5.0",    # CUDA builds available from PyTorch index
]
```

## Installation

### Using uv (Recommended)

```bash
# Sync dependencies - will automatically install CUDA builds
uv sync
```

The `uv sync` command will:
1. Check the extra-index-url for PyTorch packages
2. Prefer CUDA builds (e.g., `torch==2.5.1+cu121`) over CPU builds (e.g., `torch==2.9.0+cpu`)
3. Install the CUDA-enabled versions

### Using pip

If not using uv, install CUDA builds manually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Verifying CUDA Installation

After installation, verify CUDA is available:

```bash
# Using uv
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Or using Python directly
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Expected output:
```
CUDA available: True
PyTorch: 2.5.1+cu121
GPU: NVIDIA GeForce RTX 4060 Ti
```

## CUDA Versions

The project is configured for **CUDA 12.1**. If you need a different CUDA version:

### CUDA 11.8
```toml
extra-index-url = ["https://download.pytorch.org/whl/cu118"]
```

### CUDA 12.4
```toml
extra-index-url = ["https://download.pytorch.org/whl/cu124"]
```

Then run `uv sync` to update.

## Troubleshooting

### CPU Version Still Installed

If CPU-only PyTorch is installed instead of CUDA:

1. **Force reinstall with CUDA index**:
   ```bash
   uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Check index strategy**:
   - Ensure `index-strategy = "unsafe-best-match"` is set in `pyproject.toml`
   - This allows uv to prefer CUDA builds from the extra index

3. **Verify extra-index-url**:
   - Check that `extra-index-url = ["https://download.pytorch.org/whl/cu121"]` is in `[tool.uv]` section

### CUDA Not Available After Installation

1. **Check NVIDIA drivers**:
   ```bash
   nvidia-smi
   ```

2. **Verify PyTorch version**:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```
   Should show `+cu121` or similar (not `+cpu`)

3. **Check CUDA runtime**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Frontend Detection

The frontend automatically detects CUDA availability via the `/api/system/cuda-status` endpoint. If CUDA is available:
- ✅ CUDA option will be enabled
- ✅ GPU name will be displayed
- ✅ CUDA device will be auto-selected

The backend API server must be running with the CUDA-enabled PyTorch environment for detection to work.

