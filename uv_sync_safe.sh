#!/bin/bash
# Safe uv sync - Prevents CPU-only PyTorch installation
# Use this instead of `uv sync` to ensure CUDA PyTorch is maintained

echo "======================================================================"
echo "Safe uv sync - CUDA PyTorch Protection"
echo "======================================================================"
echo ""

# Step 1: Check configuration
echo "Step 1: Verifying CUDA configuration..."
python prevent_cpu_pytorch.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Configuration check failed"
    exit 1
fi
echo "[OK] Configuration verified"
echo ""

# Step 2: Check current PyTorch (if installed)
echo "Step 2: Checking current PyTorch installation..."
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
else
    PYTHON="python"
fi

$PYTHON -c "import torch; print('Version:', torch.__version__); print('CUDA Build:', '+' in torch.__version__ and 'cu' in torch.__version__)" 2>&1
if [ $? -eq 0 ]; then
    CUDA_BUILD=$($PYTHON -c "import torch; print('+' in torch.__version__ and 'cu' in torch.__version__)" 2>&1)
    if [ "$CUDA_BUILD" = "True" ]; then
        echo "[OK] CUDA PyTorch already installed"
    else
        echo "[WARN] CPU-only PyTorch detected - will fix after sync"
    fi
else
    echo "[INFO] PyTorch not yet installed"
fi
echo ""

# Step 3: Run uv sync
echo "Step 3: Running uv sync..."
uv sync
if [ $? -ne 0 ]; then
    echo "[ERROR] uv sync failed"
    exit 1
fi
echo "[OK] uv sync completed"
echo ""

# Step 4: Verify CUDA PyTorch after sync
echo "Step 4: Verifying CUDA PyTorch after sync..."
VERIFY_CHECK=$($PYTHON -c "import torch; print('Version:', torch.__version__); print('CUDA Build:', '+' in torch.__version__ and 'cu' in torch.__version__); print('CUDA Available:', torch.cuda.is_available())" 2>&1)
echo "$VERIFY_CHECK"

CUDA_BUILD=$($PYTHON -c "import torch; print('+' in torch.__version__ and 'cu' in torch.__version__)" 2>&1)
if [ "$CUDA_BUILD" = "True" ]; then
    echo "[OK] CUDA PyTorch verified after sync"
else
    echo "[WARN] CPU-only PyTorch detected after sync - fixing..."
    echo ""
    echo "Installing CUDA PyTorch..."
    python ensure_cuda_pytorch.py --auto
    if [ $? -eq 0 ]; then
        echo "[OK] CUDA PyTorch fixed"
    else
        echo "[ERROR] Failed to fix CUDA PyTorch"
        exit 1
    fi
fi

echo ""
echo "======================================================================"
echo "Safe sync completed - CUDA PyTorch verified"
echo "======================================================================"

