# Safe uv sync - Prevents CPU-only PyTorch installation
# Use this instead of `uv sync` to ensure CUDA PyTorch is maintained

Write-Host "="*70 -ForegroundColor Cyan
Write-Host "Safe uv sync - CUDA PyTorch Protection" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan
Write-Host ""

# Step 1: Check configuration
Write-Host "Step 1: Verifying CUDA configuration..." -ForegroundColor Yellow
$configCheck = python prevent_cpu_pytorch.py 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Configuration check failed" -ForegroundColor Red
    Write-Host $configCheck
    exit 1
}
Write-Host "[OK] Configuration verified" -ForegroundColor Green
Write-Host ""

# Step 2: Check current PyTorch (if installed)
Write-Host "Step 2: Checking current PyTorch installation..." -ForegroundColor Yellow
try {
    $pytorchCheck = & .venv\Scripts\python.exe -c "import torch; print('Version:', torch.__version__); print('CUDA Build:', '+' in torch.__version__ and 'cu' in torch.__version__)" 2>&1
    if ($pytorchCheck -match "CUDA Build: True") {
        Write-Host "[OK] CUDA PyTorch already installed" -ForegroundColor Green
        Write-Host "  $pytorchCheck" -ForegroundColor Gray
    } elseif ($pytorchCheck -match "Version:") {
        Write-Host "[WARN] CPU-only PyTorch detected - will fix after sync" -ForegroundColor Yellow
        Write-Host "  $pytorchCheck" -ForegroundColor Gray
    }
} catch {
    Write-Host "[INFO] PyTorch not yet installed" -ForegroundColor Gray
}
Write-Host ""

# Step 3: Run uv sync
Write-Host "Step 3: Running uv sync..." -ForegroundColor Yellow
uv sync
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] uv sync failed" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] uv sync completed" -ForegroundColor Green
Write-Host ""

# Step 4: Verify CUDA PyTorch after sync
Write-Host "Step 4: Verifying CUDA PyTorch after sync..." -ForegroundColor Yellow
$verifyCheck = & .venv\Scripts\python.exe -c "import torch; print('Version:', torch.__version__); print('CUDA Build:', '+' in torch.__version__ and 'cu' in torch.__version__); print('CUDA Available:', torch.cuda.is_available())" 2>&1

if ($verifyCheck -match "CUDA Build: True") {
    Write-Host "[OK] CUDA PyTorch verified after sync" -ForegroundColor Green
    Write-Host $verifyCheck
} else {
    Write-Host "[WARN] CPU-only PyTorch detected after sync - fixing..." -ForegroundColor Yellow
    Write-Host $verifyCheck
    Write-Host ""
    Write-Host "Installing CUDA PyTorch..." -ForegroundColor Yellow
    python ensure_cuda_pytorch.py --auto
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] CUDA PyTorch fixed" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Failed to fix CUDA PyTorch" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "="*70 -ForegroundColor Green
Write-Host "Safe sync completed - CUDA PyTorch verified" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Green

