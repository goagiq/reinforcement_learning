# Setup CUDA PyTorch - Prevents CPU-only installation
# This script ensures CUDA-enabled PyTorch is always installed

Write-Host "="*70 -ForegroundColor Cyan
Write-Host "CUDA PyTorch Setup Script" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan
Write-Host ""

# Check current PyTorch installation
Write-Host "Checking current PyTorch installation..." -ForegroundColor Yellow
$checkResult = & .venv\Scripts\python.exe -c "import torch; print('Version:', torch.__version__); print('CUDA Build:', '+' in torch.__version__ and 'cu' in torch.__version__); print('CUDA Available:', torch.cuda.is_available())" 2>&1

if ($checkResult -match "CUDA Build: True" -and $checkResult -match "CUDA Available: True") {
    Write-Host "[OK] CUDA PyTorch is already installed and working!" -ForegroundColor Green
    Write-Host $checkResult
    exit 0
}

Write-Host "[WARN] CUDA PyTorch not detected or not working" -ForegroundColor Yellow
Write-Host $checkResult
Write-Host ""

# Uninstall existing PyTorch
Write-Host "Uninstalling existing PyTorch packages..." -ForegroundColor Yellow
uv pip uninstall torch torchvision torchaudio -y 2>&1 | Out-Null

# Install CUDA PyTorch
Write-Host "Installing CUDA-enabled PyTorch (cu121)..." -ForegroundColor Yellow
$installResult = uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] CUDA PyTorch installed successfully" -ForegroundColor Green
    
    # Verify installation
    Write-Host ""
    Write-Host "Verifying installation..." -ForegroundColor Yellow
    $verifyResult = & .venv\Scripts\python.exe -c "import torch; print('Version:', torch.__version__); print('CUDA Build:', '+' in torch.__version__ and 'cu' in torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1
    
    Write-Host $verifyResult
    
    if ($verifyResult -match "CUDA Available: True") {
        Write-Host ""
        Write-Host "[OK] CUDA PyTorch is working correctly!" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "[WARN] CUDA PyTorch installed but CUDA runtime not available" -ForegroundColor Yellow
        Write-Host "  Check NVIDIA drivers and CUDA toolkit" -ForegroundColor Yellow
    }
} else {
    Write-Host "[ERROR] Failed to install CUDA PyTorch" -ForegroundColor Red
    Write-Host $installResult
    exit 1
}

Write-Host ""
