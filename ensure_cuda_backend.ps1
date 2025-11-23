# Ensure Backend is Running with CUDA Support
# This script verifies CUDA PyTorch is installed and restarts the backend if needed

Write-Host "="*70 -ForegroundColor Cyan
Write-Host "Ensuring Backend has CUDA Support" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if CUDA PyTorch is installed
Write-Host "Step 1: Checking CUDA PyTorch installation..." -ForegroundColor Yellow
$cudaCheck = & .venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())" 2>&1

if ($cudaCheck -match "CUDA: True" -and $cudaCheck -match "cu\d+") {
    Write-Host "  ✅ CUDA PyTorch is installed" -ForegroundColor Green
    Write-Host "     $cudaCheck" -ForegroundColor Gray
} else {
    Write-Host "  ⚠️  CPU-only PyTorch detected - Installing CUDA version..." -ForegroundColor Yellow
    Write-Host "     $cudaCheck" -ForegroundColor Gray
    
    # Install CUDA PyTorch
    Write-Host "  Installing CUDA-enabled PyTorch..." -ForegroundColor Yellow
    uv pip uninstall torch torchvision torchaudio 2>&1 | Out-Null
    $installResult = uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ CUDA PyTorch installed successfully" -ForegroundColor Green
    } else {
        Write-Host "  ❌ Failed to install CUDA PyTorch" -ForegroundColor Red
        Write-Host "     $installResult" -ForegroundColor Red
        exit 1
    }
}

# Step 2: Stop existing backend
Write-Host ""
Write-Host "Step 2: Stopping existing backend..." -ForegroundColor Yellow
$backendPort = Get-NetTCPConnection -LocalPort 8200 -State Listen -ErrorAction SilentlyContinue
if ($backendPort) {
    $pid = $backendPort.OwningProcess
    Write-Host "  Stopping backend (PID: $pid)..." -ForegroundColor Yellow
    Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Write-Host "  ✅ Backend stopped" -ForegroundColor Green
} else {
    Write-Host "  ℹ️  No backend running on port 8200" -ForegroundColor Gray
}

# Step 3: Start backend with CUDA support
Write-Host ""
Write-Host "Step 3: Starting backend with CUDA support..." -ForegroundColor Yellow
$pythonExe = (Get-ChildItem .venv\Scripts\python.exe).FullName
Write-Host "  Using: $pythonExe" -ForegroundColor Gray

try {
    $backendProcess = Start-Process -FilePath $pythonExe -ArgumentList "-m", "uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8200" -NoNewWindow -PassThru
    Write-Host "  ✅ Backend started (PID: $($backendProcess.Id))" -ForegroundColor Green
    Write-Host "  Waiting for backend to initialize..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    
    # Verify CUDA detection
    $maxAttempts = 5
    $attempt = 0
    $cudaDetected = $false
    
    while ($attempt -lt $maxAttempts -and -not $cudaDetected) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8200/api/system/cuda-status" -UseBasicParsing -TimeoutSec 5
            $cudaData = $response.Content | ConvertFrom-Json
            
            Write-Host ""
            Write-Host "  ✅ Backend CUDA Detection:" -ForegroundColor Green
            Write-Host "     CUDA Available: $($cudaData.cuda_available)" -ForegroundColor $(if ($cudaData.cuda_available) { "Green" } else { "Yellow" })
            Write-Host "     PyTorch Version: $($cudaData.pytorch_version)" -ForegroundColor Gray
            if ($cudaData.gpu_name) {
                Write-Host "     GPU: $($cudaData.gpu_name)" -ForegroundColor Green
                Write-Host "     CUDA Version: $($cudaData.cuda_version)" -ForegroundColor Gray
            }
            $cudaDetected = $true
        } catch {
            $attempt++
            if ($attempt -lt $maxAttempts) {
                Write-Host "  Waiting for backend... ($attempt/$maxAttempts)" -ForegroundColor Yellow
                Start-Sleep -Seconds 2
            } else {
                Write-Host "  ⚠️  Backend started but CUDA check timed out" -ForegroundColor Yellow
                Write-Host "     Backend is running, but CUDA status could not be verified" -ForegroundColor Yellow
            }
        }
    }
} catch {
    Write-Host "  ❌ Failed to start backend: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "="*70 -ForegroundColor Green
Write-Host "✅ Backend is running with CUDA support!" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Green
Write-Host ""
Write-Host "Backend API: http://localhost:8200" -ForegroundColor Cyan
Write-Host "CUDA Status: http://localhost:8200/api/system/cuda-status" -ForegroundColor Cyan
Write-Host ""
Write-Host "The frontend should now detect CUDA." -ForegroundColor Yellow
Write-Host "Refresh your browser (Ctrl+F5) if needed." -ForegroundColor Yellow
Write-Host ""

