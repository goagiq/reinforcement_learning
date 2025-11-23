# Restart UI with CUDA Support
# Stops existing servers and restarts with proper CUDA detection

Write-Host "="*70 -ForegroundColor Cyan
Write-Host "Restarting NT8-RL UI with CUDA Support" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan
Write-Host ""

# Step 1: Stop existing processes
Write-Host "Step 1: Stopping existing servers..." -ForegroundColor Yellow

# Stop backend on port 8200
$backendPort = Get-NetTCPConnection -LocalPort 8200 -State Listen -ErrorAction SilentlyContinue
if ($backendPort) {
    $pid = $backendPort.OwningProcess
    Write-Host "  Stopping backend (PID: $pid)..." -ForegroundColor Yellow
    Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

# Stop frontend on port 3200
$frontendPort = Get-NetTCPConnection -LocalPort 3200 -State Listen -ErrorAction SilentlyContinue
if ($frontendPort) {
    $pid = $frontendPort.OwningProcess
    Write-Host "  Stopping frontend (PID: $pid)..." -ForegroundColor Yellow
    Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

# Step 2: Verify CUDA
Write-Host ""
Write-Host "Step 2: Verifying CUDA installation..." -ForegroundColor Yellow
$cudaCheck = uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1
Write-Host $cudaCheck
if ($cudaCheck -match "CUDA: True") {
    Write-Host "  ✅ CUDA is available!" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  CUDA not available - will use CPU" -ForegroundColor Yellow
}

# Step 3: Start backend
Write-Host ""
Write-Host "Step 3: Starting backend with uv run..." -ForegroundColor Yellow
Write-Host "  This ensures CUDA-enabled PyTorch is used" -ForegroundColor Gray

$backendScript = @"
import sys
from pathlib import Path
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    print("="*60)
    print("Starting Backend API Server")
    print("="*60)
    print("Using uv environment with CUDA PyTorch")
    print("="*60)
    uvicorn.run("src.api_server:app", host="0.0.0.0", port=8200, reload=False)
"@

$scriptPath = "start_backend_temp.py"
$backendScript | Out-File -FilePath $scriptPath -Encoding utf8

try {
    $backendProcess = Start-Process -FilePath "uv" -ArgumentList "run", "python", $scriptPath -NoNewWindow -PassThru
    Write-Host "  ✅ Backend started (PID: $($backendProcess.Id))" -ForegroundColor Green
    Write-Host "  Waiting for backend to initialize..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    
    # Test CUDA endpoint
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8200/api/system/cuda-status" -UseBasicParsing -TimeoutSec 5
        $cudaData = $response.Content | ConvertFrom-Json
        Write-Host ""
        Write-Host "  ✅ Backend CUDA Detection:" -ForegroundColor Green
        Write-Host "     CUDA Available: $($cudaData.cuda_available)" -ForegroundColor $(if ($cudaData.cuda_available) { "Green" } else { "Yellow" })
        if ($cudaData.gpu_name) {
            Write-Host "     GPU: $($cudaData.gpu_name)" -ForegroundColor Green
        }
    } catch {
        Write-Host "  ⚠️  Backend not responding yet, but process is running" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ❌ Failed to start backend: $_" -ForegroundColor Red
    exit 1
}

# Step 4: Start frontend
Write-Host ""
Write-Host "Step 4: Starting frontend..." -ForegroundColor Yellow

$frontendDir = "frontend"
if (Test-Path "$frontendDir\dist\index.html") {
    Write-Host "  Serving built frontend from dist/..." -ForegroundColor Gray
    # Serve built frontend
    $frontendProcess = Start-Process -FilePath "uv" -ArgumentList "run", "python", "-m", "http.server", "3200" -WorkingDirectory "$frontendDir\dist" -NoNewWindow -PassThru
    Write-Host "  ✅ Frontend server started (PID: $($frontendProcess.Id))" -ForegroundColor Green
} else {
    Write-Host "  Starting frontend dev server..." -ForegroundColor Gray
    # Check if npm is available
    $npmAvailable = Get-Command npm -ErrorAction SilentlyContinue
    if ($npmAvailable) {
        $frontendProcess = Start-Process -FilePath "npm" -ArgumentList "run", "dev" -WorkingDirectory $frontendDir -NoNewWindow -PassThru
        Write-Host "  ✅ Frontend dev server started (PID: $($frontendProcess.Id))" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  npm not found - cannot start frontend dev server" -ForegroundColor Yellow
        Write-Host "     Install Node.js or serve the built frontend manually" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "="*70 -ForegroundColor Green
Write-Host "✅ UI Restart Complete!" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Green
Write-Host ""
Write-Host "Backend API:  http://localhost:8200" -ForegroundColor Cyan
Write-Host "Frontend UI:  http://localhost:3200" -ForegroundColor Cyan
Write-Host ""
Write-Host "The frontend should now:" -ForegroundColor Yellow
Write-Host "  ✅ Show train_config_adaptive.yaml in the config dropdown" -ForegroundColor Green
Write-Host "  ✅ Detect CUDA and auto-select GPU device" -ForegroundColor Green
Write-Host ""
Write-Host "If CUDA is not detected:" -ForegroundColor Yellow
Write-Host "  1. Check browser console (F12) for CUDA status messages" -ForegroundColor Gray
Write-Host "  2. Verify backend is running: http://localhost:8200/api/system/cuda-status" -ForegroundColor Gray
Write-Host "  3. Restart backend: .\restart_backend.ps1" -ForegroundColor Gray
Write-Host ""

# Clean up temp script
if (Test-Path $scriptPath) {
    Remove-Item $scriptPath -ErrorAction SilentlyContinue
}

