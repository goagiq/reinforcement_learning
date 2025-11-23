# Restart Backend with CUDA Support
# This script stops any running backend and starts it fresh with CUDA-enabled environment

Write-Host "="*60 -ForegroundColor Cyan
Write-Host "Restarting Backend with CUDA Support" -ForegroundColor Cyan
Write-Host "="*60 -ForegroundColor Cyan
Write-Host ""

# Find and stop any running backend processes
Write-Host "Step 1: Stopping existing backend processes..." -ForegroundColor Yellow
$backendProcesses = Get-Process | Where-Object {
    $_.CommandLine -like "*uvicorn*api_server*" -or 
    $_.CommandLine -like "*python*api_server*"
} -ErrorAction SilentlyContinue

if ($backendProcesses) {
    foreach ($proc in $backendProcesses) {
        Write-Host "  Stopping process: $($proc.ProcessName) (PID: $($proc.Id))" -ForegroundColor Yellow
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
} else {
    Write-Host "  No backend processes found" -ForegroundColor Gray
}

# Check if port 8200 is in use
$portInUse = Get-NetTCPConnection -LocalPort 8200 -State Listen -ErrorAction SilentlyContinue
if ($portInUse) {
    Write-Host "  Port 8200 is still in use, waiting..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
}

# Verify CUDA is available
Write-Host ""
Write-Host "Step 2: Verifying CUDA installation..." -ForegroundColor Yellow
$cudaCheck = uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('PyTorch:', torch.__version__)" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host $cudaCheck
    if ($cudaCheck -match "CUDA: True") {
        Write-Host "  ✅ CUDA is available!" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  CUDA not available - backend will use CPU" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ⚠️  Could not verify CUDA" -ForegroundColor Yellow
}

# Start backend using uv run
Write-Host ""
Write-Host "Step 3: Starting backend with uv run..." -ForegroundColor Yellow
Write-Host "  This ensures the correct environment with CUDA PyTorch is used" -ForegroundColor Gray
Write-Host ""

$backendScript = @"
import sys
import uvicorn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    uvicorn.run("src.api_server:app", host="0.0.0.0", port=8200, reload=False)
"@

$scriptPath = "start_backend_temp.py"
$backendScript | Out-File -FilePath $scriptPath -Encoding utf8

try {
    $process = Start-Process -FilePath "uv" -ArgumentList "run", "python", $scriptPath -NoNewWindow -PassThru
    Write-Host "  ✅ Backend started (PID: $($process.Id))" -ForegroundColor Green
    Write-Host "  Backend will be available at: http://localhost:8200" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Waiting for backend to initialize..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    
    # Test CUDA endpoint
    Write-Host ""
    Write-Host "Step 4: Testing CUDA detection endpoint..." -ForegroundColor Yellow
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8200/api/system/cuda-status" -UseBasicParsing -TimeoutSec 5
        $cudaData = $response.Content | ConvertFrom-Json
        Write-Host "  API Response:" -ForegroundColor Cyan
        Write-Host "    CUDA Available: $($cudaData.cuda_available)" -ForegroundColor $(if ($cudaData.cuda_available) { "Green" } else { "Yellow" })
        if ($cudaData.gpu_name) {
            Write-Host "    GPU: $($cudaData.gpu_name)" -ForegroundColor Green
            Write-Host "    CUDA Version: $($cudaData.cuda_version)" -ForegroundColor Green
        }
    } catch {
        Write-Host "  ⚠️  Backend not responding yet. It may still be starting..." -ForegroundColor Yellow
        Write-Host "     Try accessing http://localhost:8200/api/system/cuda-status in a few seconds" -ForegroundColor Gray
    }
    
    Write-Host ""
    Write-Host "="*60 -ForegroundColor Green
    Write-Host "Backend Restart Complete!" -ForegroundColor Green
    Write-Host "="*60 -ForegroundColor Green
    Write-Host ""
    Write-Host "The frontend should now detect CUDA when you refresh the page." -ForegroundColor Cyan
    Write-Host "Backend logs will appear in this window." -ForegroundColor Gray
    
} catch {
    Write-Host "  ❌ Failed to start backend: $_" -ForegroundColor Red
    Write-Host "  Try running manually: uv run python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8200" -ForegroundColor Yellow
} finally {
    # Clean up temp script
    if (Test-Path $scriptPath) {
        Remove-Item $scriptPath -ErrorAction SilentlyContinue
    }
}

