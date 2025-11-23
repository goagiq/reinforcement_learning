# Deploy Kong Gateway to Docker
# This script sets up and deploys Kong Gateway for the NT8 RL Trading System

Write-Host "="*70 -ForegroundColor Cyan
Write-Host "Kong Gateway Deployment Script" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Docker
Write-Host "Step 1: Checking Docker..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version 2>&1
    Write-Host "  [OK] Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Docker not found. Please install Docker Desktop." -ForegroundColor Red
    Write-Host "     Download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
}

# Step 2: Check if Docker Desktop is running
Write-Host ""
Write-Host "Step 2: Checking if Docker Desktop is running..." -ForegroundColor Yellow
try {
    $dockerInfo = docker info 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] Docker Desktop is running" -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] Docker Desktop is not running" -ForegroundColor Red
        Write-Host "     Please start Docker Desktop and try again." -ForegroundColor Yellow
        Write-Host "     Waiting 10 seconds for you to start Docker Desktop..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
        
        # Check again
        $dockerInfo = docker info 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [ERROR] Docker Desktop is still not running" -ForegroundColor Red
            exit 1
        }
        Write-Host "  [OK] Docker Desktop is now running" -ForegroundColor Green
    }
} catch {
    Write-Host "  [ERROR] Cannot connect to Docker. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Step 3: Check if ports are available
Write-Host ""
Write-Host "Step 3: Checking if required ports are available..." -ForegroundColor Yellow
$ports = @(8300, 8301, 5434)
$portsInUse = @()

foreach ($port in $ports) {
    $connection = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($connection) {
        $pid = $connection.OwningProcess
        $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
        $procName = if ($proc) { $proc.ProcessName } else { "Unknown" }
        Write-Host "  [WARN] Port $port is in use by PID $pid ($procName)" -ForegroundColor Yellow
        $portsInUse += $port
    } else {
        Write-Host "  [OK] Port $port is available" -ForegroundColor Green
    }
}

if ($portsInUse.Count -gt 0) {
    Write-Host ""
    Write-Host "  [WARN] Warning: Some ports are in use. Kong may not start properly." -ForegroundColor Yellow
    Write-Host "     Ports in use: $($portsInUse -join ', ')" -ForegroundColor Yellow
    $continue = Read-Host "     Continue anyway? (y/N)"
    if ($continue -ne 'y' -and $continue -ne 'Y') {
        Write-Host "  Deployment cancelled." -ForegroundColor Yellow
        exit 1
    }
}

# Step 4: Stop existing Kong containers (if any)
Write-Host ""
Write-Host "Step 4: Stopping existing Kong containers..." -ForegroundColor Yellow
$existingContainers = docker ps -a --filter "name=kong" --format "{{.Names}}" 2>&1
if ($existingContainers) {
    Write-Host "  Found existing containers: $existingContainers" -ForegroundColor Gray
    Write-Host "  Stopping and removing..." -ForegroundColor Yellow
    docker-compose -f docker-compose.yml down 2>&1 | Out-Null
    Start-Sleep -Seconds 2
    Write-Host "  [OK] Existing containers stopped" -ForegroundColor Green
} else {
    Write-Host "  [INFO]  No existing Kong containers found" -ForegroundColor Gray
}

# Step 5: Pull latest images
Write-Host ""
Write-Host "Step 5: Pulling latest Docker images..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes on first run..." -ForegroundColor Gray

$images = @("postgres:15-alpine", "kong:3.7")
foreach ($image in $images) {
    Write-Host "  Pulling $image..." -ForegroundColor Gray
    docker pull $image 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] $image pulled successfully" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] Warning: Failed to pull $image (may use cached version)" -ForegroundColor Yellow
    }
}

# Step 6: Start Kong
Write-Host ""
Write-Host "Step 6: Starting Kong Gateway..." -ForegroundColor Yellow
Write-Host "  This will start PostgreSQL and Kong containers..." -ForegroundColor Gray

$startResult = docker-compose -f docker-compose.yml up -d 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Kong containers started" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] Failed to start Kong containers" -ForegroundColor Red
    Write-Host "     Error: $startResult" -ForegroundColor Red
    exit 1
}

# Step 7: Wait for Kong to be ready
Write-Host ""
Write-Host "Step 7: Waiting for Kong to be ready..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0
$kongReady = $false

while ($attempt -lt $maxAttempts -and -not $kongReady) {
    Start-Sleep -Seconds 2
    $attempt++
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8301/" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            $kongReady = $true
            Write-Host "  [OK] Kong is ready!" -ForegroundColor Green
        }
    } catch {
        if ($attempt % 5 -eq 0) {
            Write-Host "  Still waiting... ($attempt/$maxAttempts)" -ForegroundColor Gray
        }
    }
}

if (-not $kongReady) {
    Write-Host "  [WARN] Kong did not become ready within timeout" -ForegroundColor Yellow
    Write-Host "     Check logs with: docker-compose -f docker-compose.yml logs" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "="*70 -ForegroundColor Green
    Write-Host "[OK] Kong Gateway Deployed Successfully!" -ForegroundColor Green
    Write-Host "="*70 -ForegroundColor Green
    Write-Host ""
    Write-Host "Kong Gateway is now running:" -ForegroundColor Cyan
    Write-Host "  Proxy Endpoint:  http://localhost:8300" -ForegroundColor White
    Write-Host "  Admin API:      http://localhost:8301" -ForegroundColor White
    Write-Host "  PostgreSQL:     localhost:5434" -ForegroundColor White
    Write-Host ""
    Write-Host "Useful commands:" -ForegroundColor Yellow
    Write-Host "  View logs:      docker-compose -f docker-compose.yml logs -f" -ForegroundColor Gray
    Write-Host "  Stop Kong:      docker-compose -f docker-compose.yml down" -ForegroundColor Gray
    Write-Host "  Restart Kong:   docker-compose -f docker-compose.yml restart" -ForegroundColor Gray
    Write-Host "  Check status:   docker-compose -f docker-compose.yml ps" -ForegroundColor Gray
    Write-Host ""
    
    # Test Kong
    Write-Host "Testing Kong Admin API..." -ForegroundColor Yellow
    try {
        $testResponse = Invoke-WebRequest -Uri "http://localhost:8301/" -UseBasicParsing -TimeoutSec 5
        Write-Host "  [OK] Kong Admin API is responding" -ForegroundColor Green
    } catch {
        Write-Host "  [WARN] Kong Admin API test failed: $_" -ForegroundColor Yellow
    }
}

Write-Host ""


