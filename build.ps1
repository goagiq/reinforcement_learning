# PowerShell script to build and deploy NT8 RL Trading System

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "NT8 RL Trading System - Build & Deploy" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Docker
Write-Host "Step 1: Checking Docker Desktop..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Docker Desktop is not running" -ForegroundColor Red
        Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "✓ Docker Desktop is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker Desktop is not accessible" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Step 2: Build image
Write-Host "Step 2: Building Docker image..." -ForegroundColor Yellow
Write-Host "This may take 10-15 minutes on first build..." -ForegroundColor Gray
docker-compose build

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Build failed" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Build completed successfully" -ForegroundColor Green
Write-Host ""

# Step 3: Verify image
Write-Host "Step 3: Verifying image..." -ForegroundColor Yellow
$imageExists = docker images | Select-String "nt8-rl-nt8-rl-app"
if ($imageExists) {
    Write-Host "✓ Image created successfully" -ForegroundColor Green
    docker images | Select-String "nt8-rl"
} else {
    Write-Host "⚠ Image not found, but build may have succeeded" -ForegroundColor Yellow
}

Write-Host ""

# Step 4: Ask to start
Write-Host "Step 4: Start container?" -ForegroundColor Yellow
$response = Read-Host "Start container now? (Y/n)"
if ($response -eq "" -or $response -eq "Y" -or $response -eq "y") {
    Write-Host "Starting container..." -ForegroundColor Yellow
    docker-compose up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Container started successfully" -ForegroundColor Green
        Write-Host ""
        Write-Host "Services are now available:" -ForegroundColor Cyan
        Write-Host "  Frontend: http://localhost:3200" -ForegroundColor White
        Write-Host "  API: http://localhost:8200" -ForegroundColor White
        Write-Host "  API Docs: http://localhost:8200/docs" -ForegroundColor White
        Write-Host "  NT8 Bridge: localhost:8888" -ForegroundColor White
        Write-Host ""
        Write-Host "View logs: docker-compose logs -f" -ForegroundColor Gray
        Write-Host "Stop: docker-compose down" -ForegroundColor Gray
    } else {
        Write-Host "✗ Failed to start container" -ForegroundColor Red
        Write-Host "Check logs: docker-compose logs" -ForegroundColor Yellow
    }
} else {
    Write-Host "Container not started. Start manually with:" -ForegroundColor Yellow
    Write-Host "  docker-compose up -d" -ForegroundColor Cyan
}

