# PowerShell script to check Docker status
Write-Host "Checking Docker Desktop status..." -ForegroundColor Yellow

# Check if Docker is running
try {
    $dockerVersion = docker --version 2>&1
    Write-Host "✓ Docker CLI found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker CLI not found. Please install Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check if Docker daemon is accessible
try {
    $dockerPs = docker ps 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Docker Desktop is running" -ForegroundColor Green
        Write-Host ""
        Write-Host "Current containers:" -ForegroundColor Cyan
        docker ps
    } else {
        Write-Host "✗ Docker Desktop is not running" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please:" -ForegroundColor Yellow
        Write-Host "1. Open Docker Desktop from Start Menu" -ForegroundColor Yellow
        Write-Host "2. Wait for it to fully start (whale icon in system tray)" -ForegroundColor Yellow
        Write-Host "3. Run this script again" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "✗ Cannot connect to Docker daemon" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Docker is ready! You can now run:" -ForegroundColor Green
Write-Host "  docker-compose build" -ForegroundColor Cyan
Write-Host "  docker-compose up -d" -ForegroundColor Cyan

