# Simple PowerShell script to run seed detection on Windows
Write-Host "🌱 Starting Seed Detection System..." -ForegroundColor Green

# Check if Docker is installed
try {
    docker --version | Out-Null
    Write-Host "✅ Docker is installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not installed. Please install Docker Desktop first." -ForegroundColor Red
    Write-Host "   Download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if docker-compose is available
try {
    docker-compose --version | Out-Null
    Write-Host "✅ docker-compose is available" -ForegroundColor Green
} catch {
    Write-Host "❌ docker-compose is not available. Please install it." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "🔍 Checking system..." -ForegroundColor Cyan

# Create results directory
if (!(Test-Path "results")) {
    New-Item -ItemType Directory -Name "results" | Out-Null
    Write-Host "📁 Created results directory" -ForegroundColor Green
}

# Build and run the container
Write-Host "🏗️ Building Docker image..." -ForegroundColor Cyan
docker-compose build

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Build successful" -ForegroundColor Green
    Write-Host "🚀 Starting seed detection system..." -ForegroundColor Cyan
    docker-compose up
} else {
    Write-Host "❌ Build failed" -ForegroundColor Red
}

Write-Host "✅ Done! Check the 'results' folder for saved images." -ForegroundColor Green
Read-Host "Press Enter to exit"
