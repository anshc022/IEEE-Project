@echo off
REM Simple script to run seed detection on Windows for testing

echo 🌱 Starting Seed Detection System...

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    echo    Download from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Check for camera (basic check)
echo 🔍 Checking system...

REM Create results directory
if not exist results mkdir results

REM Build and run the container
echo 🏗️ Building Docker image...
docker-compose build

echo 🚀 Starting seed detection system...
docker-compose up

echo ✅ Done! Check the 'results' folder for saved images.
pause
