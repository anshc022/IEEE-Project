# PowerShell script to prepare files for Jetson Nano deployment
# Run this on Windows before transferring files to Jetson Nano

Write-Host "=== Jetson Nano Docker Setup - Windows Preparation ===" -ForegroundColor Green
Write-Host "This script prepares the project files for deployment on Jetson Nano" -ForegroundColor Yellow

# Function to write colored output
function Write-Status {
    param($Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warning {
    param($Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param($Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if we're in the correct directory
if (-not (Test-Path "app.py")) {
    Write-Error "app.py not found. Please run this script from the project directory."
    exit 1
}

Write-Status "Found project files in current directory"

# Create deployment package
Write-Status "Creating deployment package..."

# Create a deployment directory
$deployDir = "jetson-deployment"
if (Test-Path $deployDir) {
    Remove-Item $deployDir -Recurse -Force
}
New-Item -ItemType Directory -Path $deployDir | Out-Null

# Copy essential files
$filesToCopy = @(
    "app.py",
    "test_camera.py", 
    "fix_camera.py",
    "requirements.txt",
    "corn11.pt",
    "README.md",
    "DOCKER_README.md",
    "Dockerfile.jetson-fixed",
    "Dockerfile.ultra-minimal", 
    "docker-compose-jetson-fixed.yml",
    "docker-compose-ultra-minimal.yml",
    "docker-compose-legacy.yml",
    "docker-setup-jetson-fixed.sh",
    "test-docker-setup.sh"
)

foreach ($file in $filesToCopy) {
    if (Test-Path $file) {
        Copy-Item $file -Destination $deployDir
        Write-Status "Copied $file"
    } else {
        Write-Warning "$file not found, skipping"
    }
}

# Create directories
New-Item -ItemType Directory -Path "$deployDir\data" -Force | Out-Null
New-Item -ItemType Directory -Path "$deployDir\logs" -Force | Out-Null
New-Item -ItemType Directory -Path "$deployDir\models" -Force | Out-Null

Write-Status "Created data, logs, and models directories"

# Create a README for deployment
$deployReadme = @"
# Jetson Nano Deployment Package

This package contains all necessary files for deploying the YOLOv11 seed detection application on Jetson Nano.

## Quick Deployment Steps:

1. Transfer this entire folder to your Jetson Nano
2. SSH into your Jetson Nano and navigate to this directory
3. Make scripts executable:
   ```bash
   chmod +x *.sh
   ```

4. Run the enhanced setup script:
   ```bash
   ./docker-setup-jetson-fixed.sh
   ```

5. Build and run the application:
   ```bash
   # Recommended approach (handles repository issues)
   docker-compose -f docker-compose-jetson-fixed.yml up --build

   # Alternative if repository issues persist
   docker-compose -f docker-compose-ultra-minimal.yml up --build
   ```

6. Test camera functionality:
   ```bash
   docker-compose -f docker-compose-jetson-fixed.yml run --rm yolo-seed-detection python3 test_camera.py
   ```

## Files Included:

- app.py - Main application
- test_camera.py - Camera testing utility  
- fix_camera.py - Green screen fix utility
- Dockerfile.jetson-fixed - Repository-issue fixed Docker image
- Dockerfile.ultra-minimal - Minimal dependencies Docker image
- docker-compose-jetson-fixed.yml - Main compose file
- docker-compose-ultra-minimal.yml - Minimal compose file
- docker-compose-legacy.yml - For older Docker Compose versions
- docker-setup-jetson-fixed.sh - Enhanced setup script
- test-docker-setup.sh - Docker functionality test script
- requirements.txt - Python dependencies
- DOCKER_README.md - Detailed deployment guide

## Troubleshooting:

If you encounter repository/GPG key errors:
1. Use docker-compose-ultra-minimal.yml instead
2. Check DOCKER_README.md for detailed troubleshooting steps

Generated on: $(Get-Date)
"@

$deployReadme | Out-File -FilePath "$deployDir\DEPLOYMENT_README.md" -Encoding UTF8

# Create a simple transfer script
$transferScript = @"
#!/bin/bash
# Simple file transfer script
# Run this on your local machine to transfer files to Jetson Nano

# Update these variables for your setup
JETSON_IP="192.168.1.100"  # Replace with your Jetson Nano IP
JETSON_USER="your-username" # Replace with your username
JETSON_PATH="/home/your-username/yolo-seed-detection"  # Target directory

echo "Transferring files to Jetson Nano..."
echo "Target: \$JETSON_USER@\$JETSON_IP:\$JETSON_PATH"

# Create target directory
ssh \$JETSON_USER@\$JETSON_IP "mkdir -p \$JETSON_PATH"

# Transfer files
scp -r ./* \$JETSON_USER@\$JETSON_IP:\$JETSON_PATH/

echo "Transfer complete!"
echo "Now SSH into your Jetson Nano and run:"
echo "cd \$JETSON_PATH"
echo "./docker-setup-jetson-fixed.sh"
"@

$transferScript | Out-File -FilePath "$deployDir\transfer-to-jetson.sh" -Encoding UTF8

Write-Status "Created deployment package in '$deployDir' directory"

# Display summary
Write-Host "`n=== Deployment Package Created ===" -ForegroundColor Green
Write-Host "Location: $deployDir" -ForegroundColor Cyan
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Transfer the '$deployDir' folder to your Jetson Nano" -ForegroundColor White
Write-Host "2. SSH into your Jetson Nano" -ForegroundColor White  
Write-Host "3. Navigate to the transferred directory" -ForegroundColor White
Write-Host "4. Run: chmod +x *.sh" -ForegroundColor White
Write-Host "5. Run: ./docker-setup-jetson-fixed.sh" -ForegroundColor White
Write-Host "6. Run: docker-compose -f docker-compose-jetson-fixed.yml up --build" -ForegroundColor White

Write-Host "`nTransfer options:" -ForegroundColor Yellow
Write-Host "- Edit and use transfer-to-jetson.sh for SCP transfer" -ForegroundColor White
Write-Host "- Use WinSCP, FileZilla, or similar GUI tools" -ForegroundColor White
Write-Host "- Use USB drive or SD card for physical transfer" -ForegroundColor White

Write-Host "`nFor troubleshooting, see DOCKER_README.md in the deployment package" -ForegroundColor Cyan
