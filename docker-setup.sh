#!/bin/bash

# Docker Setup Script for Jetson Nano
# This script prepares the Jetson Nano for running the YOLOv11 seed detection application

set -e

echo "=== Jetson Nano Docker Setup for YOLOv11 Seed Detection ==="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Jetson Nano
if ! grep -q "jetson" /proc/device-tree/model 2>/dev/null; then
    print_warning "This script is designed for Jetson Nano. Proceeding anyway..."
fi

# Update system
print_status "Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    print_status "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
else
    print_status "Docker is already installed"
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    print_status "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
else
    print_status "Docker Compose is already installed"
fi

# Setup NVIDIA Docker runtime for GPU support
print_status "Setting up NVIDIA Docker runtime..."
if [ ! -f /etc/docker/daemon.json ]; then
    sudo mkdir -p /etc/docker
    sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
EOF
else
    print_warning "Docker daemon.json already exists. Please manually add NVIDIA runtime if needed."
fi

# Restart Docker service
print_status "Restarting Docker service..."
sudo systemctl restart docker

# Enable Docker service
sudo systemctl enable docker

# Setup camera permissions
print_status "Setting up camera permissions..."
sudo usermod -a -G video $USER

# Create necessary directories
print_status "Creating application directories..."
mkdir -p logs data calibration

# Set up X11 forwarding for GUI applications
print_status "Setting up X11 permissions for GUI applications..."
if [ -n "$DISPLAY" ]; then
    xhost +local:docker 2>/dev/null || print_warning "Could not set X11 permissions. GUI may not work."
else
    print_warning "DISPLAY variable not set. Setting to :0 for local display."
    export DISPLAY=:0
    xhost +local:docker 2>/dev/null || print_warning "Could not set X11 permissions. GUI may not work."
fi

# Check CUDA installation
print_status "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    print_status "CUDA version $CUDA_VERSION detected"
else
    print_warning "CUDA not found. GPU acceleration may not work."
fi

# Verify camera devices
print_status "Checking camera devices..."
if ls /dev/video* 1> /dev/null 2>&1; then
    print_status "Camera devices found:"
    ls -la /dev/video*
else
    print_warning "No camera devices found in /dev/video*"
fi

# Check for CSI camera
if [ -c /dev/video0 ] || [ -c /dev/video1 ]; then
    print_status "Video devices detected"
else
    print_warning "No video devices found. Check camera connections."
fi

print_status "Setup completed successfully!"
print_status "You may need to log out and back in for group changes to take effect."
print_status ""

# Detect Docker Compose version and recommend appropriate command
print_status "Detecting Docker Compose version..."
if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
    MAJOR_VERSION=$(echo $COMPOSE_VERSION | cut -d. -f1)
    MINOR_VERSION=$(echo $COMPOSE_VERSION | cut -d. -f2)
    
    print_status "Docker Compose version: $COMPOSE_VERSION"
    
    if [[ $MAJOR_VERSION -gt 1 ]] || [[ $MAJOR_VERSION -eq 1 && $MINOR_VERSION -ge 27 ]]; then
        print_status "Using standard docker-compose.yml (v3.3 format with GPU deploy syntax)"
        COMPOSE_FILE="docker-compose.yml"
    elif [[ $MAJOR_VERSION -eq 1 && $MINOR_VERSION -ge 18 ]]; then
        print_status "Using compatible docker-compose-compatible.yml (v3.3 format, no GPU deploy)"
        COMPOSE_FILE="docker-compose-compatible.yml"
    else
        print_status "Using very old docker-compose-v1.yml (v1 format)"
        COMPOSE_FILE="docker-compose-v1.yml"
    fi
else
    print_warning "Docker Compose not found. Using direct docker run command."
    COMPOSE_FILE=""
fi

print_status ""
print_status "To build and run the application:"
print_status "1. cd to your project directory"

if [ -n "$COMPOSE_FILE" ]; then
    if [ "$COMPOSE_FILE" = "docker-compose.yml" ]; then
        print_status "2. Run: docker-compose up --build"
    else
        print_status "2. Run: docker-compose -f $COMPOSE_FILE up --build"
    fi
    print_status ""
    print_status "To run in background: docker-compose -f $COMPOSE_FILE up -d --build"
    print_status "To view logs: docker-compose -f $COMPOSE_FILE logs -f"
    print_status "To stop: docker-compose -f $COMPOSE_FILE down"
else
    print_status "2. Run: docker build -t yolo-seed-detection ."
    print_status "3. Run: docker run --runtime=nvidia --privileged -it \\"
    print_status "   -e NVIDIA_VISIBLE_DEVICES=all -e DISPLAY=\$DISPLAY \\"
    print_status "   -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /dev:/dev \\"
    print_status "   --net=host yolo-seed-detection"
fi
