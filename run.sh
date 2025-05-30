#!/bin/bash

# Simple script to run seed detection on Jetson Nano
echo "🌱 Starting Seed Detection System on Jetson Nano..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first:"
    echo "   curl -fsSL https://get.docker.com -o get-docker.sh"
    echo "   sudo sh get-docker.sh"
    echo "   sudo usermod -aG docker $USER"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Installing..."
    sudo apt-get update
    sudo apt-get install -y docker-compose
fi

# Make sure user is in docker group
if ! groups $USER | grep -q docker; then
    echo "⚠️  Adding user to docker group..."
    sudo usermod -aG docker $USER
    echo "⚠️  Please log out and log back in, then run this script again"
    exit 1
fi

# Check for camera
if [ ! -e /dev/video0 ]; then
    echo "⚠️  Camera not found at /dev/video0"
    echo "   Available video devices:"
    ls -la /dev/video* 2>/dev/null || echo "   No video devices found"
fi

# Check for ESP32 (optional)
echo "🔍 Checking for ESP32..."
if [ -e /dev/ttyUSB0 ]; then
    echo "✅ Found ESP32 at /dev/ttyUSB0"
elif [ -e /dev/ttyACM0 ]; then
    echo "✅ Found ESP32 at /dev/ttyACM0"
else
    echo "⚠️  ESP32 not found. Servo control will be disabled."
    echo "   Available serial devices:"
    ls -la /dev/tty* | grep -E "(USB|ACM)" || echo "   No USB serial devices found"
fi

# Create results directory
mkdir -p results

# Build and run the container
echo "🏗️  Building Docker image..."

# Try different docker-compose files for compatibility
if docker-compose build 2>/dev/null; then
    echo "✅ Build successful with docker-compose.yml"
    COMPOSE_FILE="docker-compose.yml"
elif docker-compose -f docker-compose-simple.yml build 2>/dev/null; then
    echo "✅ Build successful with docker-compose-simple.yml"
    COMPOSE_FILE="docker-compose-simple.yml"
else
    echo "❌ Build failed. Trying simple Docker run..."
    echo "🐳 Building image directly..."
    docker build -t seed-detection .
    echo "🚀 Running container directly..."
    docker run --rm -it \
        --privileged \
        --device=/dev/video0:/dev/video0 \
        -v $(pwd)/results:/app/results \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -e DISPLAY=$DISPLAY \
        --network host \
        seed-detection
    exit 0
fi

echo "🚀 Starting seed detection system..."
if [ "$COMPOSE_FILE" = "docker-compose-simple.yml" ]; then
    docker-compose -f docker-compose-simple.yml up
else
    docker-compose up
fi

echo "✅ Done! Check the 'results' folder for saved images."
