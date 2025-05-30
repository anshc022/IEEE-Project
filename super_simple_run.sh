#!/bin/bash
echo "🎯 SUPER SIMPLE - One Command Fix for Jetson Nano"
echo "================================================="
echo ""

# Method 1: Fix Docker permissions and try again
echo "🔧 Method 1: Fixing Docker permissions..."
sudo usermod -aG docker $USER
sudo chmod 666 /var/run/docker.sock
sudo systemctl restart docker

echo "⏳ Waiting for Docker to restart..."
sleep 3

# Try Docker again
echo "🐳 Attempting Docker build..."
if docker build -t seed-detection . 2>/dev/null; then
    echo "✅ Docker build successful!"
    mkdir -p results
    echo "🚀 Starting with Docker..."
    docker run --rm -it \
        --privileged \
        --device=/dev/video0:/dev/video0 \
        -v "$(pwd)/results:/app/results" \
        seed-detection
    exit 0
fi

echo "⚠️  Docker still not working. Trying Python directly..."
echo ""

# Method 2: Run with Python directly (no Docker)
echo "🐍 Method 2: Running with Python directly..."
pip3 install --user ultralytics opencv-python torch pyserial matplotlib numpy 2>/dev/null

mkdir -p results
echo "🌱 Starting seed detection..."
python3 app.py
