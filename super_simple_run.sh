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

# Check if we're on Jetson Nano
if [ -f /etc/nv_tegra_release ]; then
    echo "✅ Jetson Nano detected - installing PyTorch wheel..."
    # Install PyTorch for Jetson Nano
    wget -q --no-check-certificate \
        https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl \
        -O torch-1.11.0-cp38-cp38-linux_aarch64.whl 2>/dev/null
    
    if [ -f "torch-1.11.0-cp38-cp38-linux_aarch64.whl" ]; then
        pip3 install --user torch-1.11.0-cp38-cp38-linux_aarch64.whl 2>/dev/null
        rm torch-1.11.0-cp38-cp38-linux_aarch64.whl
        echo "✅ PyTorch for Jetson Nano installed"
    else
        echo "⚠️  Using fallback PyTorch installation..."
        pip3 install --user torch==1.11.0 2>/dev/null
    fi
else
    echo "🐍 Installing PyTorch (generic Linux)..."
    pip3 install --user torch torchvision 2>/dev/null
fi

# Install other dependencies
echo "📦 Installing other dependencies..."
pip3 install --user ultralytics opencv-python pyserial matplotlib numpy 2>/dev/null

mkdir -p results
echo "🌱 Starting seed detection..."
python3 app.py
