#!/bin/bash
echo "🚀 ULTIMATE JETSON NANO SETUP"
echo "============================="
echo "This will install everything needed for your seed detection"
echo ""

# Check if we're on Jetson Nano
if [ -f /etc/nv_tegra_release ]; then
    echo "✅ Jetson Nano detected"
    JETSON=true
else
    echo "⚠️  Not a Jetson Nano, using generic Linux setup"
    JETSON=false
fi

# Update system
echo "📦 Updating system..."
sudo apt-get update

# Install basic dependencies
echo "📦 Installing system packages..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    python3-numpy \
    python3-matplotlib \
    v4l-utils \
    curl \
    wget

# Install PyTorch (Jetson-specific or generic)
if [ "$JETSON" = true ]; then
    echo "🔥 Installing PyTorch for Jetson Nano..."
    # Install PyTorch wheel for Jetson
    wget -q https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.11.0-cp38-cp38-linux_aarch64.whl
    pip3 install --user torch-1.11.0-cp38-cp38-linux_aarch64.whl
    rm torch-1.11.0-cp38-cp38-linux_aarch64.whl
else
    echo "🔥 Installing PyTorch (generic)..."
    pip3 install --user torch torchvision
fi

# Install other Python packages
echo "📦 Installing Python packages..."
pip3 install --user \
    pyserial \
    pillow \
    requests \
    tqdm \
    psutil

# Install Ultralytics YOLO
echo "🤖 Installing Ultralytics YOLO..."
pip3 install --user ultralytics

# Create results directory
mkdir -p results

# Test everything
echo ""
echo "🧪 Testing installations..."

# Test Python imports
python3 -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
except ImportError as e:
    print(f'❌ PyTorch: {e}')

try:
    import cv2
    print(f'✅ OpenCV: {cv2.__version__}')
except ImportError as e:
    print(f'❌ OpenCV: {e}')

try:
    from ultralytics import YOLO
    print('✅ Ultralytics: OK')
except ImportError as e:
    print(f'❌ Ultralytics: {e}')

try:
    import serial
    print('✅ PySerial: OK')
except ImportError as e:
    print(f'❌ PySerial: {e}')

try:
    import numpy as np
    print(f'✅ NumPy: {np.__version__}')
except ImportError as e:
    print(f'❌ NumPy: {e}')
"

# Check camera
echo ""
echo "📷 Checking camera..."
if [ -e /dev/video0 ]; then
    echo "✅ Camera found at /dev/video0"
else
    echo "⚠️  No camera at /dev/video0"
    echo "Available video devices:"
    ls -la /dev/video* 2>/dev/null || echo "None found"
fi

# Check model file
echo ""
echo "🧠 Checking model..."
if [ -f "corn11.pt" ]; then
    echo "✅ Model file corn11.pt found"
else
    echo "❌ Model file corn11.pt not found!"
    echo "Make sure corn11.pt is in the current directory"
fi

echo ""
echo "🌱 Starting seed detection..."
echo "Press Ctrl+C to stop"
echo ""

# Run the application
python3 app.py
