#!/bin/bash
echo "🔧 Installing Dependencies for Jetson Nano"
echo "=========================================="

# Update system first
echo "📦 Updating system packages..."
sudo apt-get update

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get install -y python3-pip python3-dev python3-opencv

# Install PyTorch for Jetson Nano (specific version)
echo "🔥 Installing PyTorch for Jetson Nano..."
# For Jetson Nano, we need to install torch from specific wheel
pip3 install --user https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl

# Install other dependencies
echo "📦 Installing other Python packages..."
pip3 install --user numpy==1.21.0
pip3 install --user pillow
pip3 install --user pyserial
pip3 install --user matplotlib

# Install ultralytics (YOLOv11)
echo "🤖 Installing Ultralytics YOLO..."
pip3 install --user ultralytics

# Verify installations
echo ""
echo "✅ Verifying installations..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null && echo "✅ PyTorch OK" || echo "❌ PyTorch failed"
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')" 2>/dev/null && echo "✅ OpenCV OK" || echo "❌ OpenCV failed"
python3 -c "from ultralytics import YOLO; print('Ultralytics OK')" 2>/dev/null && echo "✅ Ultralytics OK" || echo "❌ Ultralytics failed"
python3 -c "import serial; print('PySerial OK')" 2>/dev/null && echo "✅ PySerial OK" || echo "❌ PySerial failed"

echo ""
echo "🌱 Now starting seed detection..."
python3 app.py
