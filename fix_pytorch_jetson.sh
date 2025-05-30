#!/bin/bash
echo "⚡ JETSON NANO QUICK TORCH FIX"
echo "============================="

# This specifically fixes PyTorch installation on Jetson Nano

echo "🔥 Installing PyTorch for Jetson Nano..."

# Method 1: Try pre-built wheel
echo "📦 Trying pre-built PyTorch wheel..."
wget -q --no-check-certificate \
    https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl \
    -O torch-1.11.0-cp38-cp38-linux_aarch64.whl

if [ -f "torch-1.11.0-cp38-cp38-linux_aarch64.whl" ]; then
    pip3 install --user torch-1.11.0-cp38-cp38-linux_aarch64.whl
    rm torch-1.11.0-cp38-cp38-linux_aarch64.whl
    echo "✅ PyTorch wheel installed"
else
    echo "⚠️  Wheel download failed, trying pip install..."
    # Method 2: Try pip install with specific index
    pip3 install --user torch==1.11.0
fi

# Install torchvision
echo "📦 Installing torchvision..."
pip3 install --user torchvision

# Test PyTorch
echo "🧪 Testing PyTorch..."
python3 -c "
try:
    import torch
    print(f'✅ PyTorch {torch.__version__} installed successfully')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA device: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ PyTorch test failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🌱 PyTorch is ready! Now installing other dependencies..."
    
    # Install remaining packages
    pip3 install --user ultralytics opencv-python pyserial matplotlib numpy
    
    echo ""
    echo "🚀 Starting seed detection..."
    python3 app.py
else
    echo "❌ PyTorch installation failed"
    echo ""
    echo "💡 Manual installation steps:"
    echo "1. wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl"
    echo "2. pip3 install --user torch-1.11.0-cp38-cp38-linux_aarch64.whl"
    echo "3. pip3 install --user ultralytics opencv-python pyserial matplotlib"
    echo "4. python3 app.py"
fi
