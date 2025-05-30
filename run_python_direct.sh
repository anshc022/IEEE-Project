#!/bin/bash
echo "🐍 Running Seed Detection WITHOUT Docker"
echo "========================================"
echo "This will run your seed detection directly with Python"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3."
    exit 1
fi

# Create results directory
mkdir -p results

echo "📦 Installing Python dependencies..."
pip3 install --user ultralytics opencv-python torch torchvision pyserial matplotlib numpy

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
else
    echo "⚠️  Some dependencies may have failed to install, but continuing..."
fi

echo ""
echo "🌱 Starting seed detection (Python mode)..."
echo "Press Ctrl+C to stop"
echo ""

# Run the application directly
python3 app.py
