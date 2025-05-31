#!/bin/bash
# Ultra Simple YOLO Setup - No Docker, Just Python!

echo "🌱 Ultra Simple YOLO Seed Detection Setup"
echo "========================================="

# Make sure we can run python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python3"
    exit 1
fi

echo "🚀 Installing dependencies and running..."

# Check if we're on Jetson Nano
if [ -f /etc/nv_tegra_release ] || grep -q "tegra" /proc/version 2>/dev/null; then
    echo "🤖 Jetson Nano detected - using specialized setup"
    python3 jetson_setup.py
else
    echo "💻 Regular system detected - using standard setup"
    python3 run.py
fi

echo "✅ Done!"
