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

# Install packages and run
python3 run.py

echo "✅ Done!"
