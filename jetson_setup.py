#!/usr/bin/env python3
"""
Jetson Nano Specific Setup Script
Handles PyTorch installation properly for ARM64 architecture
"""

import os
import sys
import subprocess
import platform

def run_cmd(cmd, ignore_errors=False):
    """Run command and return success"""
    print(f"⚡ Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=not ignore_errors)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        if not ignore_errors:
            print(f"❌ Failed: {e}")
        return False

def install_jetson_pytorch():
    """Install PyTorch specifically for Jetson Nano"""
    print("🤖 Installing PyTorch for Jetson Nano...")
    
    # Method 1: Try NVIDIA's prebuilt wheel
    print("📦 Method 1: NVIDIA prebuilt PyTorch wheel...")
    pytorch_wheel = "https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl"
    if run_cmd(f"pip3 install {pytorch_wheel}", ignore_errors=True):
        return True
    
    # Method 2: Try PyTorch with CPU-only
    print("📦 Method 2: CPU-only PyTorch...")
    if run_cmd("pip3 install torch==1.13.0+cpu torchvision==0.14.0+cpu -f https://download.pytorch.org/whl/torch_stable.html", ignore_errors=True):
        return True
    
    # Method 3: Try standard pip install
    print("📦 Method 3: Standard PyTorch installation...")
    if run_cmd("pip3 install torch torchvision", ignore_errors=True):
        return True
    
    print("❌ All PyTorch installation methods failed")
    return False

def install_packages():
    """Install all required packages"""
    print("📦 Installing Python packages...")
    
    # Update pip first
    run_cmd("pip3 install --upgrade pip", ignore_errors=True)
    
    # Install PyTorch for Jetson
    if not install_jetson_pytorch():
        print("⚠️  PyTorch installation failed - continuing with other packages")
    
    # Install other packages
    packages = [
        "ultralytics",
        "opencv-python",
        "numpy",
        "matplotlib", 
        "pyserial",
        "Pillow"
    ]
    
    for pkg in packages:
        print(f"📦 Installing {pkg}...")
        run_cmd(f"pip3 install {pkg}", ignore_errors=True)

def test_installation():
    """Test if everything is working"""
    print("🧪 Testing installation...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA not available")
    except ImportError:
        print("❌ PyTorch not working")
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not working")
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO working")
    except ImportError:
        print("❌ Ultralytics not working")

def main():
    """Main setup function"""
    print("🤖 Jetson Nano YOLO Setup")
    print("=" * 40)
    
    # Check if we're on Jetson
    if not (os.path.exists('/etc/nv_tegra_release') or 'tegra' in platform.platform().lower()):
        print("⚠️  This script is for Jetson Nano only!")
        print("   Use run.py for other systems")
        return
    
    print("✅ Jetson Nano detected")
    
    # Install packages
    install_packages()
    
    # Test installation
    test_installation()
    
    print("\n🚀 Setup complete! Running main app...")
    
    # Run the app
    try:
        exec(open('app.py').read())
    except FileNotFoundError:
        print("❌ app.py not found in current directory")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    main()
