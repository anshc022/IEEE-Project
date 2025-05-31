#!/usr/bin/env python3
"""
Simple One-File CUDA YOLO Setup for Jetson Nano
No Docker, No Complexity - Just Works!
"""

import os
import sys
import subprocess
import platform
import time

def run_command(cmd, ignore_errors=False):
    """Run a system command"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0 and not ignore_errors:
            print(f"❌ Command failed: {cmd}")
            print(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        if not ignore_errors:
            print(f"❌ Error running command: {e}")
        return False

def install_python_packages():
    """Install required Python packages"""
    print("📦 Installing Python packages...")
    
    packages = [
        "ultralytics>=8.0.196",
        "opencv-python",
        "numpy",
        "matplotlib",
        "pyserial",
        "Pillow"
    ]
    
    # Check if we're on Jetson Nano and install PyTorch properly
    is_jetson = os.path.exists('/etc/nv_tegra_release') or 'tegra' in platform.platform().lower()
    
    if is_jetson:
        print("🤖 Detected Jetson Nano - installing optimized PyTorch...")
        # Install PyTorch for Jetson
        torch_cmd = "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        run_command(torch_cmd, ignore_errors=True)
    else:
        print("💻 Installing standard PyTorch...")
        packages.append("torch")
        packages.append("torchvision")
        packages.append("torchaudio")
    
    for package in packages:
        print(f"Installing {package}...")
        run_command(f"pip3 install {package}", ignore_errors=True)
    
    print("✅ Python packages installed!")

def test_cuda():
    """Test if CUDA is available"""
    print("🔥 Testing CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("⚠️  CUDA not available - using CPU")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_hardware():
    """Check camera and ESP32"""
    print("🔍 Checking hardware...")
    
    # Check camera
    camera_found = False
    for i in range(5):  # Check video0 to video4
        if os.path.exists(f'/dev/video{i}'):
            print(f"📷 Camera found: /dev/video{i}")
            camera_found = True
            break
    
    if not camera_found:
        print("⚠️  No camera detected - will try anyway")
    
    # Check ESP32 (serial ports)
    esp32_found = False
    serial_ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', '/dev/ttyACM1']
    for port in serial_ports:
        if os.path.exists(port):
            print(f"🔌 ESP32 possibly found: {port}")
            esp32_found = True
            break
    
    if not esp32_found:
        print("⚠️  No ESP32 detected - servo control disabled")

def main():
    """Main setup and run function"""
    print("🌱 Simple CUDA YOLO Setup for Jetson Nano")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists('corn11.pt'):
        print("❌ Model file 'corn11.pt' not found!")
        print("   Please copy your model file to this directory")
        return
    
    # Install packages
    install_python_packages()
    
    # Test CUDA
    cuda_available = test_cuda()
    
    # Check hardware
    check_hardware()
    
    print("\n" + "=" * 50)
    print("🚀 Starting YOLO Seed Detection...")
    
    # Run the main application
    try:
        import app
        print("✅ Application started successfully!")
    except ImportError as e:
        print(f"❌ Failed to import app.py: {e}")
        print("   Make sure app.py is in the same directory")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    main()
