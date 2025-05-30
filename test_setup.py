#!/usr/bin/env python3
"""
Simple test script to verify your setup before Docker
"""

import sys
import os

def test_imports():
    print("🔍 Testing Python imports...")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
        print(f"   OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
        print(f"   PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("   CUDA not available (using CPU)")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO imported successfully")
    except ImportError as e:
        print(f"❌ Ultralytics import failed: {e}")
        print("   Install with: pip install ultralytics")
        return False
    
    try:
        import serial
        print("✅ PySerial imported successfully")
    except ImportError as e:
        print(f"❌ PySerial import failed: {e}")
        return False
    
    return True

def test_model():
    print("\n🧠 Testing YOLO model...")
    
    model_path = "corn11.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found!")
        return False
    
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Model classes: {list(model.names.values()) if hasattr(model, 'names') else 'Unknown'}")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_camera():
    print("\n📷 Testing camera...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("⚠️  Camera 0 not available, trying other indices...")
            for i in range(1, 4):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"✅ Camera found at index {i}")
                    break
            else:
                print("❌ No camera found!")
                return False
        else:
            print("✅ Camera 0 is available")
        
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera test successful - Frame size: {frame.shape}")
        else:
            print("❌ Could not read frame from camera")
            return False
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_serial():
    print("\n🔌 Testing serial ports...")
    
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        
        if not ports:
            print("⚠️  No serial ports found")
            return True
        
        print("📋 Available serial ports:")
        esp32_found = False
        for port in ports:
            print(f"   {port.device} - {port.description}")
            if any(keyword in port.description.upper() for keyword in ['USB', 'SERIAL', 'CH340', 'CP210', 'ESP32']):
                print(f"   ✅ Potential ESP32 port: {port.device}")
                esp32_found = True
        
        if not esp32_found:
            print("⚠️  No ESP32-like devices found (servo control will be disabled)")
        
        return True
        
    except Exception as e:
        print(f"❌ Serial port test failed: {e}")
        return False

def main():
    print("🌱 Seed Detection System - Pre-Docker Test")
    print("=" * 50)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_model()
    all_passed &= test_camera()
    all_passed &= test_serial()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Ready to run with Docker.")
        print("\nNext steps:")
        print("1. On Jetson Nano: chmod +x run.sh && ./run.sh")
        print("2. On Windows: run.bat")
    else:
        print("⚠️  Some tests failed. Please fix the issues above before running.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
