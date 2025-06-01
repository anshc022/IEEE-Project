#!/usr/bin/env python3
"""
Simple Live Camera Detection using YOLOv5 detect.py
This script uses the official YOLOv5 detect.py with webcam source
"""

import subprocess
import sys
import os
from pathlib import Path

def check_yolov5_installation():
    """Check if YOLOv5 is available"""
    yolov5_path = Path("yolov5")
    if not yolov5_path.exists():
        print("❌ YOLOv5 directory not found!")
        print("🔧 Cloning YOLOv5 repository...")
        try:
            subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"], check=True)
            print("✅ YOLOv5 cloned successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error cloning YOLOv5: {e}")
            return False
    return True

def install_requirements():
    """Install YOLOv5 requirements"""
    requirements_file = Path("yolov5/requirements.txt")
    if requirements_file.exists():
        print("📦 Installing YOLOv5 requirements...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], check=True)
            print("✅ Requirements installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Some requirements might have failed to install: {e}")

def run_live_detection():
    """Run live detection using YOLOv5"""
    model_path = Path("corn11.pt")
    if not model_path.exists():
        print(f"❌ Model file {model_path} not found!")
        return False
    
    print("🎥 Starting live camera detection with corn model...")
    print("📹 Press 'q' in the video window to quit")
    
    # Change to yolov5 directory and run detection
    cmd = [
        sys.executable, "detect.py",
        "--source", "0",  # Use webcam (camera index 0)
        "--weights", "../corn11.pt",  # Path to your custom model
        "--img", "640",  # Image size
        "--conf", "0.25",  # Confidence threshold
        "--view-img",  # Display results
        "--save-txt",  # Save results as txt files
        "--save-conf",  # Save confidences in txt files
    ]
    
    try:
        # Run the detection command in yolov5 directory
        os.chdir("yolov5")
        subprocess.run(cmd, check=True)
        print("🏁 Detection completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running detection: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️ Detection interrupted by user")
        return True
    finally:
        # Change back to original directory
        os.chdir("..")

def main():
    """Main function"""
    print("🚀 YOLOv5 Live Camera Detection Setup")
    print("=" * 50)
    
    # Check and setup YOLOv5
    if not check_yolov5_installation():
        return
    
    # Install requirements
    install_requirements()
    
    # Run live detection
    run_live_detection()

if __name__ == "__main__":
    main()
