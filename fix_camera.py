#!/usr/bin/env python3
"""
Quick camera fix script for Jetson Nano green screen issues
Run this if app.py shows green screen or camera issues
"""

import cv2
import numpy as np
import os
import sys

def fix_opencv_qt():
    """Fix OpenCV Qt issues"""
    # Set proper Qt backend
    if os.environ.get('DISPLAY'):
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
        print("✅ Set Qt backend to xcb (with display)")
    else:
        print("⚠️ No DISPLAY detected - running headless")
        return False
    return True

def test_usb_camera_with_fixes():
    """Test USB camera with various fixes for green screen"""
    print("🔧 Testing USB camera with green screen fixes...")
    
    # Try different camera configurations
    configurations = [
        {"index": 0, "width": 640, "height": 480, "format": "default"},
        {"index": 0, "width": 320, "height": 240, "format": "default"},
        {"index": 0, "width": 640, "height": 480, "format": "MJPG"},
        {"index": 0, "width": 1280, "height": 720, "format": "MJPG"},
    ]
    
    for i, config in enumerate(configurations):
        print(f"\n🔍 Testing configuration {i+1}: {config}")
        
        try:
            cap = cv2.VideoCapture(config["index"])
            if not cap.isOpened():
                print("❌ Failed to open camera")
                continue
            
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height"])
            
            # Try to set format if specified
            if config["format"] == "MJPG":
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
                    print("  📹 Set MJPG format")
                except:
                    print("  ⚠️ Could not set MJPG format")
            
            # Additional settings for better compatibility
            cap.set(cv2.CAP_PROP_FPS, 15)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test reading frames
            success_count = 0
            green_screen_count = 0
            
            for frame_num in range(10):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    success_count += 1
                    
                    # Check for green screen
                    mean_channels = np.mean(frame, axis=(0, 1))
                    
                    # Detect all-green or mostly-green frames
                    if (mean_channels[0] == 0 and mean_channels[2] == 0 and mean_channels[1] > 0) or \
                       (mean_channels[1] > mean_channels[0] * 3 and mean_channels[1] > mean_channels[2] * 3):
                        green_screen_count += 1
                        
                        if frame_num == 0:  # Only try fix on first frame
                            print(f"  🟢 Green screen detected - means: {mean_channels}")
                            
                            # Try to fix green screen
                            if mean_channels[0] == 0 and mean_channels[2] == 0:
                                # All green - convert to grayscale
                                try:
                                    gray = frame[:,:,1]  # Use green channel
                                    fixed_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                                    cv2.imwrite(f"fixed_frame_config_{i+1}.jpg", fixed_frame)
                                    print(f"  ✅ Fixed green screen - saved as fixed_frame_config_{i+1}.jpg")
                                except Exception as e:
                                    print(f"  ❌ Could not fix green screen: {e}")
                    
                    if frame_num == 0:
                        # Save first frame for inspection
                        filename = f"test_frame_config_{i+1}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"  📸 Saved test frame: {filename}")
            
            cap.release()
            
            print(f"  📊 Results: {success_count}/10 frames, {green_screen_count} green screens")
            
            if success_count >= 8 and green_screen_count <= 2:
                print(f"  ✅ Configuration {i+1} looks good!")
                return config
            elif success_count >= 5:
                print(f"  ⚠️ Configuration {i+1} partially working")
            else:
                print(f"  ❌ Configuration {i+1} failed")
                
        except Exception as e:
            print(f"  ❌ Configuration {i+1} error: {e}")
    
    return None

def main():
    print("🚀 Jetson Nano Camera Fix Tool")
    print("=" * 50)
    
    # Fix Qt issues first
    if not fix_opencv_qt():
        print("❌ Cannot run with display - this tool requires a display")
        return
    
    # Test camera configurations
    best_config = test_usb_camera_with_fixes()
    
    if best_config:
        print(f"\n🎯 RECOMMENDED CONFIGURATION:")
        print(f"   Camera index: {best_config['index']}")
        print(f"   Resolution: {best_config['width']}x{best_config['height']}")
        print(f"   Format: {best_config['format']}")
        print(f"\n💡 To use this in app.py, modify the USB camera settings accordingly.")
    else:
        print(f"\n❌ No working configuration found!")
        print(f"\n🔧 Troubleshooting steps:")
        print(f"1. Check camera connection")
        print(f"2. Try: sudo chmod 666 /dev/video0")
        print(f"3. Try: sudo modprobe uvcvideo")
        print(f"4. Check: lsusb | grep -i camera")
        print(f"5. Check: v4l2-ctl --list-devices")

if __name__ == "__main__":
    main()
