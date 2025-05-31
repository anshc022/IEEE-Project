#!/usr/bin/env python3
"""
Simple camera test script for Jetson Nano CSI camera
Use this to test camera functionality and diagnose green screen issues
"""

import cv2
import numpy as np
import time

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    """Return GStreamer pipeline for CSI camera"""
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def test_camera_configuration(config):
    """Test a specific camera configuration"""
    print(f"\n🔍 Testing configuration: {config}")
    
    try:
        if config['type'] == 'csi':
            gst_str = gstreamer_pipeline(
                sensor_id=config.get('sensor_id', 0),
                flip_method=config.get('flip_method', 0),
                framerate=config.get('framerate', 15)
            )
            cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        else:  # USB
            cap = cv2.VideoCapture(config['index'])
        
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return False, None
        
        # Test reading frames
        success_count = 0
        frames_to_test = 5
        
        for i in range(frames_to_test):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                success_count += 1
                if i == 0:  # Analyze first frame
                    print(f"✅ Frame shape: {frame.shape}, dtype: {frame.dtype}")
                    
                    # Check color channels
                    mean_channels = np.mean(frame, axis=(0, 1))
                    print(f"📊 Channel means (B,G,R): {mean_channels}")
                    
                    # Calculate green dominance
                    green_dominance = mean_channels[1] / (mean_channels[0] + mean_channels[2] + 1e-6)
                    print(f"🟢 Green dominance ratio: {green_dominance:.2f}")
                    
                    # Check for green screen
                    if green_dominance > 1.5:
                        print("⚠️ GREEN SCREEN DETECTED!")
                    else:
                        print("✅ Colors look normal")
                    
                    # Save test frame
                    filename = f"test_frame_{config['name']}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Saved test frame as '{filename}'")
        
        cap.release()
        
        print(f"📈 Success rate: {success_count}/{frames_to_test} frames")
        return success_count == frames_to_test, mean_channels if success_count > 0 else None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, None

def main():
    print("🚀 Jetson Nano Camera Test Tool")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        # CSI camera configurations
        {"name": "csi_default", "type": "csi", "sensor_id": 0, "flip_method": 0, "framerate": 15},
        {"name": "csi_flip_180", "type": "csi", "sensor_id": 0, "flip_method": 2, "framerate": 15},
        {"name": "csi_flip_horiz", "type": "csi", "sensor_id": 0, "flip_method": 4, "framerate": 15},
        {"name": "csi_flip_vert", "type": "csi", "sensor_id": 0, "flip_method": 6, "framerate": 15},
        {"name": "csi_sensor1", "type": "csi", "sensor_id": 1, "flip_method": 0, "framerate": 15},
        
        # USB camera fallbacks
        {"name": "usb_0", "type": "usb", "index": 0},
        {"name": "usb_1", "type": "usb", "index": 1},
    ]
    
    working_configs = []
    best_config = None
    best_score = float('inf')
    
    for config in test_configs:
        success, mean_channels = test_camera_configuration(config)
        
        if success:
            working_configs.append(config)
            
            # Score based on green dominance (lower is better)
            if mean_channels is not None:
                green_dominance = mean_channels[1] / (mean_channels[0] + mean_channels[2] + 1e-6)
                if green_dominance < best_score:
                    best_score = green_dominance
                    best_config = config
    
    print(f"\n🎯 RESULTS")
    print("=" * 50)
    print(f"Working configurations: {len(working_configs)}")
    
    for config in working_configs:
        print(f"✅ {config['name']}")
    
    if best_config:
        print(f"\n🏆 BEST CONFIGURATION: {best_config['name']}")
        print(f"   Green dominance: {best_score:.2f}")
        print(f"   Configuration: {best_config}")
        
        if best_config['type'] == 'csi':
            print(f"\n💡 To use this in your app, update the gstreamer_pipeline call with:")
            print(f"   sensor_id={best_config.get('sensor_id', 0)}")
            print(f"   flip_method={best_config.get('flip_method', 0)}")
            print(f"   framerate={best_config.get('framerate', 15)}")
    else:
        print("❌ No working camera configuration found!")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check camera connection to CSI port")
        print("2. Make sure camera ribbon cable is properly seated")
        print("3. Try 'dmesg | grep -i camera' to check for hardware detection")
        print("4. Verify GStreamer is installed: 'gst-launch-1.0 --version'")

if __name__ == "__main__":
    main()
