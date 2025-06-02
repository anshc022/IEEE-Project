#!/usr/bin/env python3
"""
Camera Scanner - Helps identify available cameras and their properties
Use this to find your external webcam index
"""

import cv2
import time

def scan_cameras():
    """Scan for available cameras and their properties"""
    print("🔍 Scanning for available cameras...")
    print("=" * 50)
    
    available_cameras = []
    
    for camera_index in range(10):  # Check first 10 camera indices
        try:
            print(f"\n📹 Testing Camera Index {camera_index}...")
            cap = cv2.VideoCapture(camera_index)
            
            if cap.isOpened():
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Try to read a frame
                ret, frame = cap.read()
                
                if ret and frame is not None and frame.size > 0:
                    available_cameras.append({
                        'index': camera_index,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'actual_shape': frame.shape,
                        'working': True
                    })
                    
                    camera_type = "🖥️  Laptop Camera" if camera_index == 0 else "🔌 External Camera"
                    print(f"   ✅ {camera_type} FOUND!")
                    print(f"   📐 Resolution: {width}x{height}")
                    print(f"   🎬 FPS: {fps}")
                    print(f"   📊 Actual frame shape: {frame.shape}")
                    
                    # Save a test image
                    test_filename = f"camera_{camera_index}_test.jpg"
                    cv2.imwrite(test_filename, frame)
                    print(f"   📸 Test image saved: {test_filename}")
                    
                else:
                    print(f"   ❌ Camera {camera_index}: Cannot read frames")
                
                cap.release()
            else:
                print(f"   ❌ Camera {camera_index}: Cannot open")
                
        except Exception as e:
            print(f"   ❌ Camera {camera_index}: Error - {e}")
    
    print("\n" + "=" * 50)
    print("📋 CAMERA SCAN SUMMARY")
    print("=" * 50)
    
    if available_cameras:
        print(f"✅ Found {len(available_cameras)} working camera(s):")
        for cam in available_cameras:
            camera_type = "Laptop Camera" if cam['index'] == 0 else "External Camera"
            print(f"   Camera {cam['index']}: {camera_type} - {cam['width']}x{cam['height']} @ {cam['fps']} FPS")
        
        print("\n💡 RECOMMENDATIONS:")
        if len(available_cameras) > 1:
            external_cameras = [cam for cam in available_cameras if cam['index'] != 0]
            if external_cameras:
                recommended_index = external_cameras[0]['index']
                print(f"   🎯 Use Camera Index {recommended_index} for your external webcam")
                print(f"   📝 Add this to your code: force_external_camera({recommended_index})")
                print(f"   📝 Or run: list_available_cameras() in your script")
            else:
                print(f"   ⚠️  Only laptop camera found. Make sure external webcam is connected.")
        else:
            if available_cameras[0]['index'] == 0:
                print(f"   ⚠️  Only laptop camera found. Connect your external webcam and run this again.")
            else:
                print(f"   🎯 Use Camera Index {available_cameras[0]['index']} for your external webcam")
    else:
        print("❌ No working cameras found!")
        print("   🔧 Troubleshooting:")
        print("   - Make sure your camera is connected")
        print("   - Check camera permissions")
        print("   - Try different USB ports")
        print("   - Restart camera applications")

if __name__ == "__main__":
    scan_cameras()
    input("\nPress Enter to exit...")
