#!/usr/bin/env python3
"""
Live Camera YOLOv5 Detection Script
Runs YOLOv5 corn detection on live webcam feed
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import time

def load_model(model_path="corn11.pt"):
    """Load YOLOv5 model"""
    try:
        # Load the custom trained model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        model.conf = 0.25  # confidence threshold
        model.iou = 0.45   # IoU threshold
        print(f"✅ Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def run_live_detection(model, camera_index=0):
    """Run live detection on camera feed"""
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Error: Cannot open camera {camera_index}")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("🎥 Camera opened successfully!")
    print("📹 Starting live detection... Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Cannot read frame from camera")
            break
        
        # Run YOLOv5 detection
        results = model(frame)
        
        # Render results on frame
        annotated_frame = results.render()[0]
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Add FPS text to frame
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add instructions
        cv2.putText(annotated_frame, "Press 'q' to quit", (10, annotated_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('YOLOv5 Live Detection - Corn Model', annotated_frame)
        
        # Print detection results
        detections = results.pandas().xyxy[0]  # Get detections as pandas DataFrame
        if not detections.empty:
            print(f"🌽 Detected {len(detections)} objects:")
            for _, detection in detections.iterrows():
                confidence = detection['confidence']
                class_name = detection['name']
                print(f"  - {class_name}: {confidence:.2f}")
        
        # Reset FPS counter every 30 frames
        if frame_count % 30 == 0:
            frame_count = 0
            start_time = time.time()
        
        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' key or ESC
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("🏁 Detection stopped. Camera released.")

def main():
    """Main function"""
    print("🚀 Starting YOLOv5 Live Camera Detection")
    print("=" * 50)
    
    # Load the model
    model = load_model("corn11.pt")
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Run live detection
    try:
        run_live_detection(model, camera_index=0)
    except KeyboardInterrupt:
        print("\n⏹️ Detection interrupted by user")
    except Exception as e:
        print(f"❌ Error during detection: {e}")

if __name__ == "__main__":
    main()
