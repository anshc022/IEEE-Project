import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO

def test_cuda_performance():
    """Test CUDA performance with video processing"""
    
    print("🔍 CUDA Performance Test")
    print("=" * 50)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"🔧 CUDA Version: {torch.version.cuda}")
    else:
        print("❌ CUDA Not Available")
        return
    
    # Load model
    print("\n🚀 Loading YOLO model...")
    model = YOLO("corn.pt")
    print(f"📱 Model device: {model.device}")
    
    # Test with camera
    print("\n📹 Testing with camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    total_inference_time = 0
    total_frame_time = 0
    
    print("🎥 Processing 30 frames for performance test...")
    print("Press 'q' to quit early")
    
    while frame_count < 30:
        frame_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read frame")
            break
        
        # Run inference
        inference_start = time.time()
        results = model(frame, verbose=False)
        inference_end = time.time()
        
        inference_time = inference_end - inference_start
        total_inference_time += inference_time
        
        # Draw results
        annotated_frame = results[0].plot()
        
        frame_end = time.time()
        frame_time = frame_end - frame_start
        total_frame_time += frame_time
        
        frame_count += 1
        
        # Display performance info on frame
        fps = 1.0 / frame_time if frame_time > 0 else 0
        inference_fps = 1.0 / inference_time if inference_time > 0 else 0
        
        cv2.putText(annotated_frame, f"Frame: {frame_count}/30", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Inference FPS: {inference_fps:.1f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Device: {model.device}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow('CUDA Performance Test', annotated_frame)
        
        # Print progress
        if frame_count % 5 == 0:
            print(f"📊 Frame {frame_count}: {fps:.1f} FPS, Inference: {inference_fps:.1f} FPS")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Calculate final statistics
    avg_fps = frame_count / total_frame_time if total_frame_time > 0 else 0
    avg_inference_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
    
    print("\n📈 PERFORMANCE RESULTS")
    print("=" * 50)
    print(f"🎯 Total Frames Processed: {frame_count}")
    print(f"⚡ Average FPS: {avg_fps:.1f}")
    print(f"🧠 Average Inference FPS: {avg_inference_fps:.1f}")
    print(f"⏱️  Average Frame Time: {total_frame_time/frame_count*1000:.1f}ms")
    print(f"🔥 Average Inference Time: {total_inference_time/frame_count*1000:.1f}ms")
    print(f"💻 Using Device: {model.device}")
    
    # Performance assessment
    if avg_fps > 20:
        print("✅ EXCELLENT: Real-time performance achieved!")
    elif avg_fps > 10:
        print("✅ GOOD: Acceptable performance for most applications")
    elif avg_fps > 5:
        print("⚠️  MODERATE: Performance could be better")
    else:
        print("❌ POOR: Performance issues detected")
    
    # GPU memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        memory_cached = torch.cuda.memory_reserved() / 1024**2
        print(f"🎮 GPU Memory Allocated: {memory_allocated:.1f} MB")
        print(f"🎮 GPU Memory Cached: {memory_cached:.1f} MB")

if __name__ == "__main__":
    test_cuda_performance()