import cv2
import torch
from ultralytics import YOLO
import time

def simple_cuda_video_test():
    """Simple visual test to show CUDA is working with video"""
    
    print("🎥 Simple CUDA Video Test")
    print("=" * 40)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model = YOLO("corn.pt")
    print(f"📱 Model device: {next(model.model.parameters()).device}")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("\n🚀 Starting CUDA-accelerated video processing...")
    print("Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference (CUDA accelerated)
        inference_start = time.time()
        results = model(frame, verbose=False)
        inference_time = time.time() - inference_start
        
        # Draw results
        annotated_frame = results[0].plot()
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        inference_fps = 1.0 / inference_time if inference_time > 0 else 0
        
        # Add performance info
        cv2.putText(annotated_frame, f"CUDA FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Inference: {inference_fps:.1f} FPS", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"GPU: {torch.cuda.get_device_name(0)}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Show CUDA status
        if fps > 20:
            status_text = "✅ EXCELLENT CUDA Performance"
            color = (0, 255, 0)
        elif fps > 15:
            status_text = "✅ Good CUDA Performance"
            color = (0, 255, 255)
        else:
            status_text = "⚠️ CUDA Performance Could Be Better"
            color = (0, 165, 255)
        
        cv2.putText(annotated_frame, status_text, (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow('CUDA Video Test - Press Q to Quit', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final results
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"🎯 Frames processed: {frame_count}")
    print(f"⚡ Average FPS: {avg_fps:.1f}")
    print(f"💻 Device used: {next(model.model.parameters()).device}")
    
    if avg_fps > 20:
        print("✅ CUDA is working excellently for video!")
    elif avg_fps > 15:
        print("✅ CUDA is working well for video!")
    else:
        print("⚠️ CUDA performance could be better")

if __name__ == "__main__":
    simple_cuda_video_test()