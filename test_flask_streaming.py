import cv2
import torch
from ultralytics import YOLO
import time

def test_flask_streaming_simulation():
    """Simulate Flask streaming to identify bottlenecks"""
    
    print("🌐 Flask Streaming Simulation Test")
    print("=" * 50)
    
    # Load model
    model = YOLO("corn.pt")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("🚀 Testing optimized streaming pipeline...")
    
    frame_count = 0
    start_time = time.time()
    
    # Simulate streaming for 30 seconds
    while time.time() - start_time < 10:  # 10 seconds test
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Inference
        with torch.no_grad():
            results = model(frame, conf=0.3, verbose=False)
            
        # Draw results
        if results and len(results) > 0:
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame
            
        # Optimized encoding
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, 70,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1
        ]
        ret, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)
        
        # Simulate network transmission delay
        frame_bytes = buffer.tobytes()
        
        frame_count += 1
        
        # Display current stats
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed
        
        if frame_count % 20 == 0:
            print(f"📊 Frame {frame_count}: {current_fps:.1f} FPS")
            
        # Frame rate limiting (15 FPS target)
        time.sleep(max(0, 1/15 - (time.time() - (start_time + elapsed))))
    
    cap.release()
    
    total_time = time.time() - start_time
    final_fps = frame_count / total_time
    
    print(f"\n📈 FINAL RESULTS:")
    print(f"🎯 Total frames: {frame_count}")
    print(f"⚡ Average FPS: {final_fps:.1f}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    
    if final_fps >= 12:
        print("✅ Good streaming performance!")
    elif final_fps >= 8:
        print("⚠️  Moderate - check Flask threading")
    else:
        print("❌ Poor - Flask optimization needed")

if __name__ == "__main__":
    test_flask_streaming_simulation()