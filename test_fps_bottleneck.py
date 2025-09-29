import cv2
import torch
from ultralytics import YOLO
import time
import numpy as np

def test_frame_processing_speed():
    """Test actual frame processing speed vs inference speed"""
    
    print("🔍 Frame Processing Speed Test")
    print("=" * 50)
    
    # Load model
    model = YOLO("corn.pt")
    if torch.cuda.is_available():
        print(f"✅ Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Using CPU")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n🎥 Testing frame processing pipeline...")
    
    # Test metrics
    total_frames = 50
    inference_times = []
    encoding_times = []
    complete_times = []
    
    for i in range(total_frames):
        complete_start = time.time()
        
        # 1. Frame capture
        ret, frame = cap.read()
        if not ret:
            continue
            
        # 2. Inference
        inference_start = time.time()
        with torch.no_grad():
            results = model(frame, conf=0.3, verbose=False)
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)
        
        # 3. Draw results (if any)
        if results and len(results) > 0:
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame
            
        # 4. JPEG encoding
        encoding_start = time.time()
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        encoding_time = time.time() - encoding_start
        encoding_times.append(encoding_time)
        
        # 5. Complete cycle
        complete_time = time.time() - complete_start
        complete_times.append(complete_time)
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"📊 Processed {i + 1}/{total_frames} frames...")
    
    cap.release()
    
    # Calculate statistics
    avg_inference = np.mean(inference_times) * 1000
    avg_encoding = np.mean(encoding_times) * 1000
    avg_complete = np.mean(complete_times) * 1000
    
    inference_fps = 1000 / avg_inference
    complete_fps = 1000 / avg_complete
    
    print("\n📈 PERFORMANCE RESULTS")
    print("=" * 50)
    print(f"🧠 Average Inference Time: {avg_inference:.1f}ms ({inference_fps:.1f} FPS)")
    print(f"📸 Average Encoding Time: {avg_encoding:.1f}ms")
    print(f"⏱️  Complete Frame Time: {avg_complete:.1f}ms ({complete_fps:.1f} FPS)")
    print(f"📊 Encoding Overhead: {avg_encoding/avg_complete*100:.1f}%")
    
    # Performance assessment
    print(f"\n🎯 BOTTLENECK ANALYSIS:")
    if avg_encoding > avg_inference:
        print("⚠️  JPEG encoding is the main bottleneck")
        print("💡 Consider reducing JPEG quality or resolution")
    else:
        print("⚠️  Model inference is the main bottleneck")
        print("💡 Consider optimizing model or reducing resolution")
    
    if complete_fps >= 15:
        print("✅ Good performance for web streaming!")
    elif complete_fps >= 10:
        print("⚠️  Moderate performance - consider optimizations")
    else:
        print("❌ Poor performance - significant optimizations needed")
    
    return {
        'inference_fps': inference_fps,
        'complete_fps': complete_fps,
        'encoding_overhead': avg_encoding/avg_complete*100
    }

if __name__ == "__main__":
    test_frame_processing_speed()