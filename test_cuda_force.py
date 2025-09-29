import torch
import cv2
import time
from ultralytics import YOLO

def force_cuda_test():
    """Force CUDA usage and test performance"""
    
    print("🔥 CUDA Force Test")
    print("=" * 40)
    
    # Check CUDA
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    # Load model and force to CUDA
    print("\n🚀 Loading model...")
    model = YOLO("corn.pt")
    
    # Force model to CUDA
    if torch.cuda.is_available():
        model.model = model.model.cuda()
        print(f"✅ Model forced to CUDA")
        print(f"Model device: {next(model.model.parameters()).device}")
    
    # Test with a single frame
    print("\n📸 Testing single frame inference...")
    
    # Create test image
    test_image = torch.randn(3, 640, 640)
    if torch.cuda.is_available():
        test_image = test_image.cuda()
    
    # Warm up
    print("🔥 Warming up...")
    for _ in range(5):
        with torch.no_grad():
            results = model(test_image, verbose=False)
    
    # Time inference
    print("⏱️  Timing inference...")
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            results = model(test_image, verbose=False)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    fps = 1.0 / avg_time
    
    print(f"\n📊 RESULTS:")
    print(f"Average inference time: {avg_time*1000:.2f}ms")
    print(f"Theoretical FPS: {fps:.1f}")
    
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU Memory Used: {memory_allocated:.1f} MB")
    
    if fps > 30:
        print("✅ EXCELLENT CUDA Performance!")
    elif fps > 15:
        print("✅ GOOD CUDA Performance")
    else:
        print("⚠️  CUDA Performance could be better")
    
    return fps

def test_cpu_vs_cuda():
    """Compare CPU vs CUDA performance"""
    print("\n🏁 CPU vs CUDA Comparison")
    print("=" * 40)
    
    # Test CPU
    print("Testing CPU...")
    model_cpu = YOLO("corn.pt")
    model_cpu.model = model_cpu.model.cpu()
    
    test_image = torch.randn(3, 640, 640)
    
    # Warm up CPU
    for _ in range(3):
        results = model_cpu(test_image, verbose=False)
    
    # Time CPU
    start_time = time.time()
    for _ in range(10):
        results = model_cpu(test_image, verbose=False)
    cpu_time = (time.time() - start_time) / 10
    
    print(f"CPU Average: {cpu_time*1000:.2f}ms ({1.0/cpu_time:.1f} FPS)")
    
    if torch.cuda.is_available():
        # Test CUDA
        print("Testing CUDA...")
        model_cuda = YOLO("corn.pt")
        model_cuda.model = model_cuda.model.cuda()
        test_image_cuda = test_image.cuda()
        
        # Warm up CUDA
        for _ in range(3):
            results = model_cuda(test_image_cuda, verbose=False)
        
        # Time CUDA
        start_time = time.time()
        for _ in range(10):
            results = model_cuda(test_image_cuda, verbose=False)
        cuda_time = (time.time() - start_time) / 10
        
        print(f"CUDA Average: {cuda_time*1000:.2f}ms ({1.0/cuda_time:.1f} FPS)")
        
        speedup = cpu_time / cuda_time
        print(f"\n🚀 CUDA Speedup: {speedup:.1f}x faster")
        
        if speedup > 2:
            print("✅ CUDA is working excellently!")
        elif speedup > 1.2:
            print("✅ CUDA is working well")
        else:
            print("⚠️  CUDA speedup is minimal")

if __name__ == "__main__":
    force_cuda_test()
    test_cpu_vs_cuda()