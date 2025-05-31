#!/usr/bin/env python3
"""
CUDA Performance Testing Script
Tests and benchmarks the YOLO seed detection system with various CUDA optimizations
"""

import os
import cv2
import torch
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
import gc

class CUDAPerformanceTester:
    def __init__(self, model_path="corn11.pt"):
        self.model_path = model_path
        self.model = None
        self.device = None
        self.results = {}
        
    def setup_device(self):
        """Setup and detect CUDA capabilities"""
        print("🔧 Setting up device...")
        
        if torch.cuda.is_available():
            self.device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_props = torch.cuda.get_device_properties(0)
            memory_gb = gpu_props.total_memory / 1024**3
            
            print(f"✅ CUDA Device: {gpu_name}")
            print(f"📊 Total Memory: {memory_gb:.1f} GB")
            print(f"🔧 Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            print(f"⚡ Multiprocessors: {gpu_props.multiprocessor_count}")
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            return True
        else:
            self.device = 'cpu'
            print("⚠️  CUDA not available, using CPU")
            return False
    
    def load_model(self, optimize=True):
        """Load YOLO model with optimizations"""
        print(f"🤖 Loading model: {self.model_path}")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        
        if self.device == 'cuda' and optimize:
            print("🚀 Applying CUDA optimizations...")
            
            # Fuse layers
            self.model.fuse()
            
            # Convert to half precision
            try:
                self.model.model.half()
                print("✅ FP16 mixed precision enabled")
            except Exception as e:
                print(f"⚠️  FP16 failed: {e}")
            
            # Try torch.compile if available
            try:
                if hasattr(torch, 'compile'):
                    self.model.model = torch.compile(self.model.model, mode='max-autotune')
                    print("✅ torch.compile optimization enabled")
            except Exception as e:
                print(f"ℹ️  torch.compile not available: {e}")
    
    def warmup_model(self, num_warmup=5):
        """Warm up the model with dummy inferences"""
        print(f"🔥 Warming up model ({num_warmup} iterations)...")
        
        warmup_sizes = [320, 416, 640]
        
        for size in warmup_sizes:
            dummy_input = torch.randn(1, 3, size, size, device=self.device)
            if self.device == 'cuda':
                dummy_input = dummy_input.half()
            
            with torch.no_grad():
                for i in range(num_warmup):
                    _ = self.model.predict(dummy_input, verbose=False, device=self.device, half=True)
            
            print(f"  ✅ Warmed up for {size}x{size} input")
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        print("🔥 Warmup completed")
    
    def create_test_images(self, num_images=100):
        """Create synthetic test images"""
        print(f"📸 Creating {num_images} test images...")
        
        images = []
        for i in range(num_images):
            # Create random image with seed-like patterns
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add some seed-like circular patterns
            for _ in range(np.random.randint(0, 5)):
                center = (np.random.randint(50, 590), np.random.randint(50, 430))
                radius = np.random.randint(10, 30)
                color = (np.random.randint(100, 200), np.random.randint(80, 150), np.random.randint(60, 120))
                cv2.circle(img, center, radius, color, -1)
            
            images.append(img)
        
        return images
    
    def benchmark_inference(self, images, batch_sizes=[1], input_sizes=[640]):
        """Benchmark inference performance"""
        print("⏱️  Starting inference benchmarks...")
        
        results = {}
        
        for batch_size in batch_sizes:
            for input_size in input_sizes:
                test_name = f"batch_{batch_size}_size_{input_size}"
                print(f"\n🧪 Testing {test_name}...")
                
                # Prepare batches
                batches = []
                for i in range(0, len(images), batch_size):
                    batch_images = images[i:i+batch_size]
                    if len(batch_images) == batch_size:
                        batches.append(batch_images)
                
                if not batches:
                    continue
                
                # Warm up for this configuration
                for _ in range(3):
                    _ = self.model.predict(batches[0][0], imgsz=input_size, verbose=False, device=self.device, half=True)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                # Benchmark
                times = []
                memory_usage = []
                
                for batch in batches[:20]:  # Test first 20 batches
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                        start_memory = torch.cuda.memory_allocated(0) / 1024**3
                    
                    start_time = time.time()
                    
                    with torch.no_grad():
                        for img in batch:
                            _ = self.model.predict(
                                img, 
                                imgsz=input_size, 
                                verbose=False, 
                                device=self.device, 
                                half=True,
                                save=False
                            )
                    
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    inference_time = (end_time - start_time) / len(batch)  # Per image
                    times.append(inference_time)
                    
                    if self.device == 'cuda':
                        end_memory = torch.cuda.memory_allocated(0) / 1024**3
                        memory_usage.append(end_memory)
                
                # Calculate statistics
                avg_time = np.mean(times) * 1000  # Convert to ms
                std_time = np.std(times) * 1000
                fps = 1.0 / np.mean(times)
                
                result = {
                    'avg_time_ms': avg_time,
                    'std_time_ms': std_time,
                    'fps': fps,
                    'batch_size': batch_size,
                    'input_size': input_size
                }
                
                if self.device == 'cuda' and memory_usage:
                    result['avg_memory_gb'] = np.mean(memory_usage)
                    result['peak_memory_gb'] = np.max(memory_usage)
                
                results[test_name] = result
                
                print(f"  📊 Avg: {avg_time:.1f}±{std_time:.1f}ms | FPS: {fps:.1f}")
                if self.device == 'cuda' and memory_usage:
                    print(f"  🧠 Memory: {np.mean(memory_usage):.2f}GB (peak: {np.max(memory_usage):.2f}GB)")
        
        return results
    
    def memory_stress_test(self, images):
        """Test memory usage under stress"""
        print("\n🧠 Memory stress test...")
        
        if self.device != 'cuda':
            print("Skipping memory test (CPU mode)")
            return {}
        
        initial_memory = torch.cuda.memory_allocated(0) / 1024**3
        print(f"Initial GPU memory: {initial_memory:.2f}GB")
        
        # Process many images without cleanup
        max_memory = initial_memory
        memory_trend = []
        
        for i, img in enumerate(images[:50]):
            _ = self.model.predict(img, verbose=False, device=self.device, half=True, save=False)
            
            current_memory = torch.cuda.memory_allocated(0) / 1024**3
            max_memory = max(max_memory, current_memory)
            memory_trend.append(current_memory)
            
            if i % 10 == 0:
                print(f"  After {i+1} images: {current_memory:.2f}GB")
        
        # Test cleanup
        torch.cuda.empty_cache()
        gc.collect()
        final_memory = torch.cuda.memory_allocated(0) / 1024**3
        
        print(f"Memory after cleanup: {final_memory:.2f}GB")
        print(f"Peak memory usage: {max_memory:.2f}GB")
        
        return {
            'initial_memory_gb': initial_memory,
            'peak_memory_gb': max_memory,
            'final_memory_gb': final_memory,
            'memory_trend': memory_trend
        }
    
    def compare_optimizations(self, images):
        """Compare different optimization levels"""
        print("\n🔬 Comparing optimization levels...")
        
        configs = [
            {'name': 'baseline', 'fuse': False, 'half': False, 'compile': False},
            {'name': 'fused', 'fuse': True, 'half': False, 'compile': False},
            {'name': 'fp16', 'fuse': True, 'half': True, 'compile': False},
        ]
        
        if hasattr(torch, 'compile'):
            configs.append({'name': 'compiled', 'fuse': True, 'half': True, 'compile': True})
        
        comparison_results = {}
        
        for config in configs:
            print(f"\n🧪 Testing {config['name']} configuration...")
            
            # Reload model with specific configuration
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            if config['fuse']:
                self.model.fuse()
            
            if config['half'] and self.device == 'cuda':
                try:
                    self.model.model.half()
                except:
                    pass
            
            if config['compile'] and hasattr(torch, 'compile'):
                try:
                    self.model.model = torch.compile(self.model.model, mode='max-autotune')
                except:
                    pass
            
            # Warmup
            for _ in range(3):
                _ = self.model.predict(images[0], verbose=False, device=self.device, half=config['half'])
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            times = []
            for img in images[:20]:
                start_time = time.time()
                with torch.no_grad():
                    _ = self.model.predict(img, verbose=False, device=self.device, half=config['half'], save=False)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start_time)
            
            avg_time = np.mean(times) * 1000
            fps = 1.0 / np.mean(times)
            
            comparison_results[config['name']] = {
                'avg_time_ms': avg_time,
                'fps': fps,
                'speedup': 1.0  # Will be calculated relative to baseline
            }
            
            print(f"  📊 {config['name']}: {avg_time:.1f}ms | {fps:.1f} FPS")
        
        # Calculate speedup relative to baseline
        baseline_time = comparison_results['baseline']['avg_time_ms']
        for name, result in comparison_results.items():
            result['speedup'] = baseline_time / result['avg_time_ms']
            if name != 'baseline':
                speedup = result['speedup']
                print(f"  🚀 {name}: {speedup:.1f}x speedup over baseline")
        
        return comparison_results
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        print("\n📋 Generating performance report...")
        
        device_info = {
            'device': self.device,
            'cuda_available': torch.cuda.is_available()
        }
        
        if self.device == 'cuda':
            gpu_props = torch.cuda.get_device_properties(0)
            device_info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'memory_gb': gpu_props.total_memory / 1024**3,
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}",
                'cuda_version': torch.version.cuda,
                'pytorch_version': torch.__version__
            })
        
        return {
            'device_info': device_info,
            'benchmark_results': self.results.get('benchmark', {}),
            'memory_results': self.results.get('memory', {}),
            'optimization_comparison': self.results.get('optimization', {}),
            'timestamp': time.time()
        }
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("🚀 Starting comprehensive CUDA performance benchmark...\n")
        
        # Setup
        cuda_available = self.setup_device()
        self.load_model()
        
        # Create test data
        test_images = self.create_test_images(100)
        
        # Warmup
        self.warmup_model()
        
        # Run benchmarks
        self.results['benchmark'] = self.benchmark_inference(
            test_images, 
            batch_sizes=[1], 
            input_sizes=[320, 640, 800]
        )
        
        # Memory test
        if cuda_available:
            self.results['memory'] = self.memory_stress_test(test_images)
        
        # Optimization comparison
        self.results['optimization'] = self.compare_optimizations(test_images)
        
        # Generate report
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("🎯 BENCHMARK SUMMARY")
        print("="*60)
        
        if self.device == 'cuda':
            print(f"GPU: {report['device_info']['gpu_name']}")
            print(f"Memory: {report['device_info']['memory_gb']:.1f}GB")
            print(f"CUDA: {report['device_info']['cuda_version']}")
        else:
            print("Device: CPU")
        
        print(f"PyTorch: {torch.__version__}")
        
        print("\n📊 Best Performance:")
        best_result = min(self.results['benchmark'].values(), key=lambda x: x['avg_time_ms'])
        print(f"  ⚡ {best_result['fps']:.1f} FPS ({best_result['avg_time_ms']:.1f}ms)")
        print(f"  🖼️  Input Size: {best_result['input_size']}x{best_result['input_size']}")
        
        if 'optimization' in self.results:
            print("\n🚀 Optimization Benefits:")
            for name, result in self.results['optimization'].items():
                if name != 'baseline':
                    print(f"  {name}: {result['speedup']:.1f}x speedup")
        
        return report

def main():
    """Main function"""
    print("🌱 YOLO Seed Detection - CUDA Performance Benchmark")
    print("="*60)
    
    # Check if model exists
    model_path = "corn11.pt"
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure the model file is in the current directory.")
        return
    
    try:
        tester = CUDAPerformanceTester(model_path)
        report = tester.run_full_benchmark()
        
        # Save detailed report
        import json
        report_path = "cuda_performance_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_path}")
        print("🎉 Benchmark completed successfully!")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
