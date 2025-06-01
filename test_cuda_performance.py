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
import psutil
from typing import Dict, List, Tuple, Union, Any, Callable
import platform

class CUDAPerformanceTester:
    def __init__(self, model_path="corn11.pt"):
        self.model_path = model_path
        self.model = None
        self.device = None
        self.results = {}
        self.baseline_memory = 0
        
    def get_system_info(self) -> Dict:
        """Get detailed system information"""
        info = {
            'os': platform.system(),
            'os_version': platform.version(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_threads': psutil.cpu_count(logical=True),
            'total_ram': f"{psutil.virtual_memory().total / (1024**3):.1f} GB"
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version(),
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_count': torch.cuda.device_count()
            })
        
        return info
        
    def setup_device(self) -> bool:
        """Setup and detect CUDA capabilities with detailed reporting"""
        print("\n🔍 Analyzing System Configuration...")
        
        sys_info = self.get_system_info()
        print(f"OS: {sys_info['os']} {sys_info['os_version']}")
        print(f"Python: {sys_info['python_version']}")
        print(f"PyTorch: {sys_info['pytorch_version']}")
        print(f"CPU Cores: {sys_info['cpu_count']} (Physical), {sys_info['cpu_threads']} (Logical)")
        print(f"System RAM: {sys_info['total_ram']}")
        
        if torch.cuda.is_available():
            self.device = 'cuda'
            gpu_props = torch.cuda.get_device_properties(0)
            
            print("\n🚀 CUDA Configuration:")
            print(f"CUDA Version: {sys_info['cuda_version']}")
            print(f"cuDNN Version: {sys_info['cudnn_version']}")
            print(f"GPU: {sys_info['gpu_name']}")
            print(f"Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            print(f"Total GPU Memory: {gpu_props.total_memory/1024**3:.1f} GB")
            print(f"Multi-Processors: {gpu_props.multi_processor_count}")
            print(f"Warp Size: {gpu_props.warp_size}")
            
            # Test CUDA memory management
            self.baseline_memory = torch.cuda.memory_allocated(0)
            
            # Enable optimizations
            self._enable_cuda_optimizations()
            return True
        else:
            self.device = 'cpu'
            print("\n⚠️ CUDA not available. System will run on CPU.")
            print("💡 To enable CUDA, ensure you have:")
            print("  1. An NVIDIA GPU")
            print("  2. CUDA Toolkit installed")
            print("  3. CUDA-enabled PyTorch installed")
            return False
    
    def _enable_cuda_optimizations(self):
        """Enable all available CUDA optimizations"""
        print("\n⚡ Enabling CUDA Optimizations...")
        
        # Basic optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Memory management
        torch.cuda.empty_cache()
        
        # Advanced settings
        if hasattr(torch.cuda, 'memory_stats'):
            torch.cuda.memory_stats(True)
        
        print("✅ CUDA optimizations enabled")
    
    def test_memory_management(self) -> Dict:
        """Test GPU memory management capabilities"""
        if self.device != 'cuda':
            return {"error": "CUDA not available"}
        
        print("\n🧪 Testing GPU Memory Management...")
        results = {}
        
        # Test memory allocation
        try:
            # Record initial state
            initial_memory = torch.cuda.memory_allocated(0)
            
            # Test large tensor allocation
            test_size = 1000
            test_tensor = torch.zeros((test_size, test_size), device='cuda')
            peak_memory = torch.cuda.max_memory_allocated(0)
            
            # Test memory release
            del test_tensor
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated(0)
            
            results = {
                "initial_memory_mb": initial_memory / 1024**2,
                "peak_memory_mb": peak_memory / 1024**2,
                "final_memory_mb": final_memory / 1024**2,
                "memory_leaked_mb": (final_memory - initial_memory) / 1024**2
            }
            
            print(f"Peak Memory Usage: {results['peak_memory_mb']:.1f} MB")
            print(f"Memory Leaked: {results['memory_leaked_mb']:.1f} MB")
            
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
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

    def test_pcie_bandwidth(self) -> Dict[str, Union[float, str]]:
        """Test PCIe bandwidth between CPU and GPU"""
        if self.device != 'cuda':
            return {"error": "CUDA not available"}
        
        print("\n🔄 Testing PCIe Bandwidth...")
        results: Dict[str, Union[float, str]] = {}
        
        try:
            # Test sizes from 1MB to 1GB
            sizes = [2**i for i in range(20, 30)]  # 1MB to 1GB
            h2d_speeds: List[float] = []  # Host to Device
            d2h_speeds: List[float] = []  # Device to Host
            
            for size in sizes:
                # Host to Device transfer
                cpu_data = torch.randn(size, dtype=torch.float32)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record(None)
                gpu_data = cpu_data.cuda()
                end.record(None)
                torch.cuda.synchronize()
                h2d_speeds.append(size * 4 / (start.elapsed_time(end) / 1000) / 1e9)  # GB/s
                
                # Device to Host transfer
                start.record(None)
                cpu_data = gpu_data.cpu()
                end.record(None)
                torch.cuda.synchronize()
                d2h_speeds.append(size * 4 / (start.elapsed_time(end) / 1000) / 1e9)  # GB/s
                
                del cpu_data, gpu_data
                torch.cuda.empty_cache()
            
            results = {
                "h2d_bandwidth_gbs": float(max(h2d_speeds)),
                "d2h_bandwidth_gbs": float(max(d2h_speeds)),
                "avg_h2d_bandwidth_gbs": float(np.mean(h2d_speeds)),
                "avg_d2h_bandwidth_gbs": float(np.mean(d2h_speeds))
            }
            
            print(f"Host to Device Bandwidth: {results['h2d_bandwidth_gbs']:.1f} GB/s")
            print(f"Device to Host Bandwidth: {results['d2h_bandwidth_gbs']:.1f} GB/s")
            
        except Exception as e:
            results["error"] = str(e)
        
        return results

    def test_multi_gpu(self) -> Dict[str, Any]:
        """Test multi-GPU capabilities if available"""
        results: Dict[str, Any] = {}
        
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            return {"error": "Multiple GPUs not available", "gpu_count": gpu_count}
        
        print(f"\n💻 Testing Multi-GPU Configuration ({gpu_count} GPUs)...")
        
        try:
            # Test peer-to-peer access
            p2p_matrix: List[List[bool]] = []
            for i in range(gpu_count):
                row: List[bool] = []
                for j in range(gpu_count):
                    if i != j:
                        can_access = torch.cuda.can_device_access_peer(i, j)
                        row.append(bool(can_access))
                    else:
                        row.append(True)
                p2p_matrix.append(row)
            
            results["gpu_count"] = gpu_count
            results["peer_to_peer_matrix"] = p2p_matrix
            
            # Test multi-GPU data transfer
            transfer_speeds: List[float] = []
            for i in range(gpu_count):
                for j in range(gpu_count):
                    if i != j:
                        with torch.cuda.device(i):
                            data = torch.randn(1024*1024*32, device=f'cuda:{i}')  # 128MB
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            
                            start.record(None)
                            data_dest = data.to(f'cuda:{j}')
                            end.record(None)
                            torch.cuda.synchronize()
                            
                            speed = float((data.element_size() * data.nelement()) / (start.elapsed_time(end) / 1000) / 1e9)
                            transfer_speeds.append(speed)
                            
                            del data, data_dest
                            torch.cuda.empty_cache()
            
            results["avg_transfer_speed_gbs"] = float(np.mean(transfer_speeds))
            print(f"Average GPU-to-GPU Transfer Speed: {results['avg_transfer_speed_gbs']:.1f} GB/s")
            
        except Exception as e:
            results["error"] = str(e)
        
        return results

    def test_tensor_cores(self) -> Dict[str, Union[bool, float, str]]:
        """Test tensor core capabilities and performance"""
        if self.device != 'cuda':
            return {"error": "CUDA not available"}
        
        print("\n⚡ Testing Tensor Core Capabilities...")
        results: Dict[str, Union[bool, float, str]] = {}
        
        try:
            # Check if tensor cores are available
            gpu_props = torch.cuda.get_device_properties(torch.cuda.current_device())
            has_tensor_cores = bool(gpu_props.major >= 7)  # Volta (sm_70) and newer
            
            if not has_tensor_cores:
                return {"error": "Tensor Cores not available on this GPU"}
            
            # Test FP16 matrix multiplication performance
            matrix_size = 4096
            iterations = 100
            
            # FP32 baseline
            a = torch.randn(matrix_size, matrix_size, device='cuda')
            b = torch.randn(matrix_size, matrix_size, device='cuda')
            
            torch.cuda.synchronize()
            start_fp32 = time.perf_counter()
            
            for _ in range(iterations):
                _ = torch.matmul(a, b)
            
            torch.cuda.synchronize()
            fp32_time = time.perf_counter() - start_fp32
            
            # FP16 with tensor cores
            a_half = a.half()
            b_half = b.half()
            
            torch.cuda.synchronize()
            start_fp16 = time.perf_counter()
            
            for _ in range(iterations):
                _ = torch.matmul(a_half, b_half)
            
            torch.cuda.synchronize()
            fp16_time = time.perf_counter() - start_fp16
            
            speedup = fp32_time / fp16_time
            
            results = {
                "has_tensor_cores": has_tensor_cores,
                "fp32_time": float(fp32_time),
                "fp16_time": float(fp16_time),
                "speedup": float(speedup)
            }
            
            print(f"Tensor Core Speedup: {speedup:.2f}x")
            print(f"FP16 Time: {fp16_time:.3f}s")
            print(f"FP32 Time: {fp32_time:.3f}s")
            
        except Exception as e:
            results["error"] = str(e)
        
        return results

    def test_memory_fragmentation(self) -> Dict[str, Union[float, str]]:
        """Test GPU memory fragmentation and allocation patterns"""
        if self.device != 'cuda':
            return {"error": "CUDA not available"}
        
        print("\n🧩 Testing Memory Fragmentation...")
        results: Dict[str, Union[float, str]] = {}
        
        try:
            # Initial state
            torch.cuda.empty_cache()
            initial_memory = float(torch.cuda.memory_allocated(0))
            
            # Create fragmentation by allocating and freeing different sized tensors
            tensors: List[torch.Tensor] = []
            allocation_sizes = [2**i for i in range(20, 25)]  # 1MB to 16MB
            
            for size in allocation_sizes:
                tensors.append(torch.zeros(size, device='cuda'))
            
            # Free every other tensor
            for i in range(0, len(tensors), 2):
                del tensors[i:i+1]  # Delete slice to avoid index shifting
            
            # Try to allocate a large tensor
            try:
                large_tensor = torch.zeros(2**26, device='cuda')  # 64MB
                del large_tensor
                fragmentation_impact = "Low"
            except RuntimeError:
                fragmentation_impact = "High"
            
            # Cleanup
            del tensors
            torch.cuda.empty_cache()
            
            final_memory = float(torch.cuda.memory_allocated(0))
            
            results = {
                "initial_memory_mb": initial_memory / 1024**2,
                "final_memory_mb": final_memory / 1024**2,
                "fragmentation_impact": fragmentation_impact,
                "memory_leaked_mb": (final_memory - initial_memory) / 1024**2
            }
            
            print(f"Fragmentation Impact: {fragmentation_impact}")
            print(f"Memory Leaked: {results['memory_leaked_mb']:.1f} MB")
            
        except Exception as e:
            results["error"] = str(e)
        
        return results

    def run_comprehensive_cuda_test(self) -> Dict[str, Any]:
        """Run all CUDA performance tests and generate a comprehensive report"""
        print("🚀 Starting Comprehensive CUDA Performance Test...")
        
        report: Dict[str, Any] = {
            "system_info": self.get_system_info(),
            "tests": {}
        }
        
        # Basic CUDA setup test
        try:
            print("\n📋 Testing CUDA Setup...")
            cuda_available = torch.cuda.is_available()
            if not cuda_available:
                return {
                    "error": "CUDA not available",
                    "system_info": report["system_info"]
                }
            
            report["tests"]["cuda_setup"] = {
                "cuda_available": cuda_available,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_capability": torch.cuda.get_device_capability(),
                "device_name": torch.cuda.get_device_name(),
            }
            
            # Run all tests
            test_functions: List[Tuple[str, Callable[[], Dict[str, Any]]]] = [
                ("pcie_bandwidth", self.test_pcie_bandwidth),
                ("memory_management", self.test_memory_management),
                ("memory_fragmentation", self.test_memory_fragmentation),
                ("tensor_cores", self.test_tensor_cores)
            ]
            
            if torch.cuda.device_count() > 1:
                test_functions.append(("multi_gpu", self.test_multi_gpu))
            
            for test_name, test_func in test_functions:
                print(f"\n🔍 Running {test_name} test...")
                try:
                    result: Dict[str, Any] = test_func()
                    report["tests"][test_name] = result
                except Exception as e:
                    report["tests"][test_name] = {
                        "error": f"Test failed: {str(e)}"
                    }
            
            # Summary
            print("\n📊 Test Summary:")
            success_count = sum(1 for test in report["tests"].values() if "error" not in test)
            total_tests = len(report["tests"])
            
            report["summary"] = {
                "total_tests": total_tests,
                "successful_tests": success_count,
                "failed_tests": total_tests - success_count,
                "success_rate": f"{(success_count/total_tests)*100:.1f}%"
            }
            
            print(f"Total Tests: {total_tests}")
            print(f"Successful: {success_count}")
            print(f"Failed: {total_tests - success_count}")
            print(f"Success Rate: {report['summary']['success_rate']}")
            
        except Exception as e:
            report["error"] = str(e)
        
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
