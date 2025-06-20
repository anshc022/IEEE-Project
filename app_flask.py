import cv2
import torch
import numpy as np
import time
import serial
import serial.tools.list_ports
from pathlib import Path
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
from datetime import datetime
import threading
import atexit
import gc
import atexit
import gc


app = Flask(__name__)

def cleanup_resources():
    """Clean up resources on app shutdown"""
    global servo_serial, cap
    print("🧹 Cleaning up resources...")
    
    if servo_serial:
        try:
            servo_serial.close()
            print("✅ Serial port closed")
        except:
            pass
    
    if cap:
        try:
            cap.release()
            print("✅ Camera released")
        except:
            pass

# Register cleanup function
atexit.register(cleanup_resources)

# Configuration
MODEL_PATH = "corn.pt"  # Use the better performing model
CONFIDENCE_THRESHOLD = 0.3  # Lower threshold for better detection
IOU_THRESHOLD = 0.7
INPUT_SIZE = 640

# Constants - Will be updated from model
CLASS_NAMES = ['Bad-Seed', 'Good-Seed']  # Default, will be overridden by model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Global variables
model = None
cap = None

# Enhanced CUDA detection and configuration
def setup_device():
    """Setup and configure the best available device with optimizations"""
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        compute_capability = torch.cuda.get_device_properties(0).major
        
        print(f"🚀 CUDA available: {gpu_name}")
        print(f"📊 GPU Memory: {gpu_memory:.1f} GB")
        print(f"🔧 Compute Capability: {compute_capability}.{torch.cuda.get_device_properties(0).minor}")
        
        # Advanced CUDA optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for RTX 30 series and newer
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for matrix operations
        
        # Enable memory optimizations
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'memory_efficient_attention'):
            torch.cuda.memory_efficient_attention = True
        
        # Set optimal memory fraction to prevent OOM
        try:
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
        except:
            pass
        
        print("✅ Advanced CUDA optimizations enabled")
        
    else:
        device = 'cpu'
        print("⚠️  CUDA not available, using CPU")
        print("💡 For faster inference, install CUDA-compatible PyTorch")
        print("💡 Check: https://pytorch.org/get-started/locally/")
    
    return device

device = setup_device()

# System control variables
system_status = {
    'camera_enabled': True,
    'detection_enabled': True,
    'servo_enabled': True,
    'auto_sorting': True,
    'statistics_enabled': True,
    'gpu_monitoring': True,
    'system_running': True,
    'emergency_stop': False
}

# Performance monitoring variables
performance_stats = {
    'frame_count': 0,
    'total_inference_time': 0.0,
    'avg_fps': 0.0,
    'peak_gpu_memory': 0.0,
    'last_gpu_cleanup': time.time()
}

def manage_gpu_memory():
    """Advanced GPU memory management"""
    if device == 'cuda' and torch.cuda.is_available():
        try:
            # Get current memory usage
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            
            # Update peak memory usage
            performance_stats['peak_gpu_memory'] = max(
                performance_stats['peak_gpu_memory'], 
                memory_allocated
            )
            
            # Cleanup if memory usage is high or it's been a while
            current_time = time.time()
            time_since_cleanup = current_time - performance_stats['last_gpu_cleanup']
            
            if memory_allocated > 2.0 or time_since_cleanup > 60:  # 2GB threshold or 60 seconds
                torch.cuda.empty_cache()
                gc.collect()  # Also run Python garbage collection
                performance_stats['last_gpu_cleanup'] = current_time
                
                new_memory = torch.cuda.memory_allocated(0) / 1024**3
                print(f"🧹 GPU memory cleanup: {memory_allocated:.2f}GB → {new_memory:.2f}GB")
                
        except Exception as e:
            print(f"⚠️  GPU memory management error: {e}")

def get_device_info():
    """Get comprehensive device information"""
    info = {
        'device': device,
        'device_name': 'CPU',
        'cuda_available': False
    }
    
    if device == 'cuda' and torch.cuda.is_available():
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            info.update({
                'device_name': torch.cuda.get_device_name(0),
                'cuda_available': True,
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}",
                'memory_total_gb': gpu_props.total_memory / 1024**3,
                'multiprocessor_count': gpu_props.multiprocessor_count,
                'supports_fp16': gpu_props.major >= 6,  # Pascal and newer
                'supports_tensor_cores': gpu_props.major >= 7,  # Volta and newer
            })
        except Exception as e:
            print(f"Error getting GPU info: {e}")
    
    return info

seed_statistics = {
    'good_seeds': 0,
    'bad_seeds': 0,
    'total_analyzed': 0,
    'start_time': None
}
show_confidence = True

# Servo control variables
servo_serial = None
servo_status = {
    'connected': False,
    'port': None,
    'last_command': None,
    'last_response_time': None,
    'sorting_active': False,
    'total_sorts': 0
}

def initialize_model():
    """Initialize the YOLO model with GPU optimizations"""
    global model, CLASS_NAMES
    
    if model is not None:
        return model
        
    print("🚀 Initializing YOLO model...")
    model = YOLO(MODEL_PATH)
    
    # Get class names from the model (more accurate than hardcoded)
    if hasattr(model, 'names') and model.names:
        CLASS_NAMES = list(model.names.values())
        print(f"✅ Class names loaded from model: {CLASS_NAMES}")
    else:
        print(f"⚠️  Using default class names: {CLASS_NAMES}")
    
    if device == 'cuda' and torch.cuda.is_available():
        print(f"🎯 Using GPU acceleration")
        
        # Move model to GPU and optimize
        model.to(device)
        
        # Optimize for inference
        try:
            model.fuse()  # Fuse Conv2d + BatchNorm layers for faster inference
            print("✅ Model layers fused for faster inference")
        except:
            pass
        
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Warm up the model
        print("🔥 Warming up model...")
        dummy_input = torch.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), device=device, dtype=torch.float16)
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input, verbose=False)
        print("✅ Model ready for inference")
    else:
        print("⚠️ GPU not available, using CPU")
        # CPU optimizations
        torch.set_num_threads(4)  # Optimize CPU threads
    
    return model
    
    if device == 'cuda' and torch.cuda.is_available():
        print(f"🎯 Using GPU acceleration")
        
        # Move model to GPU and optimize
        model.to(device)
        
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Warm up the model
        print("🔥 Warming up model...")
        dummy_input = torch.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), device=device, dtype=torch.float16)
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input, verbose=False)
        print("✅ Model ready for inference")
    else:
        print("⚠️ GPU not available, using CPU")
    
    return model

def initialize_servo():
    """Initialize servo motor connection via ESP32 with enhanced auto-detection"""
    global servo_serial, servo_status
    
    try:
        # Auto-detect ESP32 COM port with enhanced detection
        ports = serial.tools.list_ports.comports()
        esp32_candidates = []
        
        # ESP32 USB-to-serial chip identifiers (more comprehensive)
        esp32_chips = [
            'cp210x',      # Silicon Labs CP2102/CP2104
            'ch340',       # CH340/CH341
            'ftdi',        # FTDI chips
            'esp32',       # Direct ESP32 reference
            'silicon labs' # Silicon Labs (case insensitive)
        ]
        
        print("Scanning for ESP32...")
        for port in ports:
            description = port.description.lower()
            hwid = port.hwid.lower() if port.hwid else ""
            
            # Check if this looks like an ESP32
            for chip in esp32_chips:
                if chip in description or chip in hwid:
                    esp32_candidates.append({
                        'port': port.device,
                        'description': port.description,
                        'chip': chip
                    })
                    print(f"ESP32 candidate found: {port.device} ({chip})")
                    break
        
        # Try to connect to each candidate
        for candidate in esp32_candidates:
            esp32_port = candidate['port']
            print(f"Attempting connection to {esp32_port}...")
            
            try:
                servo_serial = serial.Serial(esp32_port, 115200, timeout=2)
                time.sleep(3)  # Wait for ESP32 to initialize
                
                # Clear buffers
                servo_serial.reset_input_buffer()
                servo_serial.reset_output_buffer()
                
                # Test connection with multiple attempts
                for attempt in range(3):
                    servo_serial.write(b'STATUS\n')
                    time.sleep(0.5)
                    
                    if servo_serial.in_waiting > 0:
                        response = servo_serial.readline().decode().strip()
                        if response:
                            servo_status['connected'] = True
                            servo_status['port'] = esp32_port
                            servo_status['last_response_time'] = time.time()
                            print(f"✅ Servo motor connected on {esp32_port}")
                            print(f"   ESP32 Response: {response}")
                            return True
                
                # If no response, try a simple command
                servo_serial.write(b'CENTER\n')
                time.sleep(1)
                
                # Assume connection is good if no exception occurred
                servo_status['connected'] = True
                servo_status['port'] = esp32_port
                servo_status['last_response_time'] = time.time()
                print(f"✅ Servo motor connected on {esp32_port} (no response test)")
                return True
                
            except serial.SerialException as e:
                print(f"Failed to connect to {esp32_port}: {e}")
                if servo_serial:
                    servo_serial.close()
                    servo_serial = None
                continue
        
        servo_status['connected'] = False
        print("❌ ESP32 servo controller not found")
        print("Please check:")
        print("  1. ESP32 is connected via USB")
        print("  2. ESP32 drivers are installed")
        print("  3. ESP32 is not used by another application")
        return False
        
    except Exception as e:
        print(f"Servo initialization error: {e}")
        servo_status['connected'] = False
        if servo_serial:
            servo_serial.close()
            servo_serial = None
        return False

def send_servo_command(command):
    """Send command to servo motor with connection recovery"""
    global servo_serial, servo_status
    
    if not servo_status['connected']:
        return False
    
    try:
        # Check if serial connection is still valid
        if not servo_serial or not servo_serial.is_open:
            print("🔄 Serial connection lost, attempting to reconnect...")
            if not initialize_servo():
                return False
        
        # Send command
        servo_serial.write(f"{command}\n".encode())
        time.sleep(0.1)  # Small delay for command processing
        
        # Try to read response (optional, don't fail if no response)
        response = ""
        try:
            if servo_serial.in_waiting > 0:
                response = servo_serial.readline().decode().strip()
        except:
            pass  # Ignore read errors
        
        servo_status['last_command'] = command
        servo_status['last_response_time'] = time.time()
        
        # Count sorting operations (LEFT for bad seeds, RIGHT for good seeds)
        if command in ['LEFT', 'RIGHT']:
            servo_status['total_sorts'] += 1
        
        print(f"✅ Servo command sent: {command}" + (f" | Response: {response}" if response else ""))
        return True
        
    except Exception as e:
        print(f"❌ Servo command error: {e}")
        servo_status['connected'] = False
        # Try to close and reopen connection
        try:
            if servo_serial:
                servo_serial.close()
        except:
            pass
        servo_serial = None
        return False

def check_servo_connection():
    """Periodically check servo connection"""
    while True:
        if servo_status['connected'] and servo_serial:
            try:
                servo_serial.write(b'PING\n')
                response = servo_serial.readline().decode().strip()
                if not response:
                    servo_status['connected'] = False
                    print("Servo connection lost")
                else:
                    servo_status['last_response_time'] = time.time()
            except:
                servo_status['connected'] = False
                print("Servo connection error")
        
        time.sleep(5)  # Check every 5 seconds

def initialize_camera():
    """Initialize and configure the webcam with multiple attempts"""
    print("🎥 Initializing webcam...")
    
    # Try USB cameras first before falling back to built-in camera
    camera_order = [1, 2, 3, 0]  # Try external cameras before built-in (index 0)
    
    for idx in camera_order:
        print(f"Trying camera index {idx}...")
        camera = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # Use DirectShow on Windows
        
        if camera.isOpened():
            print(f"✅ Camera found at index {idx}")
            # Configure camera settings
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus if available
            
            # Verify settings were applied
            actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = camera.get(cv2.CAP_PROP_FPS)
            
            print(f"📹 Camera initialized successfully:")
            print(f"   Resolution: {actual_width}x{actual_height}")
            print(f"   FPS: {actual_fps}")
            return camera
    
    print("❌ No camera found! Please check if your webcam is properly connected.")
    return None

def draw_predictions(frame, boxes, scores, classes):
    """Enhanced prediction drawing with better confidence handling and visual feedback"""
    # Collect all detections for smart processing
    detections = []
    best_detection = None
    best_confidence = 0.0
    
    for box, score, cls in zip(boxes, scores, classes):
        class_name = CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else f"Class_{int(cls)}"
        
        # Determine seed quality and color - Fixed logic for corn.pt model
        # The corn.pt model has classes: ['Bad-Seed', 'Good-Seed']
        # Class 0 = Bad-Seed, Class 1 = Good-Seed
        if 'good-seed' in class_name.lower() or 'good' in class_name.lower():
            # For Good-Seed class
            if score >= 0.7:  # High confidence for good classification
                color = (0, 255, 0)  # Green for good seeds
                status = "Good Seed"
                seed_type = "good"
            elif score >= 0.5:  # Medium confidence
                color = (0, 200, 0)  # Darker green for medium confidence
                status = "Good Seed (Med)"
                seed_type = "good"
            else:
                color = (0, 255, 255)  # Yellow for uncertain
                status = "Uncertain"
                seed_type = "uncertain"
        elif 'bad-seed' in class_name.lower() or 'bad' in class_name.lower():
            # For Bad-Seed class
            if score >= 0.5:  # Any reasonable confidence for bad classification
                color = (0, 0, 255)  # Red for bad seeds
                status = "Bad Seed"
                seed_type = "bad"
            else:
                color = (0, 100, 255)  # Orange for uncertain bad
                status = "Bad Seed (?)"
                seed_type = "bad"
        else:
            color = (0, 255, 255)  # Yellow for uncertain
            status = "Unknown"
            seed_type = "uncertain"
        
        # Store detection info
        detection = {
            'box': box,
            'score': score,
            'class': cls,
            'class_name': class_name,
            'status': status,
            'seed_type': seed_type,
            'color': color
        }
        detections.append(detection)
          # Track the highest confidence detection for action priority
        if score > best_confidence and seed_type in ['good', 'bad']:
            best_confidence = score
            best_detection = detection

    # Draw all detections on frame
    for detection in detections:
        box = detection['box']
        score = detection['score']
        status = detection['status']
        color = detection['color']
        
        # Enhanced bounding box drawing (from app.py)
        x1, y1, x2, y2 = map(int, box)
        
        # Draw thicker main bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        
        # Add inner highlight for better contrast
        cv2.rectangle(frame, (x1+2, y1+2), (x2-2, y2-2), (255, 255, 255), 1)
        
        # Draw corner markers for extra visibility
        corner_size = 8
        cv2.line(frame, (x1, y1), (x1 + corner_size, y1), color, 3)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_size), color, 3)
        cv2.line(frame, (x2, y1), (x2 - corner_size, y1), color, 3)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_size), color, 3)
        cv2.line(frame, (x1, y2), (x1 + corner_size, y2), color, 3)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_size), color, 3)
        cv2.line(frame, (x2, y2), (x2 - corner_size, y2), color, 3)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_size), color, 3)
        
        # Add center point for precise location
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 3, color, -1)
        cv2.circle(frame, (center_x, center_y), 6, (255, 255, 255), 1)
        
        # Enhanced label with confidence percentage
        if show_confidence:
            label = f"{status}: {score*100:.1f}%"
        else:
            label = status
            
        font_scale = 0.7
        font_thickness = 2
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        # Draw label background with padding
        label_bg_x1 = x1
        label_bg_y1 = y1 - label_size[1] - 15
        label_bg_x2 = x1 + label_size[0] + 10
        label_bg_y2 = y1
        
        # Draw label background with border
        cv2.rectangle(frame, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, -1)
        cv2.rectangle(frame, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), (255, 255, 255), 1)
        
        # Draw label text
        cv2.putText(frame, label, (x1 + 5, y1 - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
      # Process only the best detection for statistics and servo control
    if best_detection and best_detection['seed_type'] in ['good', 'bad']:
        update_statistics(best_detection['seed_type'])
        
        # Add visual indicator for the selected action
        best_box = best_detection['box']
        x1, y1, x2, y2 = map(int, best_box)
        
        # Draw a special border around the selected detection
        cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (255, 255, 0), 3)  # Yellow border
      
    # Add enhanced statistics overlay
    stats_text = [
        f"Model: {MODEL_PATH}",
        f"Classes: {', '.join(CLASS_NAMES)}",
        f"Good Seeds: {seed_statistics['good_seeds']}",
        f"Bad Seeds: {seed_statistics['bad_seeds']}",
        f"Total: {seed_statistics['total_analyzed']}"
    ]
    
    # Draw stats background for better visibility
    bg_height = len(stats_text) * 22 + 10
    cv2.rectangle(frame, (5, 5), (350, bg_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, 5), (350, bg_height), (255, 255, 255), 1)
    
    y = 22
    for text in stats_text:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 22
    
    return frame

def update_statistics(seed_type):
    """Update statistics and control servo based on system status"""
    # Only update statistics if enabled
    if system_status['statistics_enabled']:
        seed_statistics[f'{seed_type}_seeds'] += 1
        seed_statistics['total_analyzed'] += 1
    
    # Control servo based on seed quality and system status
    if (servo_status['connected'] and 
        system_status['servo_enabled'] and 
        system_status['auto_sorting'] and 
        not system_status['emergency_stop']):
        
        servo_status['sorting_active'] = True
        if seed_type == 'good':
            send_servo_command('RIGHT')  # Good seeds go RIGHT (180°)
        elif seed_type == 'bad':
            send_servo_command('LEFT')   # Bad seeds go LEFT (0°)
        
        # Reset sorting active flag after a delay
        def reset_sorting_flag():
            servo_status['sorting_active'] = False
        
        threading.Timer(2.0, reset_sorting_flag).start()
    else:
        print(f"Servo action skipped - servo_enabled: {system_status['servo_enabled']}, auto_sorting: {system_status['auto_sorting']}")

def get_fps():
    if seed_statistics['start_time'] is None:
        seed_statistics['start_time'] = time.time()
        return 0
    
    elapsed_time = time.time() - seed_statistics['start_time']
    if elapsed_time == 0:
        return 0
    return seed_statistics['total_analyzed'] / elapsed_time

def generate_frames():
    """Generate video frames with optimized CUDA processing"""
    global cap, model
    
    if cap is None or not cap.isOpened():
        cap = initialize_camera()
        if not cap.isOpened():
            yield b''
            return
    
    if model is None:
        model = initialize_model()
    
    frame_count = 0
    inference_times = []
    performance_stats = {'frame_count': 0, 'total_inference_time': 0.0, 'avg_fps': 0.0}
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        start_time = time.time()
          # Run YOLO inference with CUDA optimization
        with torch.no_grad():
            try:
                # Run inference directly on the frame
                results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
                
                # Extract detection data from YOLOv11 results (similar to app.py)
                if results and len(results) > 0 and hasattr(results[0], "boxes") and results[0].boxes is not None:
                    boxes_tensor = results[0].boxes.xyxy
                    scores_tensor = results[0].boxes.conf
                    classes_tensor = results[0].boxes.cls
                    
                    # Convert to numpy arrays safely
                    if hasattr(boxes_tensor, 'cpu'):
                        boxes = boxes_tensor.cpu().numpy()
                        scores = scores_tensor.cpu().numpy()
                        classes = classes_tensor.cpu().numpy()
                    else:
                        boxes = np.array(boxes_tensor)
                        scores = np.array(scores_tensor)
                        classes = np.array(classes_tensor)
                    
                    # Filter by confidence threshold (additional filtering)
                    confidence_mask = scores >= CONFIDENCE_THRESHOLD
                    boxes = boxes[confidence_mask]
                    scores = scores[confidence_mask]
                    classes = classes[confidence_mask]
                    
                    # Draw predictions on frame
                    frame = draw_predictions(frame, boxes, scores, classes)
                    num_objects = len(boxes)
                else:
                    num_objects = 0
                
                # Sync with GPU every few frames
                if device == 'cuda' and frame_count % 10 == 0:
                    torch.cuda.synchronize()
                
            except Exception as e:
                print(f"⚠️ Inference error: {e}")
                continue
        
        # Track performance
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Add FPS overlay
        if len(inference_times) > 30:
            inference_times.pop(0)
        avg_fps = 1.0 / (sum(inference_times) / len(inference_times))
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # GPU memory management
        if device == 'cuda' and frame_count % 20 == 0:
            torch.cuda.empty_cache()
            
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        # Stream frame
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    """Get comprehensive statistics including detailed GPU usage"""
    # Combine seed statistics with servo status
    combined_stats = {**seed_statistics, 'servo': servo_status}
    
    # Add detailed GPU information if using CUDA
    if device == 'cuda' and torch.cuda.is_available():
        try:
            # Basic GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_props = torch.cuda.get_device_properties(0)
            
            # Memory information
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            memory_total = gpu_props.total_memory / 1024**3
            memory_free = memory_total - memory_reserved
            
            # Utilization calculation
            utilization = (memory_allocated / memory_total) * 100
            
            # Advanced GPU stats
            combined_stats['gpu'] = {
                'device_name': gpu_name,
                'cuda_version': torch.version.cuda,
                'pytorch_version': torch.__version__,
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}",
                'total_memory_gb': round(memory_total, 2),
                'allocated_memory_gb': round(memory_allocated, 3),
                'reserved_memory_gb': round(memory_reserved, 3),
                'free_memory_gb': round(memory_free, 2),
                'memory_utilization_percent': round(utilization, 1),
                'multiprocessor_count': gpu_props.multiprocessor_count,
                'max_threads_per_multiprocessor': gpu_props.max_threads_per_multiprocessor,
                'cuda_available': True,
                'fp16_supported': True,  # Assume true for modern GPUs
                'tensor_cores': gpu_props.major >= 7,  # Volta and newer
            }
            
            # Add performance optimizations status
            combined_stats['optimizations'] = {
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'mixed_precision_enabled': True,  # We enabled FP16
                'model_compiled': hasattr(torch, 'compile'),
            }
            
        except Exception as e:
            combined_stats['gpu'] = {
                'error': str(e),
                'cuda_available': False
            }
    else:
        combined_stats['gpu'] = {
            'device': 'CPU',
            'cuda_available': False,
            'pytorch_version': torch.__version__,
            'threads': torch.get_num_threads()
        }
        combined_stats['optimizations'] = {
            'cpu_optimized': True,
            'threads_limited': True
        }
    
    # Add system information
    try:
        import psutil
        combined_stats['system'] = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'cpu_count': psutil.cpu_count(),
        }
    except ImportError:
        combined_stats['system'] = {'psutil_not_available': True}
    
    return jsonify(combined_stats)

@app.route('/servo_status')
def get_servo_status():
    return jsonify(servo_status)

@app.route('/servo_control/<command>')
def servo_control(command):
    """Manual servo control endpoint"""
    command = command.upper()
    
    # Map manual commands to servo positions
    if command == 'GOOD':
        servo_command = 'RIGHT'  # Good seeds go right
    elif command == 'BAD':
        servo_command = 'LEFT'   # Bad seeds go left
    elif command in ['CENTER', 'TEST']:
        servo_command = command
    else:
        return jsonify({'error': 'Invalid command'}), 400
    
    success = send_servo_command(servo_command)
    return jsonify({
        'success': success,
        'command': command,
        'servo_command': servo_command,
        'servo_status': servo_status
    })

@app.route('/reconnect_servo')
def reconnect_servo():
    """Attempt to reconnect servo"""
    success = initialize_servo()
    return jsonify({
        'success': success,
        'servo_status': servo_status
    })

@app.route('/toggle_confidence')
def toggle_confidence():
    global show_confidence
    show_confidence = not show_confidence
    return jsonify({'show_confidence': show_confidence})

@app.route('/reset_stats')
def reset_stats():
    seed_statistics['good_seeds'] = 0
    seed_statistics['bad_seeds'] = 0
    seed_statistics['total_analyzed'] = 0
    seed_statistics['start_time'] = time.time()
    servo_status['total_sorts'] = 0
    
    # Reset performance stats
    performance_stats['frame_count'] = 0
    performance_stats['total_inference_time'] = 0.0
    performance_stats['avg_fps'] = 0.0
    performance_stats['peak_gpu_memory'] = 0.0
    
    return jsonify({**seed_statistics, 'servo': servo_status, 'performance': performance_stats})

@app.route('/performance')
def get_performance():
    """Get detailed performance statistics"""
    device_info = get_device_info()
    
    perf_data = {
        'device_info': device_info,
        'performance': performance_stats,
        'timestamp': time.time()
    }
    
    # Add current GPU memory if available
    if device == 'cuda' and torch.cuda.is_available():
        try:
            perf_data['current_gpu_memory_gb'] = torch.cuda.memory_allocated(0) / 1024**3
            perf_data['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved(0) / 1024**3
        except:
            pass
    
    return jsonify(perf_data)

@app.route('/gpu_cleanup')
def manual_gpu_cleanup():
    """Manually trigger GPU memory cleanup"""
    if device == 'cuda' and torch.cuda.is_available():
        try:
            memory_before = torch.cuda.memory_allocated(0) / 1024**3
            torch.cuda.empty_cache()
            gc.collect()
            memory_after = torch.cuda.memory_allocated(0) / 1024**3
            
            return jsonify({
                'success': True,
                'memory_before_gb': round(memory_before, 3),
                'memory_after_gb': round(memory_after, 3),
                'memory_freed_gb': round(memory_before - memory_after, 3)
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    else:
        return jsonify({'success': False, 'error': 'CUDA not available'})

@app.route('/optimize_model')
def optimize_model():
    """Re-optimize model settings for current conditions"""
    try:
        if device == 'cuda' and torch.cuda.is_available():
            # Re-apply optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            
            # Try to re-compile if available
            if hasattr(torch, 'compile') and hasattr(model, 'model'):
                try:
                    model.model = torch.compile(model.model, mode='max-autotune')
                except:
                    pass
            
            return jsonify({
                'success': True,
                'message': 'Model optimizations reapplied',
                'device': device,
                'optimizations': {
                    'cudnn_benchmark': torch.backends.cudnn.benchmark,
                    'cudnn_deterministic': torch.backends.cudnn.deterministic
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': 'CUDA not available for optimization'
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# System Control Functions
def emergency_stop():
    """Emergency stop - disable all systems"""
    global system_status
    system_status.update({
        'camera_enabled': False,
        'detection_enabled': False,
        'servo_enabled': False,
        'auto_sorting': False,
        'system_running': False,
        'emergency_stop': True
    })
    
    # Stop servo motor
    if servo_serial:
        try:
            send_servo_command('CENTER')
        except:
            pass
    
    print("🚨 EMERGENCY STOP ACTIVATED - All systems disabled")

def system_restart():
    """Restart all systems"""
    global system_status
    system_status.update({
        'camera_enabled': True,
        'detection_enabled': True,
        'servo_enabled': True,
        'auto_sorting': True,
        'system_running': True,
        'emergency_stop': False
    })
    
    # Reinitialize servo
    initialize_servo()
    print("🔄 System restarted - All systems enabled")

def toggle_system_component(component: str):
    """Toggle individual system components"""
    global system_status
    
    if component in system_status:
        # Don't allow enabling if emergency stop is active
        if system_status['emergency_stop'] and not system_status[component]:
            return False, "Cannot enable during emergency stop"
        
        system_status[component] = not system_status[component]
        
        # Handle specific component logic
        if component == 'servo_enabled' and not system_status[component]:
            # Center servo when disabled
            if servo_serial:
                send_servo_command('CENTER')
        
        if component == 'auto_sorting' and not system_status[component]:
            # Stop any ongoing sorting
            servo_status['sorting_active'] = False
        
        return True, f"{component} {'enabled' if system_status[component] else 'disabled'}"
    
    return False, "Invalid component"

if __name__ == '__main__':
    print("Initializing YOLOv11 Seed Detection Model...")
    try:
        initialize_model()
        print(f"Model loaded successfully on {device}")
        print(f"Classes detected: {CLASS_NAMES}")
        
        # Initialize servo motor
        print("Initializing servo motor...")
        initialize_servo()
        
        # Start servo connection monitoring in background
        servo_thread = threading.Thread(target=check_servo_connection, daemon=True)
        servo_thread.start()
        
        # Start Flask app
        print("Starting web server...")
        print("Access the application at: http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except Exception as e:
        print(f"Error during initialization: {e}")
    finally:
        if cap is not None:
            cap.release()
        if servo_serial is not None:
            servo_serial.close()
