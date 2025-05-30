import cv2
import torch
import numpy as np
import time
import serial
import serial.tools.list_ports
from pathlib import Path
import requests
import json
from datetime import datetime
import threading
import queue

# Configuration
MODEL_PATH = "corn11.pt"  # Your YOLOv11 model
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.7
INPUT_SIZE = 640  # YOLOv11 works better with 640x640
CLASS_NAMES = []

# LLM Configuration
LLM_ENABLED = True
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2:3b"  # Change to your preferred model
LLM_ANALYSIS_INTERVAL = 5  # Seconds between LLM analysis
llm_queue = queue.Queue()
llm_responses = queue.Queue()

# ESP32 Servo Configuration
ESP32_PORT = None  # Will be auto-detected
BAUD_RATE = 115200
servo_serial = None

# Servo Commands
SERVO_LEFT = "LEFT"    # 90 degrees left for bad seeds
SERVO_RIGHT = "RIGHT"  # 90 degrees right for good seeds
SERVO_CENTER = "CENTER"  # Center position

# Seed Analysis Configuration
seed_analysis_mode = False
servo_control_enabled = False
llm_analysis_mode = False
seed_statistics = {
    'good_seeds': 0,
    'bad_seeds': 0,
    'total_analyzed': 0
}

# LLM Analysis Data
llm_analysis_data = {
    'last_analysis': "",
    'session_summary': "",
    'quality_insights': "",
    'recommendations': ""
}

def check_ollama_connection():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        return response.status_code == 200
    except:
        return False

def query_llm(prompt, context=""):
    """Send query to local LLM via Ollama"""
    if not LLM_ENABLED:
        return "LLM disabled"
    
    try:
        full_prompt = f"Context: {context}\n\nQuery: {prompt}\n\nPlease provide a concise, helpful response:"
        
        payload = {
            "model": LLM_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 200
            }
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json().get('response', 'No response')
        else:
            return f"LLM Error: {response.status_code}"
    except Exception as e:
        return f"LLM Connection Error: {str(e)}"

def llm_worker():
    """Background worker for LLM processing"""
    while True:
        try:
            if not llm_queue.empty():
                analysis_data = llm_queue.get()
                
                # Generate analysis based on current data
                context = f"""
                Seed Detection Session Data:
                - Good seeds detected: {analysis_data['good_seeds']}
                - Bad seeds detected: {analysis_data['bad_seeds']}
                - Total analyzed: {analysis_data['total_analyzed']}
                - Detection accuracy: {analysis_data.get('accuracy', 'N/A')}
                - Session duration: {analysis_data.get('duration', 'N/A')} minutes
                """
                
                # Different types of analysis
                if analysis_data['analysis_type'] == 'quality_assessment':
                    prompt = "Analyze the seed quality detection results. What insights can you provide about the seed batch quality?"
                elif analysis_data['analysis_type'] == 'performance_review':
                    prompt = "Review the detection system performance. Any observations or recommendations?"
                elif analysis_data['analysis_type'] == 'session_summary':
                    prompt = "Summarize this seed detection session in 2-3 sentences."
                else:
                    prompt = "Provide a brief analysis of the current seed detection results."
                
                response = query_llm(prompt, context)
                llm_responses.put({
                    'type': analysis_data['analysis_type'],
                    'response': response,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                
            time.sleep(1)
        except Exception as e:
            print(f"LLM Worker Error: {e}")
            time.sleep(5)

def generate_llm_analysis(analysis_type="general"):
    """Queue LLM analysis request"""
    if not LLM_ENABLED or not llm_analysis_mode:
        return
    
    session_duration = (time.time() - start_time) / 60  # minutes
    accuracy = (seed_statistics['good_seeds'] + seed_statistics['bad_seeds']) / max(seed_statistics['total_analyzed'], 1) * 100
    
    analysis_data = {
        'analysis_type': analysis_type,
        'good_seeds': seed_statistics['good_seeds'],
        'bad_seeds': seed_statistics['bad_seeds'],
        'total_analyzed': seed_statistics['total_analyzed'],
        'accuracy': f"{accuracy:.1f}%",
        'duration': f"{session_duration:.1f}"
    }
    
    llm_queue.put(analysis_data)

def process_llm_responses():
    """Process LLM responses and update display data"""
    while not llm_responses.empty():
        response_data = llm_responses.get()
        
        if response_data['type'] == 'quality_assessment':
            llm_analysis_data['quality_insights'] = response_data['response']
        elif response_data['type'] == 'performance_review':
            llm_analysis_data['recommendations'] = response_data['response']
        elif response_data['type'] == 'session_summary':
            llm_analysis_data['session_summary'] = response_data['response']
        else:
            llm_analysis_data['last_analysis'] = response_data['response']
        
        print(f"🤖 LLM {response_data['type']}: {response_data['response']}")

def find_esp32_port():
    """Auto-detect ESP32 COM port"""
    global ESP32_PORT
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if 'USB' in port.description or 'Serial' in port.description or 'CH340' in port.description or 'CP210' in port.description:
            ESP32_PORT = port.device
            return ESP32_PORT
    return None

def init_servo_connection():
    """Initialize serial connection to ESP32"""
    global servo_serial, servo_control_enabled
    try:
        if not ESP32_PORT:
            find_esp32_port()
        
        if ESP32_PORT:
            print(f"Attempting to connect to ESP32 on {ESP32_PORT}...")
            servo_serial = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)  # Wait for ESP32 to initialize
            servo_control_enabled = True
            print(f"ESP32 connected successfully on {ESP32_PORT}")
            # Send center command to initialize servo
            send_servo_command(SERVO_CENTER)
            return True
        else:
            print("ESP32 not found. Servo control disabled.")
            servo_control_enabled = False
            return False
    except Exception as e:
        print(f"Failed to connect to ESP32: {e}")
        servo_control_enabled = False
        return False

def send_servo_command(command):
    """Send command to ESP32 servo"""
    global servo_serial
    if servo_control_enabled and servo_serial:
        try:
            command_str = f"{command}\n"
            servo_serial.write(command_str.encode())
            print(f"Servo command sent: {command}")
        except Exception as e:
            print(f"Error sending servo command: {e}")

def control_servo_for_seed(seed_type):
    """Control servo based on seed quality"""
    if servo_control_enabled:
        if seed_type == "good":
            send_servo_command(SERVO_RIGHT)  # 90 degrees right for good seeds
            print(f"🌱 Good seed detected - Servo moved RIGHT")
        elif seed_type == "bad":
            send_servo_command(SERVO_LEFT)   # 90 degrees left for bad seeds
            print(f"🚫 Bad seed detected - Servo moved LEFT")

def draw_predictions(frame, boxes, scores, classes, class_names, show_confidence=True):
    for box, score, cls in zip(boxes, scores, classes):
        class_name = class_names[int(cls)] if int(cls) < len(class_names) else f"Class_{int(cls)}"
        
        # Determine seed quality and color
        if 'good' in class_name.lower() and score >= 0.7:
            color = (0, 255, 0)  # Green for good seeds
            status = "Good Seed"
            seed_type = "good"
        elif 'bad' in class_name.lower():
            color = (0, 0, 255)  # Red for bad seeds
            status = "Bad Seed"
            seed_type = "bad"
        else:
            color = (0, 255, 255)  # Yellow for uncertain
            status = "Uncertain"
            seed_type = "uncertain"
        
        # Control servo for detected seeds
        if servo_control_enabled and seed_type in ["good", "bad"]:
            control_servo_for_seed(seed_type)
        
        # Update statistics only in analysis mode
        if seed_analysis_mode:
            if seed_type == "good":
                seed_statistics['good_seeds'] += 1
            elif seed_type == "bad":
                seed_statistics['bad_seeds'] += 1
            seed_statistics['total_analyzed'] += 1

        # Draw enhanced bounding box for better visibility
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
        
        # Enhanced label with larger font and better background
        label = f"{status}: {score:.2f}" if show_confidence else status
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

    # Add enhanced analysis overlay with larger text
    if seed_analysis_mode:
        stats_text = [
            f"Good Seeds: {seed_statistics['good_seeds']}",
            f"Bad Seeds: {seed_statistics['bad_seeds']}",
            f"Total Analyzed: {seed_statistics['total_analyzed']}"
        ]
        
        # Draw stats background for better visibility
        bg_height = len(stats_text) * 25 + 10
        cv2.rectangle(frame, (5, frame.shape[0] - bg_height - 5), 
                     (300, frame.shape[0] - 5), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, frame.shape[0] - bg_height - 5), 
                     (300, frame.shape[0] - 5), (255, 255, 255), 1)
        
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (10, frame.shape[0] - bg_height + 20 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add LLM analysis overlay
    if llm_analysis_mode and llm_analysis_data['last_analysis']:
        llm_text = f"AI: {llm_analysis_data['last_analysis'][:60]}..."
        cv2.putText(frame, llm_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 255), 2)
    
    return frame

print("Initializing YOLOv11 + LLM Seed Detection System...")
print(f"Model path: {MODEL_PATH}")

# Check LLM availability
if LLM_ENABLED:
    print("Checking LLM connection...")
    if check_ollama_connection():
        print(f"✅ Ollama connected - Using model: {LLM_MODEL}")
        # Start LLM worker thread
        llm_thread = threading.Thread(target=llm_worker, daemon=True)
        llm_thread.start()
    else:
        print("⚠️  Ollama not detected - LLM features disabled")
        print("To enable LLM: Install Ollama and run 'ollama pull llama3.2:3b'")
        LLM_ENABLED = False

# Check model file exists
if not Path(MODEL_PATH).exists():
    print(f"Error: Model file '{MODEL_PATH}' not found!")
    exit(1)

try:
    from ultralytics import YOLO
    
    # Load YOLOv11 model with your custom weights
    model = YOLO(MODEL_PATH)
    
    # Detect model version
    model_info = str(model.model)
    if 'v11' in model_info.lower() or 'yolo11' in model_info.lower():
        model_version = "YOLOv11"
    elif 'v10' in model_info.lower():
        model_version = "YOLOv10"
    elif 'v9' in model_info.lower():
        model_version = "YOLOv9"
    elif 'v8' in model_info.lower():
        model_version = "YOLOv8"
    else:
        model_version = "YOLO (Unknown version)"
    
    print(f"Using Ultralytics {model_version} for Seed Detection")
    print(f"Model loaded from: {MODEL_PATH}")
    
    # Get class names from the model
    if not CLASS_NAMES and hasattr(model, 'names'):
        CLASS_NAMES = list(model.names.values())
    
    print(f"Model classes detected: {CLASS_NAMES}")
    print(f"Number of classes: {len(CLASS_NAMES)}")
    
    # Set device (GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # Warm up the model on GPU
        dummy_img = torch.zeros((1, 3, INPUT_SIZE, INPUT_SIZE)).to(device)
        with torch.no_grad():
            _ = model.predict(dummy_img, verbose=False)
        print("Model warmed up on GPU")
    else:
        print("Using CPU")
        
    print("Model initialization complete!")

except ImportError as e:
    print(f"Error: Ultralytics package not found. Please install with: pip install ultralytics>=8.0.196")
    print(f"Import error: {e}")
    exit(1)
except Exception as e:
    print(f"Error loading YOLOv11 model: {e}")
    print("Make sure your model file is compatible with YOLOv11")
    print("You can convert older YOLO models using: yolo export model=your_model.pt")
    exit(1)

# Initialize camera
print("Initializing camera for seed detection...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    for i in range(1, 4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"Using camera index {i}")
            break
    else:
        print("Error: No camera found!")
        exit(1)

print("Camera initialized successfully")

# Initialize ESP32 servo control
print("\nInitializing ESP32 servo control...")
servo_init_success = init_servo_connection()
if servo_init_success:
    print("✅ ESP32 servo control enabled")
else:
    print("⚠️  ESP32 servo control disabled - detection will continue without sorting")

print("\n=== YOLOv11 + LLM Seed Detection Controls ===")
print("Press 'q' to quit")
print("Press 'p' to pause/resume")
print("Press 's' to save detected seed frame")
print("Press 'c' to show/hide confidence scores")
print("Press 'a' to toggle seed analysis mode")
if LLM_ENABLED:
    print("Press 'l' to toggle LLM analysis mode")
    print("Press 'r' to generate LLM report")
print("=================\n")

frame_count = 0
start_time = time.time()
fps = 0
paused = False
show_confidence = True
save_counter = 0
total_inference_time = 0
num_objects = 0
frame = None
last_llm_analysis = time.time()

# Auto-start detection
print("Auto-starting YOLOv11 + LLM seed detection in 3 seconds...")
time.sleep(1)
print("3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)

try:
    print("Starting real-time YOLOv11 + LLM seed detection... (Auto-started)")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
                
            inference_start = time.time()
            
            # YOLOv11 inference
            with torch.no_grad():
                results = model.predict(
                    frame, 
                    imgsz=INPUT_SIZE, 
                    conf=CONFIDENCE_THRESHOLD, 
                    iou=IOU_THRESHOLD, 
                    verbose=False
                )
                
                if results and len(results) > 0 and hasattr(results[0], "boxes") and results[0].boxes is not None:
                    # Extract detection data from YOLOv11 results
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
                    
                    frame = draw_predictions(frame, boxes, scores, classes, CLASS_NAMES, show_confidence)
                    num_objects = len(boxes)
                else:
                    num_objects = 0

            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            frame_count += 1

            # Periodic LLM analysis
            if LLM_ENABLED and llm_analysis_mode and (time.time() - last_llm_analysis) > LLM_ANALYSIS_INTERVAL:
                generate_llm_analysis("general")
                last_llm_analysis = time.time()

            if frame_count % 15 == 0:
                end_time = time.time()
                fps = 15 / (end_time - start_time)
                start_time = end_time
                avg_inference = total_inference_time / frame_count if frame_count > 0 else 0
                print(f"FPS: {fps:.1f} | Inference: {avg_inference*1000:.1f}ms | Seeds: {num_objects}")
        elif frame is None:
            continue

        # Process LLM responses
        if LLM_ENABLED:
            process_llm_responses()

        # Status overlay
        status_text = "ANALYZING" if seed_analysis_mode else "DETECTING"
        if LLM_ENABLED and llm_analysis_mode:
            status_text += " + AI"
        overlay_color = (0, 255, 0) if not paused else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f} | {status_text}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, overlay_color, 2)

        if not paused:
            cv2.putText(frame, f"Seeds: {num_objects}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Show controls
        controls_text = "q:quit | p:pause | s:save | c:confidence | a:analysis"
        if LLM_ENABLED:
            controls_text += " | l:LLM | r:report"
        cv2.putText(frame, controls_text, 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow('YOLOv11 + LLM Seed Quality Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print(f"Detection {'paused' if paused else 'resumed'}")
        elif key == ord('s'):
            save_filename = f"seed_detection_{save_counter:04d}.jpg"
            cv2.imwrite(save_filename, frame)
            print(f"Frame saved as {save_filename}")
            save_counter += 1
        elif key == ord('c'):
            show_confidence = not show_confidence
            print(f"Confidence scores {'shown' if show_confidence else 'hidden'}")
        elif key == ord('a'):
            seed_analysis_mode = not seed_analysis_mode
            print(f"Seed analysis mode {'enabled' if seed_analysis_mode else 'disabled'}")
            if seed_analysis_mode:
                print("\nSeed Analysis Statistics Reset")
                seed_statistics['good_seeds'] = 0
                seed_statistics['bad_seeds'] = 0
                seed_statistics['total_analyzed'] = 0
        elif key == ord('l') and LLM_ENABLED:
            llm_analysis_mode = not llm_analysis_mode
            print(f"LLM analysis mode {'enabled' if llm_analysis_mode else 'disabled'}")
        elif key == ord('r') and LLM_ENABLED:
            print("🤖 Generating LLM report...")
            generate_llm_analysis("session_summary")
            generate_llm_analysis("quality_assessment")
            generate_llm_analysis("performance_review")

except KeyboardInterrupt:
    print("\nStopping YOLOv11 + LLM seed detection...")
except Exception as e:
    print(f"Error during detection: {e}")
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    cv2.destroyAllWindows()
    if servo_serial:
        servo_serial.close()
    if frame_count > 0:
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        avg_inference = total_inference_time / frame_count
        print("\n=== YOLOv11 + LLM Session Statistics ===")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average inference time: {avg_inference*1000:.1f}ms")
        print(f"Total seeds analyzed: {seed_statistics['total_analyzed']}")
        print(f"Good seeds detected: {seed_statistics['good_seeds']}")
        print(f"Bad seeds detected: {seed_statistics['bad_seeds']}")
        print(f"Total session time: {total_time:.1f}s")
        
        # Final LLM summary
        if LLM_ENABLED and seed_statistics['total_analyzed'] > 0:
            print("\n🤖 Final AI Analysis:")
            print(f"Quality Insights: {llm_analysis_data.get('quality_insights', 'N/A')}")
            print(f"Recommendations: {llm_analysis_data.get('recommendations', 'N/A')}")
            print(f"Session Summary: {llm_analysis_data.get('session_summary', 'N/A')}")
