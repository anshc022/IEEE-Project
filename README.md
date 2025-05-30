# Intelligent Seed Quality Detection and Sorting System

## Overview
This system combines computer vision AI with automated sorting to detect and classify seed quality in real-time. Good seeds are sorted to the right, bad seeds to the left using an ESP32-controlled servo motor.

## System Features
- 🤖 **AI-Powered Detection**: YOLOv11 model for accurate seed classification
- 🎯 **Auto-Start**: 3-second countdown eliminates manual intervention
- 🔄 **Automated Sorting**: ESP32 servo control for physical seed separation
- 📊 **Real-time Statistics**: Live tracking of good/bad seed counts
- 🎨 **Enhanced Visualization**: Thick bounding boxes, corner markers, center points
- ⚙️ **Smart Port Detection**: Auto-finds camera and ESP32 connections
- 🚀 **YOLOv11 Support**: Latest YOLO architecture for improved performance

## Hardware Requirements
- Computer with USB camera
- ESP32 development board
- Servo motor (SG90 or similar)
- Jumper wires
- Breadboard (optional)
- USB cables

## Software Requirements
- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO (>=8.0.196) - YOLOv11 support
- PySerial
- NumPy

## Installation
```bash
pip install ultralytics>=8.0.196 torch torchvision opencv-python numpy pyserial
```

## Quick Start

### 1. Verify System Setup
```bash
verify_setup.bat
```

### 2. Hardware Setup (if using servo)
```
ESP32 Pin    Servo Wire    Description
---------    ----------    -----------
GPIO 18      Signal        Orange/Yellow wire
5V or 3.3V   VCC          Red wire (power)
GND          GND          Brown/Black wire
```

### 3. Upload ESP32 Code (if using servo)
1. Open `esp32_servo_control.ino` in Arduino IDE
2. Select ESP32 board and COM port
3. Upload the code

### 4. Test Servo (optional)
```bash
python test_servo.py
```

### 5. Run Seed Detection
```bash
run_seed_detection.bat
```
or
```bash
python app.py
```

## System Operation

### Startup Sequence
1. System initializes camera and AI model
2. Auto-detects and connects to ESP32 (if available)
3. 3-second countdown begins
4. Detection starts automatically

### Detection Process
1. Camera captures real-time video
2. AI model analyzes each frame for seeds
3. Seeds are classified as good or bad
4. Servo moves accordingly:
   - **Good seeds** → Servo turns **RIGHT** (180°)
   - **Bad seeds** → Servo turns **LEFT** (0°)
   - **Auto-return** → Center position (90°) after 1.5 seconds

### Controls During Operation
- **'q'** - Quit application
- **'p'** - Pause/resume detection
- **'s'** - Save current frame
- **'a'** - Toggle analysis mode (statistics)
- **'c'** - Show/hide confidence scores

## File Structure
```
IEEE-Project/
├── app.py                     # Main application
├── corn11.pt                  # YOLOv8 model file
├── esp32_servo_control.ino    # Arduino code for ESP32
├── test_servo.py              # Servo testing script
├── run_seed_detection.bat     # Windows startup script
├── verify_setup.bat           # System verification script
├── ESP32_Setup_Guide.md       # Detailed ESP32 setup
└── README.md                  # This file
```

## Troubleshooting

### Camera Issues
- Ensure camera is connected and not used by other software
- Try different camera indices (0, 1, 2...)
- Check camera permissions in Windows settings

### ESP32 Connection Issues
- Verify USB cable connection
- Install ESP32 drivers if needed
- Close Arduino Serial Monitor before running Python
- Check COM port in Device Manager

### Servo Not Moving
- Verify power connections (servo needs adequate power)
- Check signal wire connection to GPIO 18
- Test with manual commands in Arduino Serial Monitor
- Ensure servo is compatible (most standard servos work)

### Model Loading Issues
- Ensure `corn11.pt` model file is present
- Check PyTorch and Ultralytics installation
- Verify sufficient system memory

## Configuration Options

### Servo Positions (Arduino)
```cpp
const int CENTER_POS = 90;   // Center position
const int LEFT_POS = 0;      // Bad seeds position  
const int RIGHT_POS = 180;   // Good seeds position
```

### Detection Thresholds (Python)
```python
CONFIDENCE_THRESHOLD = 0.5   # Minimum confidence for detection
IOU_THRESHOLD = 0.7          # Overlap threshold for filtering
```

### Auto-Return Timing (Arduino)
```cpp
const int RETURN_DELAY = 1500;  // Milliseconds before returning to center
```

## System Statistics
The system tracks and displays:
- Total seeds analyzed
- Good seeds detected
- Bad seeds detected
- Real-time FPS
- Average inference time
- Session duration

## Safety Notes
- Ensure servo movement doesn't cause mechanical interference
- Monitor servo temperature during extended operation
- Use appropriate power supply for servo requirements
- Verify all connections before powering on

## Support
For issues or questions:
1. Check troubleshooting section above
2. Verify setup with `verify_setup.bat`
3. Test components individually (camera, ESP32, servo)
4. Review ESP32_Setup_Guide.md for detailed instructions

## System Specifications
- **Detection Speed**: ~15 FPS
- **Model**: YOLOv8 optimized for seed detection
- **Servo Response**: <2 seconds total cycle time
- **Accuracy**: Depends on trained model quality
- **Compatibility**: Windows, cross-platform capable
