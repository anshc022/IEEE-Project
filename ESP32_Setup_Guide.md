# ESP32 Servo Control Setup Guide

## Hardware Requirements
- ESP32 development board
- Servo motor (SG90 or similar)
- Jumper wires
- Breadboard (optional)
- Power supply (if needed for servo)

## Hardware Connections

### ESP32 to Servo Motor:
```
ESP32 Pin    Servo Wire    Description
---------    ----------    -----------
GPIO 18      Signal        Orange/Yellow wire
5V or 3.3V   VCC          Red wire (power)
GND          GND          Brown/Black wire
```

**Important Notes:**
- Small servos (SG90) can usually be powered from ESP32's 3.3V pin
- Larger servos may require external 5V power supply
- Always connect GND between ESP32 and servo
- Signal pin (GPIO 18) can be changed in the Arduino code if needed

## Software Setup

### 1. Arduino IDE Setup for ESP32
1. Install Arduino IDE if not already installed
2. Add ESP32 board support:
   - Go to File → Preferences
   - Add this URL to "Additional Board Manager URLs":
     ```
     https://dl.espressif.com/dl/package_esp32_index.json
     ```
   - Go to Tools → Board → Boards Manager
   - Search for "ESP32" and install "ESP32 by Espressif Systems"

### 2. Install ESP32Servo Library
1. Go to Tools → Manage Libraries
2. Search for "ESP32Servo"
3. Install "ESP32Servo by Kevin Harrington"

### 3. Upload Arduino Code
1. Open `esp32_servo_control.ino` in Arduino IDE
2. Select your ESP32 board: Tools → Board → ESP32 Dev Module
3. Select correct COM port: Tools → Port → (your ESP32 port)
4. Upload the code

### 4. Install Python Dependencies
```bash
pip install pyserial
```

## Testing the System

### 1. Test ESP32 Servo Control Manually
1. Open Serial Monitor in Arduino IDE (Tools → Serial Monitor)
2. Set baud rate to 115200
3. Send these commands to test:
   - `CENTER` - Move to center position
   - `LEFT` - Move 90 degrees left (bad seeds)
   - `RIGHT` - Move 90 degrees right (good seeds)
   - `TEST` - Run automatic test sequence

### 2. Test Python Integration
1. Make sure ESP32 is connected and servo responds to manual commands
2. Run the Python seed detection application:
   ```bash
   python app.py
   ```
3. The system should:
   - Auto-detect ESP32 COM port
   - Initialize servo to center position
   - Move servo left when bad seeds are detected
   - Move servo right when good seeds are detected

## Troubleshooting

### ESP32 Not Detected
- Check USB cable connection
- Install ESP32 drivers if needed
- Verify COM port in Device Manager (Windows)
- Try different USB port

### Servo Not Moving
- Check power connections (servo needs adequate power)
- Verify signal wire connection to GPIO 18
- Check if servo is compatible (most standard servos work)
- Test with manual Serial Monitor commands first

### Python Connection Issues
- Ensure no other software is using the COM port
- Close Arduino Serial Monitor before running Python app
- Check if pyserial is installed: `pip list | grep serial`
- Verify ESP32 port auto-detection in Python console output

### Servo Movement Issues
- Adjust delay in Arduino code for smoother/faster movement
- Check servo power supply (insufficient power causes erratic movement)
- Verify servo positions (0°, 90°, 180°) are appropriate for your setup

## Customization Options

### Servo Positions
Edit these values in the Arduino code:
```cpp
const int CENTER_POS = 90;   // Center position
const int LEFT_POS = 0;      // Bad seeds position
const int RIGHT_POS = 180;   // Good seeds position
```

### Servo Speed
Adjust movement speed in the Arduino code:
```cpp
delay(15);  // Lower value = faster movement
```

### GPIO Pin
Change servo pin in Arduino code:
```cpp
const int SERVO_PIN = 18;  // Change to desired GPIO pin
```

### Python Timing
Adjust servo activation timing in Python (`app.py`):
```python
time.sleep(1)  # Time before returning to center
```

## System Operation

1. **Startup**: ESP32 initializes servo to center position
2. **Detection**: Python app detects seeds using camera and AI model
3. **Sorting**: 
   - Good seeds → Servo moves right (180°)
   - Bad seeds → Servo moves left (0°)
   - Returns to center after 1 second
4. **Feedback**: Both systems provide console output for monitoring

## Safety Notes
- Ensure servo has adequate power supply
- Check servo movement range doesn't cause mechanical interference
- Monitor servo temperature during extended operation
- Use appropriate gauge wires for servo current requirements
