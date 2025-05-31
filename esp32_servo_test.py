#!/usr/bin/env python3
"""
ESP32 Servo Control Test Script
Auto-detects ESP32 port and tests servo movement
"""

import serial
import serial.tools.list_ports
import time
import sys

def find_esp32_port():
    """Auto-detect ESP32 COM port"""
    print("🔍 Scanning for ESP32...")
    
    ports = serial.tools.list_ports.comports()
    esp32_candidates = []
    
    # ESP32 USB-to-serial chip identifiers
    esp32_chips = [
        'cp210x',      # Silicon Labs CP2102/CP2104
        'ch340',       # CH340/CH341
        'ftdi',        # FTDI chips
        'esp32',       # Direct ESP32 reference
        'silicon labs' # Silicon Labs (case insensitive)
    ]
    
    for port in ports:
        description = port.description.lower()
        hwid = port.hwid.lower() if port.hwid else ""
        
        print(f"📍 Found: {port.device} - {port.description}")
        
        # Check if this looks like an ESP32
        for chip in esp32_chips:
            if chip in description or chip in hwid:
                esp32_candidates.append({
                    'port': port.device,
                    'description': port.description,
                    'chip': chip
                })
                print(f"   ✅ ESP32 candidate detected ({chip})")
                break
    
    if not esp32_candidates:
        print("❌ No ESP32 found. Please check:")
        print("   1. ESP32 is connected via USB")
        print("   2. ESP32 drivers are installed")
        print("   3. ESP32 is not used by another application")
        return None
    
    # Use the first candidate (or could implement selection logic)
    selected = esp32_candidates[0]
    print(f"🎯 Selected: {selected['port']} ({selected['chip']})")
    return selected['port']

def test_servo_connection(port):
    """Test connection to ESP32 servo controller"""
    print(f"\n🔗 Connecting to ESP32 on {port}...")
    
    try:
        # Open serial connection
        ser = serial.Serial(port, 115200, timeout=2)
        time.sleep(3)  # Wait for ESP32 to initialize
        
        print("✅ Serial connection established")
        
        # Clear any existing data
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        # Test connection with STATUS command
        print("📡 Testing communication...")
        ser.write(b'STATUS\n')
        time.sleep(0.5)
        
        response = ser.readline().decode().strip()
        if response:
            print(f"📨 ESP32 Response: {response}")
        else:
            print("📨 No response received, but connection established")
        
        return ser
        
    except serial.SerialException as e:
        print(f"❌ Connection failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def send_servo_command(ser, command, description=""):
    """Send command to servo and wait for response"""
    try:
        print(f"\n🎯 Sending: {command} {description}")
        ser.write(f"{command}\n".encode())
        time.sleep(0.1)
        
        # Read response with timeout
        start_time = time.time()
        response_lines = []
        
        while time.time() - start_time < 1.0:  # 1 second timeout
            if ser.in_waiting > 0:
                line = ser.readline().decode().strip()
                if line:
                    response_lines.append(line)
                    print(f"   📨 {line}")
        
        return response_lines
        
    except Exception as e:
        print(f"   ❌ Command failed: {e}")
        return []

def run_servo_test_sequence(ser):
    """Run complete servo test sequence"""
    print("\n" + "="*50)
    print("🎮 STARTING SERVO TEST SEQUENCE")
    print("="*50)
    
    # Test sequence
    test_commands = [
        ("CENTER", "- Move to center position (90°)"),
        ("LEFT", "- Move LEFT for BAD seeds (0°)"),
        ("CENTER", "- Return to center"),
        ("RIGHT", "- Move RIGHT for GOOD seeds (180°)"),
        ("CENTER", "- Return to center"),
        ("LEFT", "- Test BAD seed sorting"),
        ("RIGHT", "- Test GOOD seed sorting"),
        ("CENTER", "- Final center position")
    ]
    
    for i, (command, description) in enumerate(test_commands, 1):
        print(f"\n🔄 Step {i}/8: {command} {description}")
        send_servo_command(ser, command, description)
        time.sleep(2)  # Wait between movements
    
    print(f"\n✅ Test sequence completed!")

def interactive_control(ser):
    """Interactive servo control"""
    print("\n" + "="*50)
    print("🎮 INTERACTIVE SERVO CONTROL")
    print("="*50)
    print("Commands:")
    print("  LEFT   - Move servo left (bad seeds)")
    print("  RIGHT  - Move servo right (good seeds)")
    print("  CENTER - Move servo to center")
    print("  TEST   - Run test sequence")
    print("  QUIT   - Exit")
    print("="*50)
    
    while True:
        try:
            command = input("\n🎯 Enter command: ").strip().upper()
            
            if command == 'QUIT':
                break
            elif command in ['LEFT', 'RIGHT', 'CENTER']:
                send_servo_command(ser, command)
            elif command == 'TEST':
                run_servo_test_sequence(ser)
            elif command == '':
                continue
            else:
                print("❌ Invalid command. Use: LEFT, RIGHT, CENTER, TEST, or QUIT")
                
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    print("🤖 ESP32 SERVO CONTROL TEST")
    print("="*50)
    
    # Auto-detect ESP32 port
    port = find_esp32_port()
    if not port:
        sys.exit(1)
    
    # Test connection
    ser = test_servo_connection(port)
    if not ser:
        print("❌ Failed to connect to ESP32")
        sys.exit(1)
    
    try:
        # Run automatic test sequence
        run_servo_test_sequence(ser)
        
        # Interactive control
        interactive_control(ser)
        
    finally:
        # Clean up
        if ser and ser.is_open:
            print("\n🔌 Closing serial connection...")
            send_servo_command(ser, "CENTER")  # Return to center before closing
            time.sleep(1)
            ser.close()
            print("✅ Connection closed")

if __name__ == "__main__":
    main()
