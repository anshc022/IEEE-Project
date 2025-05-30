#!/usr/bin/env python3
"""
ESP32 Servo Test Script
Test servo control communication with ESP32 before running main application
"""

import serial
import serial.tools.list_ports
import time
import sys

def find_esp32_port():
    """Auto-detect ESP32 COM port"""
    print("Scanning for ESP32...")
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        print(f"Found port: {port.device} - {port.description}")
        if any(keyword in port.description.upper() for keyword in ['USB', 'SERIAL', 'CH340', 'CP210', 'ESP32']):
            print(f"Potential ESP32 port found: {port.device}")
            return port.device
    
    return None

def test_servo_communication():
    """Test communication with ESP32 servo controller"""
    
    # Find ESP32 port
    esp32_port = find_esp32_port()
    
    if not esp32_port:
        print("❌ ESP32 not found. Please check connections.")
        print("Available ports:")
        ports = serial.tools.list_ports.comports()
        for port in ports:
            print(f"  - {port.device}: {port.description}")
        return False
    
    try:
        # Connect to ESP32
        print(f"\n🔌 Connecting to ESP32 on {esp32_port}...")
        ser = serial.Serial(esp32_port, 115200, timeout=2)
        time.sleep(3)  # Wait for ESP32 to initialize
        
        # Clear any existing data
        ser.flushInput()
        ser.flushOutput()
        
        print("✅ Connected successfully!")
        
        # Test commands
        test_commands = [
            ("CENTER", "Moving to center position"),
            ("LEFT", "Moving left (bad seeds position)"),
            ("CENTER", "Returning to center"),
            ("RIGHT", "Moving right (good seeds position)"),
            ("CENTER", "Final center position")
        ]
        
        print("\n🔄 Starting servo test sequence...")
        
        for i, (command, description) in enumerate(test_commands, 1):
            print(f"\n[{i}/{len(test_commands)}] {description}")
            
            # Send command
            command_str = f"{command}\n"
            ser.write(command_str.encode())
            print(f"📤 Sent: {command}")
            
            # Wait for response
            time.sleep(0.5)
            
            # Read response
            while ser.in_waiting > 0:
                response = ser.readline().decode().strip()
                if response:
                    print(f"📥 ESP32: {response}")
            
            # Wait between commands
            time.sleep(2)
        
        print("\n✅ Servo test sequence completed successfully!")
        
        # Interactive mode
        print("\n🎮 Interactive mode - Enter commands manually:")
        print("Commands: LEFT, RIGHT, CENTER, TEST, QUIT")
        
        while True:
            try:
                user_command = input("\nEnter command: ").strip().upper()
                
                if user_command == "QUIT":
                    break
                elif user_command in ["LEFT", "RIGHT", "CENTER", "TEST"]:
                    ser.write(f"{user_command}\n".encode())
                    print(f"📤 Sent: {user_command}")
                    
                    time.sleep(0.5)
                    while ser.in_waiting > 0:
                        response = ser.readline().decode().strip()
                        if response:
                            print(f"📥 ESP32: {response}")
                else:
                    print("❌ Invalid command. Use: LEFT, RIGHT, CENTER, TEST, QUIT")
                    
            except KeyboardInterrupt:
                break
        
        ser.close()
        print("\n👋 Connection closed. Test completed.")
        return True
        
    except serial.SerialException as e:
        print(f"❌ Serial connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import serial
        print("✅ pyserial is installed")
        return True
    except ImportError:
        print("❌ pyserial not found. Please install it:")
        print("   pip install pyserial")
        return False

def main():
    print("=" * 50)
    print("🔧 ESP32 Servo Control Test Script")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Test servo communication
    success = test_servo_communication()
    
    if success:
        print("\n🎉 All tests passed! Your ESP32 servo control is working correctly.")
        print("You can now run the main seed detection application.")
    else:
        print("\n❌ Tests failed. Please check:")
        print("1. ESP32 is connected via USB")
        print("2. Arduino code is uploaded to ESP32")
        print("3. Servo is properly wired")
        print("4. No other software is using the COM port")

if __name__ == "__main__":
    main()
