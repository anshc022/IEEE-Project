#!/usr/bin/env python3
"""
ESP32 Serial Connection Diagnostics and Fix
This script helps diagnose and fix ESP32 connection issues
"""

import serial
import serial.tools.list_ports
import time
import sys
import subprocess
import os

def list_all_ports():
    """List all available serial ports with detailed information"""
    print("🔍 Scanning all serial ports...")
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("❌ No serial ports found!")
        return []
    
    print(f"📡 Found {len(ports)} serial port(s):")
    for port in ports:
        print(f"  • {port.device}")
        print(f"    Description: {port.description}")
        print(f"    Hardware ID: {port.hwid}")
        print(f"    Manufacturer: {port.manufacturer}")
        print(f"    Product: {port.product}")
        print(f"    Serial Number: {port.serial_number}")
        print(f"    Interface: {port.interface}")
        print()
    
    return ports

def find_esp32_ports():
    """Find potential ESP32 ports based on common identifiers"""
    print("🎯 Looking for ESP32 devices...")
    ports = serial.tools.list_ports.comports()
    esp32_ports = []
    
    # Common ESP32 USB-to-serial chip identifiers
    esp32_identifiers = [
        'ch340',       # CH340 chip
        'ch341',       # CH341 chip
        'cp210x',      # Silicon Labs CP210x
        'cp2102',      # Silicon Labs CP2102
        'cp2104',      # Silicon Labs CP2104
        'ft232',       # FTDI FT232
        'esp32',       # Generic ESP32
        'silicon labs', # Silicon Labs
        'qinheng',     # QinHeng (CH340 manufacturer)
        'usb2.0-serial' # Generic USB serial
    ]
    
    for port in ports:
        port_info = f"{port.description} {port.manufacturer or ''} {port.hwid}".lower()
        
        for identifier in esp32_identifiers:
            if identifier in port_info:
                esp32_ports.append(port)
                print(f"✅ Found potential ESP32: {port.device} ({port.description})")
                break
    
    if not esp32_ports:
        print("❌ No ESP32 devices found!")
        print("💡 Make sure:")
        print("   1. ESP32 is connected via USB")
        print("   2. ESP32 drivers are installed")
        print("   3. USB cable supports data transfer (not just power)")
    
    return esp32_ports

def test_port_access(port_name, baudrate=115200):
    """Test if we can access a specific port"""
    print(f"🔐 Testing access to {port_name}...")
    
    try:
        # Try to open the port briefly
        with serial.Serial(port_name, baudrate, timeout=1) as ser:
            print(f"✅ Successfully opened {port_name}")
            time.sleep(0.5)  # Brief pause
            
            # Try to write a test command
            ser.write(b'PING\n')
            time.sleep(0.5)
            
            # Try to read response
            if ser.in_waiting > 0:
                response = ser.readline().decode().strip()
                print(f"📨 Response: '{response}'")
                return True, response
            else:
                print(f"⚠️  No response from {port_name}")
                return True, None
                
    except PermissionError as e:
        print(f"❌ Permission denied for {port_name}: {e}")
        print("💡 Port might be used by another application")
        return False, str(e)
    except Exception as e:
        print(f"❌ Failed to access {port_name}: {e}")
        return False, str(e)

def kill_competing_processes():
    """Try to close applications that might be using the serial port"""
    print("🔧 Checking for competing processes...")
    
    # Common applications that might use serial ports
    process_names = [
        'arduino.exe',
        'platformio.exe',
        'putty.exe',
        'teraterm.exe',
        'coolterm.exe',
        'serialport.exe',
        'python.exe'  # Other Python scripts
    ]
    
    killed_any = False
    
    try:
        # Get list of running processes
        result = subprocess.run(['tasklist'], capture_output=True, text=True)
        running_processes = result.stdout.lower()
        
        for process_name in process_names:
            if process_name.lower() in running_processes:
                print(f"⚠️  Found {process_name} running")
                response = input(f"Kill {process_name}? (y/n): ").lower().strip()
                if response == 'y':
                    try:
                        subprocess.run(['taskkill', '/f', '/im', process_name], check=True)
                        print(f"✅ Killed {process_name}")
                        killed_any = True
                    except subprocess.CalledProcessError:
                        print(f"❌ Failed to kill {process_name}")
        
        if killed_any:
            print("⏳ Waiting 2 seconds for processes to close...")
            time.sleep(2)
            
    except Exception as e:
        print(f"❌ Error checking processes: {e}")

def test_esp32_connection(port_name, baudrate=115200):
    """Test full ESP32 connection with servo commands"""
    print(f"🎮 Testing ESP32 servo control on {port_name}...")
    
    try:
        with serial.Serial(port_name, baudrate, timeout=2) as ser:
            print(f"✅ Connected to ESP32 on {port_name}")
            time.sleep(2)  # Wait for ESP32 to initialize
            
            # Test commands
            commands = ['CENTER', 'LEFT', 'RIGHT', 'CENTER']
            
            for cmd in commands:
                print(f"📤 Sending: {cmd}")
                ser.write(f"{cmd}\n".encode())
                time.sleep(1)
                
                # Read response
                response = ""
                timeout = time.time() + 2
                while time.time() < timeout:
                    if ser.in_waiting > 0:
                        line = ser.readline().decode().strip()
                        if line:
                            response += line + " "
                    time.sleep(0.1)
                
                if response:
                    print(f"📨 Response: {response.strip()}")
                else:
                    print("⚠️  No response")
                
                time.sleep(1)
            
            print("✅ ESP32 servo test completed successfully!")
            return True
            
    except Exception as e:
        print(f"❌ ESP32 test failed: {e}")
        return False

def main():
    print("🔧 ESP32 CONNECTION DIAGNOSTICS & FIX")
    print("=" * 50)
    
    # Step 1: List all ports
    all_ports = list_all_ports()
    
    # Step 2: Find ESP32 ports
    esp32_ports = find_esp32_ports()
    
    if not esp32_ports:
        print("❌ No ESP32 devices found. Please check your connections.")
        return
    
    # Step 3: Test each potential ESP32 port
    working_ports = []
    
    for port in esp32_ports:
        print(f"\n🧪 Testing {port.device}...")
        success, response = test_port_access(port.device)
        
        if success:
            working_ports.append(port.device)
        else:
            print(f"⚠️  Port {port.device} is not accessible")
            
            # Try to fix permission issues
            if "permission denied" in response.lower():
                print("🔧 Attempting to fix permission issues...")
                kill_competing_processes()
                
                # Test again after killing processes
                print(f"🔄 Retesting {port.device}...")
                success, response = test_port_access(port.device)
                if success:
                    working_ports.append(port.device)
    
    # Step 4: Full ESP32 test on working ports
    if working_ports:
        print(f"\n🎯 Found {len(working_ports)} accessible port(s):")
        for port in working_ports:
            print(f"  ✅ {port}")
        
        # Test the first working port
        main_port = working_ports[0]
        print(f"\n🎮 Testing ESP32 servo functionality on {main_port}...")
        
        if test_esp32_connection(main_port):
            print(f"\n🎉 SUCCESS! ESP32 is working on {main_port}")
            print(f"\n📋 Use this port in your Flask app:")
            print(f"   servo_port = '{main_port}'")
        else:
            print(f"\n❌ ESP32 test failed on {main_port}")
    else:
        print("\n❌ No accessible ESP32 ports found!")
        print("\n💡 Troubleshooting steps:")
        print("   1. Unplug and reconnect the ESP32")
        print("   2. Close Arduino IDE and other serial applications")
        print("   3. Try a different USB cable")
        print("   4. Try a different USB port")
        print("   5. Reinstall ESP32 drivers")

if __name__ == "__main__":
    main()
