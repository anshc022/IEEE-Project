#!/usr/bin/env python3
"""
ESP32 COM Port Troubleshooting Script
Helps diagnose and fix permission errors with ESP32 communication
"""

import serial
import serial.tools.list_ports
import time
import subprocess
import sys
import os

def check_com_ports():
    """Check all available COM ports and identify potential ESP32 ports"""
    print("🔍 Scanning for COM ports...")
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("❌ No COM ports found")
        return []
    
    esp32_candidates = []
    
    print("\n📋 Available COM ports:")
    for port in ports:
        print(f"  - {port.device}: {port.description}")
        
        # Check if it might be an ESP32
        keywords = ['USB', 'SERIAL', 'CH340', 'CP210', 'ESP32', 'FTDI', 'Silicon Labs']
        if any(keyword.upper() in port.description.upper() for keyword in keywords):
            esp32_candidates.append(port.device)
            print(f"    ✅ Potential ESP32 port")
    
    return esp32_candidates

def test_port_access(port):
    """Test if we can access a specific COM port"""
    print(f"\n🔧 Testing access to {port}...")
    
    try:
        # Try to open the port
        ser = serial.Serial(port, 115200, timeout=1)
        print(f"✅ Successfully opened {port}")
        ser.close()
        return True
    except serial.SerialException as e:
        if "PermissionError" in str(e) or "Access is denied" in str(e):
            print(f"❌ Permission denied for {port}")
            print(f"   Error: {e}")
            return False
        elif "FileNotFoundError" in str(e) or "cannot find" in str(e).lower():
            print(f"❌ Port {port} not found")
            return False
        else:
            print(f"❌ Failed to access {port}: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error accessing {port}: {e}")
        return False

def check_processes_using_com_ports():
    """Check for processes that might be using COM ports"""
    print("\n🔍 Checking for processes using COM ports...")
    
    try:
        # Check for common applications that use COM ports
        processes_to_check = [
            'arduino.exe',
            'arduino_debug.exe',
            'platformio.exe',
            'putty.exe',
            'teraterm.exe',
            'hyperterminal.exe',
            'minicom.exe'
        ]
        
        # Use tasklist command on Windows
        result = subprocess.run(['tasklist'], capture_output=True, text=True, shell=True)
        running_processes = result.stdout.lower()
        
        found_processes = []
        for process in processes_to_check:
            if process.lower() in running_processes:
                found_processes.append(process)
        
        if found_processes:
            print(f"⚠️  Found these applications that might be using COM ports:")
            for process in found_processes:
                print(f"   - {process}")
            print("\n💡 Try closing these applications and run the test again")
        else:
            print("✅ No obvious COM port applications found running")
            
    except Exception as e:
        print(f"⚠️  Could not check running processes: {e}")

def provide_solutions(failed_ports):
    """Provide solutions for COM port access issues"""
    if not failed_ports:
        return
    
    print(f"\n🔧 TROUBLESHOOTING SOLUTIONS for {failed_ports}:")
    print("=" * 50)
    
    print("\n1️⃣ CLOSE CONFLICTING SOFTWARE:")
    print("   - Close Arduino IDE (especially Serial Monitor)")
    print("   - Close PlatformIO")
    print("   - Close any terminal programs (PuTTY, Tera Term, etc.)")
    
    print("\n2️⃣ DISCONNECT AND RECONNECT ESP32:")
    print("   - Unplug ESP32 USB cable")
    print("   - Wait 5 seconds")
    print("   - Plug it back in")
    print("   - Wait for Windows to recognize the device")
    
    print("\n3️⃣ RUN AS ADMINISTRATOR:")
    print("   - Right-click on your command prompt or terminal")
    print("   - Select 'Run as Administrator'")
    print("   - Try running the application again")
    
    print("\n4️⃣ CHECK DEVICE MANAGER:")
    print("   - Press Win + X, select 'Device Manager'")
    print("   - Look under 'Ports (COM & LPT)'")
    print("   - Check if ESP32 port has any warning icons")
    print("   - Try updating drivers if needed")
    
    print("\n5️⃣ TRY DIFFERENT USB PORT:")
    print("   - Use a different USB port on your computer")
    print("   - Preferably a USB 2.0 port")
    print("   - Avoid USB hubs if possible")
    
    print("\n6️⃣ INSTALL/UPDATE DRIVERS:")
    print("   - For CH340: Download from manufacturer website")
    print("   - For CP210x: Download Silicon Labs drivers")
    print("   - Restart computer after driver installation")

def run_automated_fix():
    """Attempt automated fixes"""
    print("\n🤖 Attempting automated fixes...")
    
    # Try to kill common processes that might be using COM ports
    processes_to_kill = ['arduino.exe', 'putty.exe']
    
    for process in processes_to_kill:
        try:
            result = subprocess.run(['taskkill', '/f', '/im', process], 
                                  capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                print(f"✅ Closed {process}")
            else:
                print(f"ℹ️  {process} was not running")
        except Exception as e:
            print(f"⚠️  Could not check/close {process}: {e}")

def main():
    print("=" * 60)
    print("🔧 ESP32 COM Port Troubleshooting Tool")
    print("=" * 60)
    
    # Check admin privileges
    try:
        is_admin = os.getuid() == 0
    except AttributeError:
        # Windows
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    
    if is_admin:
        print("✅ Running with administrator privileges")
    else:
        print("⚠️  Not running as administrator (might be needed)")
    
    # Step 1: Check available ports
    esp32_candidates = check_com_ports()
    
    if not esp32_candidates:
        print("\n❌ No potential ESP32 ports found!")
        print("\nPossible reasons:")
        print("- ESP32 is not connected")
        print("- ESP32 drivers not installed")
        print("- USB cable is not working")
        print("- ESP32 is damaged")
        return
    
    # Step 2: Test access to each candidate port
    failed_ports = []
    working_ports = []
    
    for port in esp32_candidates:
        if test_port_access(port):
            working_ports.append(port)
        else:
            failed_ports.append(port)
    
    # Step 3: Check for conflicting processes
    if failed_ports:
        check_processes_using_com_ports()
    
    # Step 4: Provide solutions
    if failed_ports:
        provide_solutions(failed_ports)
        
        # Ask if user wants to try automated fix
        response = input("\n❓ Would you like to try automated fixes? (y/n): ").lower()
        if response == 'y':
            run_automated_fix()
            
            # Test again after automated fix
            print("\n🔄 Testing ports again after automated fixes...")
            for port in failed_ports[:]:  # Copy list to avoid modification during iteration
                if test_port_access(port):
                    print(f"✅ {port} is now accessible!")
                    failed_ports.remove(port)
                    working_ports.append(port)
    
    # Step 5: Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    if working_ports:
        print(f"✅ Working ESP32 ports: {', '.join(working_ports)}")
        print("You can now run your seed detection application!")
    
    if failed_ports:
        print(f"❌ Failed ESP32 ports: {', '.join(failed_ports)}")
        print("Follow the troubleshooting steps above to fix these ports.")
    
    if not working_ports and not failed_ports:
        print("⚠️  No ESP32 ports detected. Check hardware connections.")

if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")
