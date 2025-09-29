import serial
import time

# Test ESP32 connection
try:
    print("Testing ESP32 connection on COM11...")
    esp32 = serial.Serial('COM11', 115200, timeout=2)
    time.sleep(3)  # Wait for ESP32 to initialize
    
    print("Sending test commands...")
    
    # Test commands
    commands = ['TEST', 'GOOD', 'BAD', 'REST']
    
    for cmd in commands:
        print(f"\nSending: {cmd}")
        esp32.write(f"{cmd}\n".encode())
        time.sleep(0.5)
        
        # Read response
        while esp32.in_waiting > 0:
            response = esp32.readline().decode().strip()
            print(f"ESP32: {response}")
    
    esp32.close()
    print("\n✅ ESP32 connection test completed!")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")