import serial
import time

print("Testing new ESP32 code...")
try:
    esp32 = serial.Serial('COM11', 115200, timeout=2)
    time.sleep(2)
    
    # Test the new commands
    commands = ['GOOD', 'BAD', 'REST']
    
    for cmd in commands:
        print(f"\nTesting: {cmd}")
        esp32.write(f"{cmd}\n".encode())
        time.sleep(1)
        
        # Read response
        while esp32.in_waiting > 0:
            response = esp32.readline().decode().strip()
            print(f"ESP32: {response}")
    
    esp32.close()
    print("✅ Test completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to upload the new Arduino code first!")