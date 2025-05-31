# Test script to verify servo control logic
import time

# Mock servo status for testing
servo_status = {
    'connected': True,
    'port': 'COM3',
    'last_command': None,
    'last_response_time': None,
    'sorting_active': False,
    'total_sorts': 0
}

def send_servo_command(command):
    """Mock servo command function for testing"""
    print(f"🎯 Servo Command: {command}")
    
    servo_status['last_command'] = command
    servo_status['last_response_time'] = time.time()
    
    # Count sorting operations
    if command in ['LEFT', 'RIGHT']:
        servo_status['total_sorts'] += 1
    
    # Show the expected physical movement
    if command == 'LEFT':
        print("   ⬅️  Servo moves LEFT (0°) - BAD SEEDS")
    elif command == 'RIGHT':
        print("   ➡️  Servo moves RIGHT (180°) - GOOD SEEDS")
    elif command == 'CENTER':
        print("   🎯 Servo moves CENTER (90°)")
    
    return True

def update_statistics(seed_type):
    """Mock statistics update function"""
    print(f"\n🌱 Detected: {seed_type.upper()} SEED")
    
    # Control servo based on seed quality
    if servo_status['connected']:
        servo_status['sorting_active'] = True
        if seed_type == 'good':
            send_servo_command('RIGHT')  # Good seeds go RIGHT (180°)
        elif seed_type == 'bad':
            send_servo_command('LEFT')   # Bad seeds go LEFT (0°)

def test_manual_controls():
    """Test manual control commands"""
    print("\n🎮 Testing Manual Controls:")
    
    # Test GOOD command (should map to RIGHT)
    print("\n👆 User clicks 'Sort Good' button:")
    command = 'GOOD'
    servo_command = 'RIGHT' if command == 'GOOD' else ('LEFT' if command == 'BAD' else command)
    send_servo_command(servo_command)
    
    # Test BAD command (should map to LEFT)
    print("\n👆 User clicks 'Sort Bad' button:")
    command = 'BAD'
    servo_command = 'RIGHT' if command == 'GOOD' else ('LEFT' if command == 'BAD' else command)
    send_servo_command(servo_command)
    
    # Test CENTER command
    print("\n👆 User clicks 'Center' button:")
    send_servo_command('CENTER')

def main():
    print("🔧 SERVO CONTROL LOGIC TEST")
    print("=" * 50)
    
    print("\n📊 Expected Behavior:")
    print("• GOOD seeds → Servo RIGHT (180°)")
    print("• BAD seeds → Servo LEFT (0°)")
    print("• Manual controls should match this logic")
    
    print("\n🤖 Testing Automatic Detection:")
    
    # Test automatic detection
    update_statistics('good')
    time.sleep(1)
    update_statistics('bad')
    time.sleep(1)
    update_statistics('good')
    
    # Test manual controls
    test_manual_controls()
    
    print(f"\n📈 Total Sorts: {servo_status['total_sorts']}")
    print(f"🔌 Last Command: {servo_status['last_command']}")
    
    print("\n✅ Test Complete!")
    print("\nIf the output shows:")
    print("• Good seeds → RIGHT movement ➡️")
    print("• Bad seeds → LEFT movement ⬅️")
    print("• Manual 'Sort Good' → RIGHT movement ➡️")
    print("• Manual 'Sort Bad' → LEFT movement ⬅️")
    print("Then the servo logic is CORRECT! 🎉")

if __name__ == "__main__":
    main()
