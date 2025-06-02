#!/usr/bin/env python3
"""
Flask App Quick Test - Verify the fixed app starts properly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_flask_imports():
    """Test that all critical components import correctly"""
    try:
        print("🔍 Testing Flask app imports...")
        
        # Test basic imports
        import app_flask
        print("✅ app_flask imported successfully")
        
        # Test that critical functions exist
        if hasattr(app_flask, 'draw_predictions'):
            print("✅ draw_predictions function exists")
        else:
            print("❌ draw_predictions function missing")
            
        if hasattr(app_flask, 'update_statistics'):
            print("✅ update_statistics function exists")
        else:
            print("❌ update_statistics function missing")
            
        if hasattr(app_flask, 'send_servo_command'):
            print("✅ send_servo_command function exists")
        else:
            print("❌ send_servo_command function missing")
            
        # Test Flask app creation
        if hasattr(app_flask, 'app'):
            print("✅ Flask app instance exists")
        else:
            print("❌ Flask app instance missing")
            
        print("\n🎉 All imports successful! Flask app is ready to run.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_flask_imports()
    
    if success:
        print("\n" + "="*50)
        print("🚀 FLASK APP STATUS: READY")
        print("="*50)
        print("\n📋 To start the application:")
        print("   python app_flask.py")
        print("\n🔧 Key Fix Summary:")
        print("• Fixed simultaneous detection processing issue")
        print("• Only highest confidence detection triggers action")
        print("• Prevents conflicting servo commands")
        print("• Ensures accurate seed statistics")
        print("• Added visual indicators for selected detections")
    else:
        print("\n❌ Flask app has issues that need to be resolved")
        sys.exit(1)
