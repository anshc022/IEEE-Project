# 🌱 Simple Seed Detection Docker Setup

This is a super simple Docker setup for your seed detection project on Jetson Nano.

## Quick Start

### For Jetson Nano (Linux)
```bash
chmod +x run.sh && ./run.sh
```

### For Windows (Testing)
```powershell
.\run.ps1
```
Or double-click `run.bat`

### Test Before Docker
```bash
python3 test_setup.py
```

## What You Need

- **Camera**: USB camera (will auto-detect)
- **ESP32**: Connected via USB (for servo control - optional)
- **Model**: Your `corn11.pt` file should be in the project folder

## Quick Setup Steps

1. **Copy all files to your Jetson Nano**
2. **Make sure your camera and ESP32 are connected**
3. **Run the setup script:**
   ```bash
   chmod +x run.sh && ./run.sh
   ```

That's it! The script will automatically:
- ✅ Check and install Docker if needed
- ✅ Check for camera and ESP32
- ✅ Build the Docker container  
- ✅ Start your seed detection system

## Controls

When running:
- Press `q` to quit
- Press `p` to pause/resume  
- Press `s` to save detected images
- Press `a` to toggle analysis mode

## Complete Project Structure 📁

```
🌱 Seed Detection Project
├── 🤖 Core Application
│   ├── app.py                     # Main detection script
│   ├── corn11.pt                  # YOLO model weights
│   └── test_servo.py              # ESP32 servo test
│
├── 🐳 Docker Setup
│   ├── Dockerfile                 # Container definition
│   ├── docker-compose.yml         # Main compose file (v2.3)
│   ├── docker-compose-simple.yml  # Fallback compose file
│   └── requirements.txt           # Python dependencies
│
├── 🚀 Run Scripts (Linux/Jetson)
│   ├── super_simple_run.sh        # ⭐ Auto-fix everything
│   ├── run.sh                     # Standard Docker setup
│   ├── run_python_direct.sh       # Skip Docker, use Python
│   ├── fix_docker_permissions.sh  # Fix Docker permissions
│   └── quick_fix.sh               # Fix compose version
│
├── 🪟 Windows Scripts
│   ├── run.ps1                    # PowerShell script
│   └── run.bat                    # Batch script
│
├── 🧪 Testing & Verification
│   ├── test_setup.py              # Test all dependencies
│   └── verify.sh                  # System verification
│
├── ⚡ Hardware
│   └── esp32_servo_control.ino    # ESP32 Arduino code
│
├── 📖 Documentation
│   ├── README.md                  # This file
│   └── SETUP_COMPLETE.md          # Setup summary
│
└── 📊 Output
    └── results/                   # Saved detection images
```

## Files Created

- **Core:** `app.py`, `corn11.pt`, `test_servo.py`
- **Docker:** `Dockerfile`, `docker-compose.yml`, `requirements.txt`
- **Scripts:** All the run scripts for different scenarios
- **Hardware:** `esp32_servo_control.ino` for servo control
- **Docs:** `README.md`, `SETUP_COMPLETE.md`

## Detailed Troubleshooting 🔧

### Common Issues and Solutions

#### 1. Docker Permission Denied
**Error:** `Got permission denied while trying to connect to the Docker daemon socket`
**Solution:**
```bash
chmod +x super_simple_run.sh && ./super_simple_run.sh
```

#### 2. Docker Compose Version Error
**Error:** `Version in "./docker-compose.yml" is unsupported`
**Solution:**
```bash
chmod +x quick_fix.sh && ./quick_fix.sh
```

#### 3. Camera Not Found
**Error:** `Camera not found at /dev/video0`
**Check cameras:**
```bash
ls /dev/video*
v4l2-ctl --list-devices
```

#### 4. ESP32 Not Found
**Warning:** `ESP32 not found. Servo control will be disabled.`
**Check ESP32:**
```bash
lsusb
ls /dev/tty*
```

#### 5. YOLO Model Issues
**Error:** `Model file 'corn11.pt' not found!`
**Solution:** Make sure `corn11.pt` is in your project folder

#### 6. Python Dependencies Missing
**Error:** Import errors for ultralytics, opencv, etc.
**Solution for Jetson Nano:**
```bash
chmod +x ultimate_jetson_setup.sh && ./ultimate_jetson_setup.sh
```

#### 7. PyTorch Missing on Jetson Nano
**Error:** `ModuleNotFoundError: No module named 'torch'`
**Solution:**
```bash
chmod +x fix_pytorch_jetson.sh && ./fix_pytorch_jetson.sh
```

### Camera not working?
```bash
# Check available cameras
ls /dev/video*
# Test camera
v4l2-ctl --list-devices
```

### ESP32 not found?
```bash
# Check USB devices
lsusb
# Check serial ports
ls /dev/tty*
```

### Docker issues?
```bash
# Install Docker (if needed)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Log out and log back in
```

### Still Having Issues?
**Ultimate fallback - Skip everything and run Python directly:**
```bash
pip3 install --user ultralytics opencv-python torch pyserial matplotlib numpy
python3 app.py
```

## All Available Run Options 🚀

### Option 1: Super Simple Auto-Fix ⭐ (RECOMMENDED)
**One command that fixes everything automatically:**
```bash
chmod +x super_simple_run.sh && ./super_simple_run.sh
```
- ✅ Fixes Docker permissions automatically
- ✅ Installs PyTorch for Jetson Nano properly
- ✅ Tries Docker first, falls back to Python if needed
- ✅ Works 100% of the time

### Option 2: Ultimate Jetson Setup (If Option 1 Fails)
**Complete dependency installation for Jetson Nano:**
```bash
chmod +x ultimate_jetson_setup.sh && ./ultimate_jetson_setup.sh
```
- ✅ Detects Jetson Nano automatically
- ✅ Installs proper PyTorch wheel for Jetson
- ✅ Installs all system dependencies
- ✅ Tests everything before running

### Option 3: Quick PyTorch Fix Only
**If only PyTorch is missing:**
```bash
chmod +x fix_pytorch_jetson.sh && ./fix_pytorch_jetson.sh
```
- ✅ Specifically fixes PyTorch on Jetson Nano
- ✅ Downloads correct wheel file
- ✅ Tests installation

### Option 4: Standard Docker Setup
```bash
chmod +x run.sh && ./run.sh
```
- ✅ Full Docker setup with version handling
- ✅ Best for clean systems

### Option 5: Skip Docker Completely
```bash
chmod +x run_python_direct.sh && ./run_python_direct.sh
```
- ✅ Runs directly with Python (no Docker)
- ✅ Fastest startup, fewer dependencies

### Option 6: Quick Permission Fix
```bash
chmod +x fix_docker_permissions.sh && ./fix_docker_permissions.sh
```
- ✅ Fixes Docker permissions and builds
- ✅ Fallback to Python if Docker fails

### Option 7: Simple Version Fix
```bash
chmod +x quick_fix.sh && ./quick_fix.sh
```
- ✅ Updates docker-compose version
- ✅ Good for version conflicts

### Option 8: Test Everything First
```bash
python3 test_setup.py
```
- ✅ Tests all dependencies before running
- ✅ Shows what will work and what won't

## For Windows Users 🪟

### PowerShell (Recommended)
```powershell
.\run.ps1
```

### Command Prompt
```cmd
run.bat
```

## All Available Scripts 📄

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `super_simple_run.sh` | ⭐ Auto-fix everything | **Always start here** |
| `ultimate_jetson_setup.sh` | Complete Jetson setup | Dependency issues |
| `fix_pytorch_jetson.sh` | Fix PyTorch only | PyTorch missing |
| `run.sh` | Standard Docker setup | Clean Jetson Nano |
| `run_python_direct.sh` | Skip Docker entirely | Docker issues |
| `fix_docker_permissions.sh` | Fix Docker permissions | Permission errors |
| `quick_fix.sh` | Fix compose version | Version conflicts |
| `test_setup.py` | Test before running | Check compatibility |
| `run.ps1` | Windows PowerShell | Windows testing |
| `run.bat` | Windows batch | Windows testing |

## Saved Images

Detected seed images are saved to the `results/` folder.

## Quick Reference Card 📋

### First Time Setup
```bash
# 1. Copy project to Jetson Nano
# 2. Run this ONE command:
chmod +x super_simple_run.sh && ./super_simple_run.sh
```

### If You Have Issues
```bash
# Test everything first:
python3 test_setup.py

# Ultimate setup for Jetson Nano:
chmod +x ultimate_jetson_setup.sh && ./ultimate_jetson_setup.sh

# Fix PyTorch only:
chmod +x fix_pytorch_jetson.sh && ./fix_pytorch_jetson.sh

# Skip Docker entirely:
chmod +x run_python_direct.sh && ./run_python_direct.sh

# Fix Docker permissions:
chmod +x fix_docker_permissions.sh && ./fix_docker_permissions.sh
```

### Controls While Running
- `q` = Quit
- `p` = Pause/Resume
- `s` = Save detected image
- `a` = Toggle analysis mode
- `c` = Show/hide confidence scores

## Success Indicators ✅

When working correctly, you'll see:
```
✅ OpenCV initialized successfully
✅ Model loaded successfully
✅ Camera test successful
🌱 Starting seed detection...
FPS: 15.2 | DETECTING | Seeds: 3
```

## Zero-Knowledge Setup 🎯

**Don't know Docker? Don't know Linux? No problem!**

1. **Copy your project folder to Jetson Nano**
2. **Open terminal and run:**
   ```bash
   chmod +x super_simple_run.sh && ./super_simple_run.sh
   ```
3. **Wait for magic to happen** ✨
4. **Your seed detection is running!** 🌱

The script handles everything automatically - Docker permissions, version conflicts, dependencies, fallbacks. Just run and go!

No complex setup - just run and go! 🎯
