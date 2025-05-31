# 🌱 Ultra Simple YOLO Seed Detection

**One-command setup for Jetson Nano with CUDA acceleration!**

## 🚀 Quick Start

### For Jetson Nano (Linux)
```bash
chmod +x super_simple_run.sh && ./super_simple_run.sh
```

### For Windows (Testing)
```cmd
run.bat
```

### Manual Run (Any OS)
```bash
python3 run.py
```

**That's it!** 🎉

## What You Need

- **Jetson Nano** (or any computer with Python3)
- **Camera** (USB camera - will auto-detect)
- **ESP32** (optional - for servo control)
- **Model file**: `corn11.pt` in the same folder

## What It Does Automatically

✅ **Installs all packages** (ultralytics, opencv, torch, etc.)  
✅ **Detects CUDA** and uses GPU acceleration if available  
✅ **Finds your camera** automatically  
✅ **Detects ESP32** for servo control  
✅ **Starts seed detection** with real-time display  

**Performance**: 2-5x faster with CUDA on Jetson Nano!

## Controls While Running

- Press `q` to quit
- Press `p` to pause/resume  
- Press `s` to save detected images
- Press `a` to toggle analysis mode
- Press `c` to show/hide confidence scores

## Project Files

**Essential files only:**
```
🌱 YOLO Seed Detection
├── app.py                 # Main detection script (CUDA-optimized)
├── corn11.pt              # Your YOLO model weights  
├── run.py                 # Setup script (installs everything)
├── super_simple_run.sh    # Linux one-command run
├── run.bat                # Windows one-command run
├── requirements.txt       # Python dependencies
├── test_servo.py          # ESP32 servo test
└── esp32_servo_control.ino # ESP32 Arduino code
```

## Troubleshooting

**Camera not working?**
```bash
ls /dev/video*  # Check available cameras
```

**ESP32 not found?**
```bash
ls /dev/tty*    # Check serial ports
```

**Python packages missing?**
```bash
pip3 install ultralytics opencv-python torch numpy matplotlib pyserial
```

**Still having issues?**
The `run.py` script handles everything automatically and shows detailed error messages.

## Zero-Knowledge Setup 🎯

**Don't know Python? Don't know Linux? No problem!**

1. **Copy these files to your Jetson Nano**
2. **Put your `corn11.pt` model file in the same folder**  
3. **Run this command:**
   ```bash
   chmod +x super_simple_run.sh && ./super_simple_run.sh
   ```
4. **Watch it work!** 🌱

No Docker, no complex setup, no configuration files. Just run and go! 🚀

## Success Output

When working, you'll see:
```
🌱 Ultra Simple YOLO Seed Detection Setup
✅ CUDA available: NVIDIA Tegra X1
📷 Camera found: /dev/video0
🔌 ESP32 possibly found: /dev/ttyUSB0
🚀 Starting YOLO Seed Detection...
FPS: 25.3 | DETECTING | Seeds: 2
```

**Expected performance on Jetson Nano**: 15-30 FPS with CUDA acceleration!
