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

## Files Created

- `Dockerfile` - Simple container setup for Jetson Nano
- `docker-compose.yml` - Easy container management
- `run.sh` - One-click startup script for Linux
- `run.bat` - One-click startup script for Windows (testing)

## Troubleshooting

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

## Saved Images

Detected seed images are saved to the `results/` folder.

## Simple Architecture

```
📁 Your Project
├── 🐳 Dockerfile (simple container)
├── 📋 docker-compose.yml (easy run)
├── 🚀 run.sh (one-click start)
├── 🤖 app.py (your detection code)
├── 🧠 corn11.pt (your AI model)
└── 📊 results/ (saved images)
```

No complex setup - just run and go! 🎯
