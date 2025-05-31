# Docker Deployment Guide for YOLOv11 Seed Detection on Jetson Nano

## Prerequisites

1. **Jetson Nano** with JetPack 4.6 or later
2. **NVIDIA Container Runtime** installed
3. **Docker** and **Docker Compose** installed
4. **Camera** connected (CSI or USB)
5. **X11 forwarding** enabled for GUI display

## Repository Issues and Solutions

⚠️ **Important**: Jetson Nano systems often have GPG key and repository issues that can prevent Docker builds. This guide provides multiple solutions.

### Common Repository Errors:
- `GPG error: https://apt.kitware.com/ubuntu/ InRelease`
- `W: Failed to fetch https://developer.download.nvidia.com/...`
- `E: Unable to locate package...`

### Solutions (Try in order):

#### Option 1: Enhanced Setup Script (Recommended)
```bash
# Use the enhanced setup script that fixes repository issues
chmod +x docker-setup-jetson-fixed.sh
./docker-setup-jetson-fixed.sh

# Then use the recommended compose file
docker-compose -f docker-compose-jetson-fixed.yml up --build
```

#### Option 2: Repository-Fixed Dockerfile
```bash
# Use the Dockerfile that specifically handles repository issues
docker-compose -f docker-compose-jetson-fixed.yml up --build
```

#### Option 3: Ultra-Minimal Build (Avoids System Packages)
```bash
# If repository issues persist, use the ultra-minimal build
docker-compose -f docker-compose-ultra-minimal.yml up --build
```

## Quick Start

### 1. Initial Setup (First Time Only)

**Option A: Enhanced Setup (Handles Repository Issues)**
```bash
# Make the enhanced setup script executable
chmod +x docker-setup-jetson-fixed.sh

# Run the enhanced setup script
./docker-setup-jetson-fixed.sh

# Log out and back in for group changes to take effect
```

**Option B: Standard Setup**
```bash
# Make the setup script executable
chmod +x docker-setup.sh

# Run the setup script
./docker-setup.sh

# Log out and back in for group changes to take effect
```

### 2. Build and Run

**Recommended Approach (Repository-Issue Fixed):**
```bash
# Build and start with the repository-fixed version
docker-compose -f docker-compose-jetson-fixed.yml up --build

# Or run in background
docker-compose -f docker-compose-jetson-fixed.yml up -d --build
```

**Alternative Approaches (if repository issues persist):**

```bash
# Ultra-minimal build (only essential packages, no system dependencies):
docker-compose -f docker-compose-ultra-minimal.yml up --build

# Minimal build (includes some optional packages):
docker-compose -f docker-compose-minimal.yml up --build

# Standard build (default - Docker Compose v3.3):
docker-compose up --build

# For Docker Compose with limited GPU support:
docker-compose -f docker-compose-compatible.yml up --build

# For older Docker Compose (v2.4 format):
docker-compose -f docker-compose-legacy.yml up --build

# For very old Docker Compose (v1 format):
docker-compose -f docker-compose-v1.yml up --build
```

**Direct Docker Run (if Docker Compose fails):**
```bash
# Build the repository-fixed image
docker build -f Dockerfile.jetson-fixed -t yolo-seed-detection-jetson .

# Run with full GPU and camera access
docker run --gpus all --privileged -it \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e DISPLAY=${DISPLAY:-:0} \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v /dev:/dev \
  --net=host \
  yolo-seed-detection-jetson
```

### 3. Monitor and Control

```bash
# View logs (adjust compose file as needed)
docker-compose -f docker-compose-jetson-fixed.yml logs -f

# Stop the application
docker-compose -f docker-compose-jetson-fixed.yml down

# Restart the application
docker-compose -f docker-compose-jetson-fixed.yml restart

# Rebuild after code changes
docker-compose -f docker-compose-jetson-fixed.yml up --build --force-recreate
```

## Troubleshooting Repository Issues

### Manual Repository Cleanup
If the automated scripts don't work, manually clean repositories:

```bash
# Remove problematic repository files
sudo rm -f /etc/apt/sources.list.d/kitware*.list*
sudo rm -f /etc/apt/sources.list.d/*cuda*.list.disabled

# Clean package cache
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*

# Update package cache
sudo apt-get update
```

### Check for Repository Errors
```bash
# Check for repository errors
sudo apt-get update 2>&1 | grep -i "error\|warning"

# List active repositories
grep -r "^deb" /etc/apt/sources.list /etc/apt/sources.list.d/
```

### Alternative Package Installation
If system packages fail, use pip-only installation:
```bash
# Use the ultra-minimal Dockerfile that only uses pip
docker build -f Dockerfile.ultra-minimal -t yolo-seed-detection-minimal .
```

## Configuration

### Camera Configuration

The docker-compose.yml file is configured to access:
- `/dev/video0` - Primary camera (usually USB)
- `/dev/video1` - Secondary camera (usually CSI)

If your camera is on a different device, edit the `devices` section in the compose file:

```yaml
devices:
  - /dev/video2:/dev/video2  # Adjust as needed
```

### Serial Port Configuration

For Arduino communication, the default serial ports are:
- `/dev/ttyUSB0`
- `/dev/ttyACM0`

Check your actual device with:
```bash
ls -la /dev/tty*
```

### GPU Memory Settings

If you encounter GPU memory issues, you can limit GPU memory in the Dockerfile by adding:

```dockerfile
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_CACHE_MAXSIZE=1073741824  # 1GB cache limit
```

## Troubleshooting

### Camera Issues

1. **Check camera devices:**
   ```bash
   ls -la /dev/video*
   v4l2-ctl --list-devices
   ```

2. **Test camera outside container:**
   ```bash
   python3 test_camera.py
   ```

3. **Check camera permissions:**
   ```bash
   sudo usermod -a -G video $USER
   ```

### Display Issues

1. **Enable X11 forwarding:**
   ```bash
   xhost +local:docker
   ```

2. **Check DISPLAY variable:**
   ```bash
   echo $DISPLAY
   ```

3. **For headless operation**, modify the Qt backend in `app.py`:
   ```python
   os.environ['QT_QPA_PLATFORM'] = 'offscreen'
   ```

### GPU/CUDA Issues

1. **Check NVIDIA runtime:**
   ```bash
   docker info | grep nvidia
   ```

2. **Verify CUDA in container:**
   ```bash
   docker-compose exec yolo-seed-detection nvidia-smi
   ```

3. **Check PyTorch CUDA:**
   ```bash
   docker-compose exec yolo-seed-detection python3 -c "import torch; print(torch.cuda.is_available())"
   ```

### Package Repository Issues

1. **GPG key errors (Kitware, etc.):**
   ```bash
   # Use the ultra-minimal build which avoids all problematic repositories
   docker-compose -f docker-compose-ultra-minimal.yml up --build
   
   # Or build directly with ultra-minimal Dockerfile
   docker build -f Dockerfile.ultra-minimal -t yolo-ultra .
   ```

2. **"Package not found" or "Repository not signed" errors:**
   ```bash
   # Try the minimal version which handles repository cleanup
   docker build -f Dockerfile.minimal.v2 -t yolo-minimal .
   
   # Or use the ultra-minimal that only installs from PyPI
   docker build -f Dockerfile.ultra-minimal -t yolo-ultra .
   ```

3. **Ubuntu version compatibility:**
   ```bash
   # Check your Ubuntu version
   lsb_release -a
   
   # The ultra-minimal version works on most Ubuntu versions
   # as it only uses pip packages, not system packages
   ```

4. **Complete repository cleanup (if needed):**
   ```bash
   # Remove all problematic repositories manually
   sudo rm -f /etc/apt/sources.list.d/kitware*
   sudo apt-get update
   
   # Then try building again
   docker-compose -f docker-compose-ultra-minimal.yml up --build
   ```

### Memory Issues

1. **Increase swap space:**
   ```bash
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

2. **Monitor memory usage:**
   ```bash
   docker stats
   ```

## Development Workflow

1. **Make code changes** in your local files
2. **Rebuild container:** `docker-compose up --build`
3. **Test changes** in the container environment
4. **Commit and push** when satisfied

## Performance Optimization

### For better performance on Jetson Nano:

1. **Enable maximum performance mode:**
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

2. **Monitor system resources:**
   ```bash
   sudo pip3 install jetson-stats
   jtop
   ```

3. **Optimize model settings** in `app.py`:
   ```python
   # Use smaller input size for faster inference
   model = YOLO('corn11.pt')
   model.conf = 0.5  # Adjust confidence threshold
   model.iou = 0.45  # Adjust IoU threshold
   ```

## File Structure

```
IEEE-Project/
├── Dockerfile              # Container definition
├── docker-compose.yml      # Service orchestration
├── docker-setup.sh         # Initial setup script
├── .dockerignore           # Build optimization
├── app.py                  # Main application
├── test_camera.py          # Camera testing tool
├── fix_camera.py           # Camera diagnostic tool
├── requirements.txt        # Python dependencies
├── corn11.pt              # YOLOv11 model
└── logs/                  # Application logs (created by container)
```

## Support

For issues specific to:
- **Camera problems**: Check `test_camera.py` and `fix_camera.py` outputs
- **Model issues**: Verify `corn11.pt` model file
- **Docker issues**: Check Docker and NVIDIA container runtime installation
- **Performance**: Monitor with `jtop` and adjust model parameters
