# Docker Deployment Guide for YOLOv11 Seed Detection on Jetson Nano

## Prerequisites

1. **Jetson Nano** with JetPack 4.6 or later
2. **NVIDIA Container Runtime** installed
3. **Docker** and **Docker Compose** installed
4. **Camera** connected (CSI or USB)
5. **X11 forwarding** enabled for GUI display

## Quick Start

### 1. Initial Setup (First Time Only)

```bash
# Make the setup script executable
chmod +x docker-setup.sh

# Run the setup script
./docker-setup.sh

# Log out and back in for group changes to take effect
```

### 2. Build and Run

```bash
# Build and start the container
docker-compose up --build

# Or run in background
docker-compose up -d --build

# If you get version errors with docker-compose.yml, use the legacy version:
docker-compose -f docker-compose-legacy.yml up --build
```

### 3. Monitor and Control

```bash
# View logs
docker-compose logs -f

# Stop the application
docker-compose down

# Restart the application
docker-compose restart

# Rebuild after code changes
docker-compose up --build --force-recreate
```

## Configuration

### Camera Configuration

The docker-compose.yml file is configured to access:
- `/dev/video0` - Primary camera (usually USB)
- `/dev/video1` - Secondary camera (usually CSI)

If your camera is on a different device, edit the `devices` section in `docker-compose.yml`:

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

### Docker Compose Version Issues

If you see "Version in docker-compose.yml is unsupported" error:

1. **Check your Docker Compose version:**
   ```bash
   docker-compose --version
   ```

2. **For older versions (< 1.12), use the legacy file:**
   ```bash
   docker-compose -f docker-compose-legacy.yml up --build
   ```

3. **Update Docker Compose (recommended):**
   ```bash
   sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

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
