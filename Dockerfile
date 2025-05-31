# Dockerfile for YOLOv11 Seed Detection on Jetson Nano
# Based on NVIDIA's L4T PyTorch container for ARM64/Jetson devices

FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=:0

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    cmake \
    libopencv-dev \
    python3-opencv \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-doc \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libqt5gui5 \
    libqt5core5a \
    libqt5dbus5 \
    qt5-default \
    x11-apps \
    v4l-utils \
    udev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional dependencies for Jetson Nano
RUN pip3 install --no-cache-dir \
    pycuda \
    jetson-stats \
    pyserial

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data

# Set permissions
RUN chmod +x *.py
RUN chmod +x *.sh

# Expose any ports if needed (uncomment if you add a web interface)
# EXPOSE 8080

# Create a non-root user for security
RUN useradd -m -u 1000 jetson && \
    chown -R jetson:jetson /app
USER jetson

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python3 -c "import cv2, ultralytics; print('Dependencies OK')" || exit 1

# Default command
CMD ["python3", "app.py"]
