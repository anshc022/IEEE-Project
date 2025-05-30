# Simple Docker setup for Jetson Nano seed detection
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Set working directory
WORKDIR /app

# Install system dependencies for camera and serial communication
RUN apt-get update && apt-get install -y \
    python3-opencv \
    libopencv-dev \
    v4l-utils \
    usbutils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your project files
COPY app.py .
COPY corn11.pt .
COPY test_servo.py .

# Create directory for temporary files
RUN mkdir -p /tmp

# Expose any ports if needed (optional)
# EXPOSE 8080

# Set environment variables for headless operation
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=:0

# Default command to run your application
CMD ["python3", "app.py"]
