#!/bin/bash
echo "🔧 Fixing Docker Permissions on Jetson Nano"
echo "============================================"

# Check if Docker is running
if ! sudo systemctl is-active --quiet docker; then
    echo "🐳 Starting Docker service..."
    sudo systemctl start docker
    sudo systemctl enable docker
fi

# Add current user to docker group
echo "👤 Adding user '$USER' to docker group..."
sudo usermod -aG docker $USER

# Change docker socket permissions (temporary fix)
echo "🔧 Fixing docker socket permissions..."
sudo chmod 666 /var/run/docker.sock

# Create results directory
mkdir -p results

echo ""
echo "✅ Docker permissions fixed!"
echo ""
echo "🚀 Now trying to run seed detection..."

# Try direct docker run with proper permissions
echo "🐳 Building Docker image..."
docker build -t seed-detection .

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Starting seed detection..."
    
    # Run with proper device mapping and volume mounting
    docker run --rm -it \
        --privileged \
        --device=/dev/video0:/dev/video0 \
        -v "$(pwd)/results:/app/results" \
        -e DISPLAY=:0 \
        --network host \
        seed-detection
else
    echo "❌ Build failed. Let's try a simpler approach..."
    echo ""
    echo "🐍 Running directly with Python (no Docker)..."
    
    # Install dependencies if needed
    pip3 install -r requirements.txt 2>/dev/null
    
    # Run the Python script directly
    python3 app.py
fi
