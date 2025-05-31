#!/bin/bash
# Simple script to set environment and run Docker Compose on old versions
# For Docker Compose v1.17.x compatibility

echo "=== Docker Compose v1.17.x Runner ==="
echo "Setting up environment for older Docker Compose..."

# Export DISPLAY variable
export DISPLAY=${DISPLAY:-:0}
echo "DISPLAY set to: $DISPLAY"

# Set up X11 permissions
xhost +local:docker 2>/dev/null || echo "Warning: Could not set X11 permissions (normal for headless)"

# Create necessary directories
mkdir -p logs data models

echo ""
echo "Available compose files:"
echo "1. docker-compose-legacy.yml (includes more dependencies)"
echo "2. docker-compose-v1-minimal.yml (minimal, faster build)"
echo ""

# Check which compose file to use
if [ "$1" = "minimal" ]; then
    COMPOSE_FILE="docker-compose-v1-minimal.yml"
    echo "Using minimal configuration..."
elif [ "$1" = "legacy" ]; then
    COMPOSE_FILE="docker-compose-legacy.yml"
    echo "Using legacy configuration..."
else
    echo "Usage: $0 [minimal|legacy] [build|up|test]"
    echo ""
    echo "Examples:"
    echo "  $0 minimal build    # Build with minimal dependencies"
    echo "  $0 minimal up       # Run with minimal dependencies"
    echo "  $0 minimal test     # Test camera with minimal dependencies"
    echo "  $0 legacy build     # Build with more dependencies"
    echo "  $0 legacy up        # Run with more dependencies"
    exit 1
fi

# Check what action to take
case "$2" in
    "build")
        echo "Building Docker image..."
        docker-compose -f $COMPOSE_FILE build
        ;;
    "up")
        echo "Starting application..."
        docker-compose -f $COMPOSE_FILE up
        ;;
    "test")
        echo "Testing camera functionality..."
        docker-compose -f $COMPOSE_FILE run --rm yolo-seed-detection python3 test_camera.py
        ;;
    "logs")
        echo "Showing logs..."
        docker-compose -f $COMPOSE_FILE logs -f
        ;;
    "down")
        echo "Stopping application..."
        docker-compose -f $COMPOSE_FILE down
        ;;
    *)
        echo "Invalid action. Use: build, up, test, logs, or down"
        exit 1
        ;;
esac
