#!/bin/bash
# Docker Test Script for Jetson Nano
# Tests Docker setup, GPU access, and camera functionality

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[TEST]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Test 1: Basic Docker functionality
test_docker() {
    print_status "Testing basic Docker functionality..."
    if docker run --rm hello-world > /dev/null 2>&1; then
        print_status "✓ Docker is working correctly"
    else
        print_error "✗ Docker test failed"
        return 1
    fi
}

# Test 2: NVIDIA Docker runtime
test_nvidia_docker() {
    print_status "Testing NVIDIA Docker runtime..."
    if docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu18.04 nvidia-smi > /dev/null 2>&1; then
        print_status "✓ NVIDIA Docker runtime is working"
    else
        print_warning "✗ NVIDIA Docker runtime test failed (this may be expected on some systems)"
        print_info "Continuing with CPU-only testing..."
    fi
}

# Test 3: Docker Compose version
test_docker_compose() {
    print_status "Testing Docker Compose..."
    if command -v docker-compose &> /dev/null; then
        COMPOSE_VERSION=$(docker-compose --version)
        print_status "✓ Docker Compose found: $COMPOSE_VERSION"
        
        # Determine best compose file
        VERSION_NUM=$(echo $COMPOSE_VERSION | grep -o '[0-9]\+\.[0-9]\+' | head -1)
        MAJOR=$(echo $VERSION_NUM | cut -d. -f1)
        MINOR=$(echo $VERSION_NUM | cut -d. -f2)
        
        if [ "$MAJOR" -ge 2 ] || ([ "$MAJOR" -eq 1 ] && [ "$MINOR" -ge 25 ]); then
            RECOMMENDED_COMPOSE="docker-compose-jetson-fixed.yml"
        elif [ "$MAJOR" -eq 1 ] && [ "$MINOR" -ge 12 ]; then
            RECOMMENDED_COMPOSE="docker-compose-legacy.yml"
        else
            RECOMMENDED_COMPOSE="docker-compose-v1.yml"
        fi
        
        print_info "Recommended compose file: $RECOMMENDED_COMPOSE"
    else
        print_error "✗ Docker Compose not found"
        return 1
    fi
}

# Test 4: Camera devices
test_camera_devices() {
    print_status "Testing camera device access..."
    CAMERAS_FOUND=0
    
    for i in {0..5}; do
        if [ -e "/dev/video$i" ]; then
            print_status "✓ Found camera device: /dev/video$i"
            v4l2-ctl --device=/dev/video$i --list-formats-ext 2>/dev/null | head -5 || true
            CAMERAS_FOUND=$((CAMERAS_FOUND + 1))
        fi
    done
    
    if [ $CAMERAS_FOUND -eq 0 ]; then
        print_warning "✗ No camera devices found"
        print_info "Make sure your camera is connected and drivers are loaded"
    else
        print_status "✓ Found $CAMERAS_FOUND camera device(s)"
    fi
}

# Test 5: Build Docker image
test_docker_build() {
    print_status "Testing Docker image build (this may take several minutes)..."
    
    # Try the repository-fixed Dockerfile first
    if docker build -f Dockerfile.jetson-fixed -t yolo-test-build . > /tmp/docker_build.log 2>&1; then
        print_status "✓ Docker image built successfully using Dockerfile.jetson-fixed"
        docker rmi yolo-test-build > /dev/null 2>&1 || true
    elif docker build -f Dockerfile.ultra-minimal -t yolo-test-build . > /tmp/docker_build_minimal.log 2>&1; then
        print_status "✓ Docker image built successfully using Dockerfile.ultra-minimal"
        docker rmi yolo-test-build > /dev/null 2>&1 || true
    else
        print_error "✗ Docker image build failed"
        print_info "Check logs at /tmp/docker_build.log and /tmp/docker_build_minimal.log"
        return 1
    fi
}

# Test 6: Python dependencies
test_python_deps() {
    print_status "Testing Python dependencies in container..."
    
    cat > /tmp/test_deps.py << 'EOF'
import sys
try:
    import cv2
    print("✓ OpenCV:", cv2.__version__)
except ImportError as e:
    print("✗ OpenCV import failed:", e)
    sys.exit(1)

try:
    import ultralytics
    print("✓ Ultralytics:", ultralytics.__version__)
except ImportError as e:
    print("✗ Ultralytics import failed:", e)
    sys.exit(1)

try:
    import numpy
    print("✓ NumPy:", numpy.__version__)
except ImportError as e:
    print("✗ NumPy import failed:", e)
    sys.exit(1)

print("✓ All essential dependencies are working")
EOF
    
    if docker run --rm -v /tmp/test_deps.py:/test_deps.py python:3.8-slim python /test_deps.py > /dev/null 2>&1; then
        print_status "✓ Python dependencies test passed"
    else
        print_warning "✗ Python dependencies test failed (expected if packages not pre-installed)"
    fi
    
    rm /tmp/test_deps.py
}

# Test 7: Display forwarding
test_display() {
    print_status "Testing display forwarding..."
    
    if [ -n "$DISPLAY" ]; then
        print_status "✓ DISPLAY variable set: $DISPLAY"
        
        if xhost +local:docker > /dev/null 2>&1; then
            print_status "✓ X11 forwarding configured"
        else
            print_warning "✗ Could not configure X11 forwarding (normal for headless setups)"
        fi
    else
        print_warning "✗ DISPLAY variable not set"
        print_info "For GUI applications, set DISPLAY=:0 or configure remote display"
    fi
}

# Main test execution
main() {
    echo "==========================================="
    echo "   Docker Setup Test for Jetson Nano"
    echo "==========================================="
    echo ""
    
    test_docker
    test_nvidia_docker
    test_docker_compose
    test_camera_devices
    test_display
    test_python_deps
    # test_docker_build  # Commented out as it takes a long time
    
    echo ""
    echo "==========================================="
    echo "                 SUMMARY"
    echo "==========================================="
    print_info "Basic tests completed. For full functionality test:"
    echo ""
    echo "1. Build the application:"
    echo "   docker-compose -f $RECOMMENDED_COMPOSE build"
    echo ""
    echo "2. Test camera functionality:"
    echo "   docker-compose -f $RECOMMENDED_COMPOSE run --rm yolo-seed-detection python3 test_camera.py"
    echo ""
    echo "3. Run the full application:"
    echo "   docker-compose -f $RECOMMENDED_COMPOSE up"
    echo ""
    print_info "If you encounter repository/package errors, use:"
    echo "   docker-compose -f docker-compose-ultra-minimal.yml up --build"
}

# Run tests
main "$@"
