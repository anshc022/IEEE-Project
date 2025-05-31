#!/bin/bash
# Enhanced Docker setup script for Jetson Nano with repository issue handling
# This script addresses common GPG key and repository problems

set -e  # Exit on any error

echo "=== Enhanced Jetson Nano Docker Setup ==="
echo "This script will handle repository issues and set up Docker for YOLOv11 seed detection"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Jetson Nano
check_jetson() {
    if [ -f /etc/nv_tegra_release ]; then
        print_status "Detected Jetson device"
        cat /etc/nv_tegra_release
    else
        print_warning "Not detected as Jetson device, continuing anyway..."
    fi
}

# Fix repository issues
fix_repositories() {
    print_status "Fixing repository and GPG key issues..."
    
    # Backup original sources
    sudo cp /etc/apt/sources.list /etc/apt/sources.list.backup.$(date +%Y%m%d_%H%M%S) || true
    
    # Remove problematic repository files
    sudo rm -f /etc/apt/sources.list.d/kitware*.list* || true
    sudo rm -f /etc/apt/sources.list.d/*cuda*.list.disabled || true
    
    # Clean package cache
    sudo apt-get clean
    sudo rm -rf /var/lib/apt/lists/*
    
    # Update package cache
    print_status "Updating package cache..."
    sudo apt-get update || {
        print_warning "Initial apt update failed, trying to fix..."
        sudo apt-get update --fix-missing || true
    }
}

# Install Docker if not present
install_docker() {
    if command -v docker &> /dev/null; then
        print_status "Docker already installed: $(docker --version)"
    else
        print_status "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
        print_status "Docker installed successfully"
    fi
}

# Install Docker Compose
install_docker_compose() {
    if command -v docker-compose &> /dev/null; then
        COMPOSE_VERSION=$(docker-compose --version | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')
        print_status "Docker Compose already installed: $COMPOSE_VERSION"
    else
        print_status "Installing Docker Compose..."
        # Install using pip for better compatibility on ARM64
        sudo apt-get install -y python3-pip
        sudo pip3 install docker-compose
        print_status "Docker Compose installed successfully"
    fi
}

# Check Docker Compose version and recommend appropriate compose file
check_compose_version() {
    if command -v docker-compose &> /dev/null; then
        COMPOSE_VERSION=$(docker-compose --version | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -1)
        MAJOR_VERSION=$(echo $COMPOSE_VERSION | cut -d. -f1)
        MINOR_VERSION=$(echo $COMPOSE_VERSION | cut -d. -f2)
        
        print_status "Detected Docker Compose version: $COMPOSE_VERSION"
        
        if [ "$MAJOR_VERSION" -ge 2 ] || ([ "$MAJOR_VERSION" -eq 1 ] && [ "$MINOR_VERSION" -ge 25 ]); then
            print_status "Using docker-compose-jetson-fixed.yml (recommended)"
            COMPOSE_FILE="docker-compose-jetson-fixed.yml"
        elif [ "$MAJOR_VERSION" -eq 1 ] && [ "$MINOR_VERSION" -ge 12 ]; then
            print_status "Using docker-compose-legacy.yml for older Docker Compose"
            COMPOSE_FILE="docker-compose-legacy.yml"
        else
            print_status "Using docker-compose-v1.yml for very old Docker Compose"
            COMPOSE_FILE="docker-compose-v1.yml"
        fi
    else
        print_error "Docker Compose not found"
        return 1
    fi
}

# Setup NVIDIA Docker runtime
setup_nvidia_docker() {
    print_status "Setting up NVIDIA Docker runtime..."
    
    # Check if nvidia-docker is already configured
    if docker info | grep -q nvidia; then
        print_status "NVIDIA Docker runtime already configured"
        return 0
    fi
    
    # Install nvidia-docker2 if not present
    if ! dpkg -l | grep -q nvidia-docker2; then
        print_status "Installing nvidia-docker2..."
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        
        sudo apt-get update
        sudo apt-get install -y nvidia-docker2
        sudo systemctl restart docker
        print_status "NVIDIA Docker runtime installed"
    fi
}

# Setup display forwarding
setup_display() {
    print_status "Setting up display forwarding..."
    
    # Add current user to video group
    sudo usermod -a -G video $USER
    
    # Set up X11 forwarding permissions
    xhost +local:docker || {
        print_warning "Could not set X11 permissions (xhost not available or no display)"
        print_status "For headless operation, this is normal"
    }
    
    # Export DISPLAY variable
    export DISPLAY=${DISPLAY:-:0}
    echo "export DISPLAY=${DISPLAY}" >> ~/.bashrc
}

# Create necessary directories
create_directories() {
    print_status "Creating project directories..."
    mkdir -p data logs models
    chmod 755 data logs models
}

# Main execution
main() {
    echo "Starting enhanced setup..."
    
    check_jetson
    fix_repositories
    install_docker
    install_docker_compose
    check_compose_version
    setup_nvidia_docker
    setup_display
    create_directories
    
    print_status "Setup completed successfully!"
    print_status "Recommended Docker Compose file: $COMPOSE_FILE"
    
    echo ""
    echo "=== Next Steps ==="
    echo "1. Log out and log back in to refresh group memberships"
    echo "2. Test Docker: docker run hello-world"
    echo "3. Test NVIDIA Docker: docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi"
    echo "4. Build the application: docker-compose -f $COMPOSE_FILE build"
    echo "5. Run the application: docker-compose -f $COMPOSE_FILE up"
    echo ""
    echo "Available Docker Compose files:"
    echo "  - docker-compose-jetson-fixed.yml (recommended, handles repository issues)"
    echo "  - docker-compose-legacy.yml (for older Docker Compose versions)"
    echo "  - docker-compose-v1.yml (for very old Docker Compose versions)"
    echo "  - docker-compose-ultra-minimal.yml (minimal dependencies only)"
    echo ""
    echo "For testing camera only: docker-compose -f $COMPOSE_FILE run --rm yolo-seed-detection python3 test_camera.py"
}

# Run main function
main "$@"
