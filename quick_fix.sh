#!/bin/bash
echo "🔧 Quick Fix for Docker Compose Version Issue"
echo "=============================================="

# Fix docker-compose.yml version
echo "📝 Updating docker-compose.yml..."
sed -i 's/version: .*/version: "2.3"/' docker-compose.yml

echo "✅ Fixed docker-compose.yml version"

# Make scripts executable
chmod +x run.sh
chmod +x verify.sh 2>/dev/null

echo "✅ Made scripts executable"

# Try to run again
echo ""
echo "🚀 Attempting to start seed detection..."
./run.sh
