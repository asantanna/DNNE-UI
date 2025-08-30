#!/bin/bash
# Build script for DNNE frontend - delegates to the actual build script in frontend directory

# Navigate to frontend directory and run the build script there
FRONTEND_DIR="$HOME/DNNE/DNNE-UI-Frontend"

if [ ! -d "$FRONTEND_DIR" ]; then
    echo "Error: Frontend directory not found at $FRONTEND_DIR"
    exit 1
fi

cd "$FRONTEND_DIR"

# Run the actual build script
./build_frontend.sh "$@"