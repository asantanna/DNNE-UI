#!/bin/bash
# Convenience script to build frontend
# Delegates to the main build script in DNNE-UI-Frontend

FRONTEND_DIR="$HOME/DNNE/DNNE-UI-Frontend"

if [ ! -d "$FRONTEND_DIR" ]; then
    echo "Error: Frontend directory not found at $FRONTEND_DIR"
    exit 1
fi

if [ ! -f "$FRONTEND_DIR/build_frontend.sh" ]; then
    echo "Error: Build script not found at $FRONTEND_DIR/build_frontend.sh"
    exit 1
fi

# Run the actual build script
exec "$FRONTEND_DIR/build_frontend.sh" "$@"