#!/bin/bash
# Build script for DNNE frontend
# This script builds the Vue.js frontend and handles the distribution files

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Building DNNE Frontend..."
echo "========================================="

# Navigate to frontend directory
FRONTEND_DIR="$HOME/DNNE/DNNE-UI-Frontend"
if [ ! -d "$FRONTEND_DIR" ]; then
    echo -e "${RED}✗ Frontend directory not found at $FRONTEND_DIR${NC}"
    exit 1
fi

cd "$FRONTEND_DIR"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    npm install
fi

# Run TypeScript check first
echo -e "${YELLOW}Step 1: Running TypeScript type checking...${NC}"
npm run typecheck
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ TypeScript check passed${NC}"
else
    echo -e "${RED}✗ TypeScript check FAILED!${NC}"
    echo -e "${RED}Fix the TypeScript errors above before the build can continue.${NC}"
    echo -e "${RED}Build ABORTED.${NC}"
    exit 1
fi

# Build the frontend
echo -e "${YELLOW}Step 2: Building with Vite...${NC}"
npm run build

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Vite build completed successfully${NC}"
    
    # Count the generated files
    JS_COUNT=$(find dist -name "*.js" -type f 2>/dev/null | wc -l)
    CSS_COUNT=$(find dist -name "*.css" -type f 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Distribution files created successfully${NC}"
    echo -e "${GREEN}  Created $JS_COUNT JavaScript files and $CSS_COUNT CSS files${NC}"
    
    echo ""
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}Frontend build completed successfully!${NC}"
    echo -e "${GREEN}=========================================${NC}"
    echo ""
    echo "The built files are in: $FRONTEND_DIR/dist/"
    echo "Server should serve from: ../DNNE-UI-Frontend/dist"
else
    echo -e "${RED}✗ Build FAILED!${NC}"
    exit 1
fi