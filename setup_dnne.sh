#!/bin/bash
# setup_dnne.sh - Quick DNNE setup script for custom installations
# Usage: ./setup_dnne.sh [base_directory]
# Example: ./setup_dnne.sh /workspace

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get base directory from argument or use current directory
BASE_DIR="${1:-$(pwd)}"

echo -e "${GREEN}DNNE Configuration Setup Script${NC}"
echo "================================="
echo "Base directory: $BASE_DIR"
echo ""

# Check if running from DNNE-UI directory
if [ -f "dnne_config.py" ]; then
    DNNE_UI_DIR=$(pwd)
else
    DNNE_UI_DIR="$BASE_DIR/DNNE-UI"
fi

# Function to check if path exists
check_path() {
    if [ -e "$1" ]; then
        echo -e "${GREEN}✓${NC} Found: $1"
        return 0
    else
        echo -e "${RED}✗${NC} Not found: $1"
        return 1
    fi
}

# Detect conda installation
detect_conda() {
    if command -v conda &> /dev/null; then
        CONDA_PATH=$(dirname $(dirname $(which conda)))
        echo -e "${GREEN}✓${NC} Conda found at: $CONDA_PATH"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} Conda not found in PATH"
        return 1
    fi
}

echo "Checking for required components..."
echo ""

# Check for repositories
echo "Checking repositories:"
check_path "$DNNE_UI_DIR" || { echo -e "${RED}Please clone DNNE-UI repository first${NC}"; exit 1; }
check_path "$BASE_DIR/DNNE-LINUX-SUPPORT" || echo -e "${YELLOW}Optional: DNNE-LINUX-SUPPORT not found${NC}"
check_path "$BASE_DIR/DNNE-LINUX-SUPPORT/IsaacGymEnvs" || echo -e "${YELLOW}Optional: IsaacGymEnvs not found${NC}"
check_path "$BASE_DIR/DNNE-LINUX-SUPPORT/isaacgym" || echo -e "${YELLOW}Optional: Isaac Gym not found${NC}"
check_path "$BASE_DIR/DNNE-LINUX-SUPPORT/rl_games_dnne" || echo -e "${YELLOW}Optional: rl_games_dnne not found${NC}"

echo ""
echo "Checking conda:"
if detect_conda; then
    # Check for DNNE environment
    if conda env list | grep -q "DNNE_PY38"; then
        echo -e "${GREEN}✓${NC} DNNE_PY38 environment exists"
    else
        echo -e "${YELLOW}⚠${NC} DNNE_PY38 environment not found"
        echo "  Create it with: conda create -n DNNE_PY38 python=3.8"
    fi
else
    CONDA_PATH="/home/$USER/miniconda3"
    echo "  Using default: $CONDA_PATH"
fi

echo ""
echo "Creating configuration file..."

# Create .dnne directory
mkdir -p ~/.dnne

# Generate configuration
cat > ~/.dnne/config.json << EOF
{
  "paths": {
    "dnne_root": "$DNNE_UI_DIR",
    "linux_support": "$BASE_DIR/DNNE-LINUX-SUPPORT",
    "isaac_gym_envs": "$BASE_DIR/DNNE-LINUX-SUPPORT/IsaacGymEnvs",
    "isaac_gym": "$BASE_DIR/DNNE-LINUX-SUPPORT/isaacgym",
    "rl_games_dnne": "$BASE_DIR/DNNE-LINUX-SUPPORT/rl_games_dnne",
    "conda_path": "$CONDA_PATH",
    "conda_env": "DNNE_PY38"
  },
  "export": {
    "default_workflow": "Cartpole_PPO",
    "workflow_path": "user/default/workflows",
    "export_base": "export_system/exports"
  },
  "profiling": {
    "default_num_envs": 512,
    "default_timeout": 300,
    "temp_directory": "/tmp"
  }
}
EOF

echo -e "${GREEN}✓${NC} Configuration written to ~/.dnne/config.json"

# Test configuration if in DNNE-UI directory
if [ -f "$DNNE_UI_DIR/dnne_config.py" ]; then
    echo ""
    echo "Testing configuration..."
    cd "$DNNE_UI_DIR"
    
    # Check if conda environment is activated
    if [[ "$CONDA_DEFAULT_ENV" == "DNNE_PY38" ]]; then
        python -c "from dnne_config import DNNEConfig; c = DNNEConfig(); print('Configuration loaded successfully!')" && \
        echo -e "${GREEN}✓${NC} Configuration test passed" || \
        echo -e "${RED}✗${NC} Configuration test failed"
    else
        echo -e "${YELLOW}⚠${NC} Activate DNNE_PY38 environment to test configuration"
        echo "  Run: conda activate DNNE_PY38"
    fi
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review and edit ~/.dnne/config.json if needed"
echo "2. Activate conda environment: conda activate DNNE_PY38"
echo "3. Install dependencies:"
echo "   pip install -r requirements.txt           # Core dependencies"
echo "   pip install -r requirements-robotics.txt  # For Isaac Gym support"
echo "   pip install -r requirements-dev.txt       # For development (optional)"
echo "4. Run tests: cd $DNNE_UI_DIR && ./dnne_test full"
echo ""
echo "For more information, see CONFIGURATION_GUIDE.md"