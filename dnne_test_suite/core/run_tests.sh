#!/bin/bash
# DNNE Test Runner with Timeout Protection
# Runs all DNNE tests with strict dependency checking and timeout protection

set -e  # Exit on error

# Configuration with defaults
DATA_PATH="${DNNE_TEST_DATA_PATH:-/mnt/e/ALS-Projects/DNNE/DNNE-UI/data}"
DOWNLOAD="${DNNE_TEST_DOWNLOAD:-true}"

echo "=================================="
echo "DNNE Test Suite Runner"
echo "=================================="
echo ""
echo "Configuration:"
echo "  Data Path: $DATA_PATH"
echo "  Download: $DOWNLOAD"
echo ""

# Change to project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Check if we're in the correct directory now
if [ ! -f "main.py" ] || [ ! -d "dnne_test_suite" ]; then
    echo "Error: Could not find project root directory"
    echo "Expected files: main.py, dnne_test_suite/"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Source dnne_config_reader helper
source "$(dirname "${BASH_SOURCE[0]}")/../../dnne_config_reader.sh"

# Get conda configuration from dnne_config
CONDA_PATH=$(get_dnne_config "paths.conda_path")
CONDA_ENV=$(get_dnne_config "paths.conda_env")

# Activate conda environment
echo "Activating conda environment..."
source $CONDA_PATH/bin/activate $CONDA_ENV || {
    echo "Failed to activate conda environment $CONDA_ENV"
    echo "Please ensure conda is installed and $CONDA_ENV environment exists"
    exit 1
}

# Export configuration for pytest
export DNNE_TEST_DATA_PATH="$DATA_PATH"
export DNNE_TEST_DOWNLOAD="$DOWNLOAD"

# Check dependencies
echo ""
echo "Checking dependencies..."
python dnne_test_suite/core/check_dependencies.py || {
    echo ""
    echo "Dependency check failed!"
    echo "Please install missing dependencies before running tests."
    exit 1
}

# Install test dependencies if needed
echo ""
echo "Ensuring test dependencies are installed..."
pip install -q pytest pytest-timeout pytest-asyncio pytest-cov || {
    echo "Failed to install test dependencies"
    exit 1
}

echo ""
echo "=================================="
echo "Running DNNE Tests"
echo "=================================="
echo ""

# Function to run tests with timeout
run_test_category() {
    local category=$1
    local path=$2
    local timeout=$3
    local test_timeout=$4
    
    echo "=== Running $category tests ==="
    echo "Path: $path"
    echo "Global timeout: $timeout, Per-test timeout: ${test_timeout}s"
    echo ""
    
    timeout $timeout pytest $path \
        --timeout=$test_timeout \
        --timeout-method=thread \
        -v \
        --tb=short \
        --no-header || {
        echo ""
        echo "❌ $category tests failed!"
        return 1
    }
    
    echo ""
    echo "✓ $category tests passed"
    echo ""
    return 0
}

# Track overall success
all_passed=true

# Run all unit tests - 30s timeout per test, 15min total
run_test_category "Unit Tests" "dnne_test_suite/core/unit/" "15m" "30" || all_passed=false

# Run integration tests - 2min timeout per test, 20min total  
run_test_category "Integration" "dnne_test_suite/core/integration/" "20m" "120" || all_passed=false

# Note: Coverage report generation removed from default test run
# Use 'dnne_test coverage' to run tests with coverage report

echo ""
echo "=================================="
echo "Test Summary"
echo "=================================="

if [ "$all_passed" = true ]; then
    echo "✅ All DNNE tests passed!"
    echo ""
    echo "To generate coverage report, run: dnne_test coverage"
    exit 0
else
    echo "❌ Some tests failed!"
    echo ""
    echo "Please check the output above for details."
    exit 1
fi