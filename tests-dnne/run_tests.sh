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

# Check if running in correct directory
if [ ! -f "main.py" ] || [ ! -d "tests-dnne" ]; then
    echo "Error: Must run from DNNE-UI project root directory"
    exit 1
fi

# Activate conda environment
echo "Activating conda environment..."
source /home/asantanna/miniconda/bin/activate DNNE_PY38 || {
    echo "Failed to activate conda environment DNNE_PY38"
    echo "Please ensure conda is installed and DNNE_PY38 environment exists"
    exit 1
}

# Export configuration for pytest
export DNNE_TEST_DATA_PATH="$DATA_PATH"
export DNNE_TEST_DOWNLOAD="$DOWNLOAD"

# Check dependencies
echo ""
echo "Checking dependencies..."
python tests-dnne/check_dependencies.py || {
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
run_test_category "Unit Tests" "tests-dnne/unit/" "15m" "30" || all_passed=false

# Run integration tests - 2min timeout per test, 20min total  
run_test_category "Integration" "tests-dnne/integration/" "20m" "120" || all_passed=false

# Generate coverage report if all tests passed
if [ "$all_passed" = true ]; then
    echo "=== Generating Coverage Report ==="
    timeout 30m pytest tests-dnne/ \
        --timeout=60 \
        --cov=custom_nodes \
        --cov=export_system \
        --cov-report=term-missing \
        --cov-report=html \
        -q || {
        echo "Coverage report generation failed (tests may have still passed)"
    }
fi

echo ""
echo "=================================="
echo "Test Summary"
echo "=================================="

if [ "$all_passed" = true ]; then
    echo "✅ All DNNE tests passed!"
    echo ""
    echo "Coverage report available at: htmlcov/index.html"
    exit 0
else
    echo "❌ Some tests failed!"
    echo ""
    echo "Please check the output above for details."
    exit 1
fi