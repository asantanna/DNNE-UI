#!/bin/bash
# DNNE Test Suite Claude Commands
# Convenient commands for running DNNE tests with different configurations

set -e  # Exit on error

# Project root directory (where this script's parent directory is located)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if running from correct directory
check_project_root() {
    if [ ! -f "$PROJECT_ROOT/main.py" ] || [ ! -d "$PROJECT_ROOT/tests-dnne" ]; then
        log_error "Must run from DNNE-UI project root directory"
        log_error "Expected files: main.py, tests-dnne/"
        log_error "Current directory: $(pwd)"
        log_error "Project root: $PROJECT_ROOT"
        exit 1
    fi
}

# Activate conda environment
activate_environment() {
    log_info "Activating conda environment DNNE_PY38..."
    
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        log_error "Conda not found. Please install conda first."
        exit 1
    fi
    
    # Activate environment
    source /home/asantanna/miniconda/bin/activate DNNE_PY38 || {
        log_error "Failed to activate conda environment DNNE_PY38"
        log_error "Please ensure conda is installed and DNNE_PY38 environment exists"
        exit 1
    }
    
    log_success "Conda environment activated"
}

# Set default configuration
setup_test_config() {
    # Use existing data path to avoid re-downloads
    export DNNE_TEST_DATA_PATH="${DNNE_TEST_DATA_PATH:-$PROJECT_ROOT/data}"
    export DNNE_TEST_DOWNLOAD="${DNNE_TEST_DOWNLOAD:-true}"
    
    log_info "Test configuration:"
    log_info "  Data Path: $DNNE_TEST_DATA_PATH"
    log_info "  Download: $DNNE_TEST_DOWNLOAD"
    log_info "  Project Root: $PROJECT_ROOT"
}

# Check if test dependencies are installed
install_test_deps() {
    log_info "Ensuring test dependencies are installed..."
    pip install -q pytest pytest-timeout pytest-asyncio pytest-cov || {
        log_error "Failed to install test dependencies"
        exit 1
    }
    log_success "Test dependencies verified"
}

# Run dependency check
check_dependencies() {
    log_info "Checking dependencies..."
    cd "$PROJECT_ROOT"
    python tests-dnne/check_dependencies.py || {
        log_error "Dependency check failed!"
        log_error "Please install missing dependencies before running tests."
        exit 1
    }
    log_success "Dependencies check passed"
}

# Main test runner using the existing shell script
dnne_test_main() {
    local description="$1"
    log_info "🚀 Starting DNNE Test Suite: $description"
    echo "================================================================"
    
    check_project_root
    activate_environment
    setup_test_config
    install_test_deps
    check_dependencies
    
    echo ""
    log_info "Running test suite..."
    echo ""
    
    cd "$PROJECT_ROOT"
    bash tests-dnne/run_tests.sh
    local exit_code=$?
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        log_success "All tests completed successfully!"
    else
        log_error "Tests failed with exit code $exit_code"
    fi
    
    return $exit_code
}

# Pytest runner for specific test categories
dnne_test_pytest() {
    local description="$1"
    local pytest_args="$2"
    local timeout="$3"
    
    log_info "🧪 Running DNNE Tests: $description"
    echo "================================================================"
    
    check_project_root
    activate_environment
    setup_test_config
    install_test_deps
    check_dependencies
    
    echo ""
    log_info "Running pytest with: $pytest_args"
    log_info "Timeout: ${timeout}s per test"
    echo ""
    
    cd "$PROJECT_ROOT"
    pytest $pytest_args \
        --timeout=$timeout \
        --timeout-method=thread \
        -v \
        --tb=short \
        --no-header
    local exit_code=$?
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        log_success "Tests completed successfully!"
    else
        log_error "Tests failed with exit code $exit_code"
    fi
    
    return $exit_code
}

# Individual command functions

# Main command - run all tests
dnne_test() {
    dnne_test_main "Full Test Suite (Unit + Integration + Coverage)"
}

# Unit tests only
dnne_test_unit() {
    dnne_test_pytest "Unit Tests Only" "tests-dnne/unit/" "30"
}

# Integration tests only
dnne_test_integration() {
    dnne_test_pytest "Integration Tests Only" "tests-dnne/integration/" "120"
}

# Quick tests (unit tests only, shorter timeouts)
dnne_test_quick() {
    dnne_test_pytest "Quick Tests (unit only, 10s timeout)" "tests-dnne/unit/" "10"
}

# Full test suite (everything)
dnne_test_full() {
    dnne_test_main "Complete Test Suite (Unit + Integration + Coverage)"
}

# Coverage report
dnne_test_coverage() {
    dnne_test_pytest "Tests with Coverage Report" "tests-dnne/ --cov=custom_nodes --cov=export_system --cov-report=term-missing --cov-report=html" "60"
}

# ML tests only
dnne_test_ml() {
    dnne_test_pytest "ML Node Tests Only" "tests-dnne/ -m ml" "30"
}

# Robotics tests only
dnne_test_robotics() {
    dnne_test_pytest "Robotics Tests Only" "tests-dnne/ -m robotics" "60"
}

# Export system tests only
dnne_test_export() {
    dnne_test_pytest "Export System Tests Only" "tests-dnne/ -m export" "30"
}

# RL comprehensive tests (Cartpole PPO)
dnne_test_rl_comprehensive() {
    log_info "🎮 Running DNNE RL Comprehensive Tests (Cartpole PPO)"
    echo "================================================================"
    
    check_project_root
    activate_environment
    
    echo ""
    log_info "Running Cartpole PPO comprehensive test suite..."
    echo ""
    
    cd "$PROJECT_ROOT"
    python claude_scripts/test_cartpole_ppo_comprehensive.py
    local exit_code=$?
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        log_success "RL comprehensive tests completed successfully!"
    else
        log_error "RL comprehensive tests failed with exit code $exit_code"
    fi
    
    return $exit_code
}

# Checkpoint system tests
dnne_test_checkpoint() {
    log_info "🔐 Running DNNE Checkpoint System Tests"
    echo "================================================================"
    
    check_project_root
    activate_environment
    
    echo ""
    log_info "Running checkpoint system functionality tests..."
    echo ""
    
    cd "$PROJECT_ROOT"
    python claude_scripts/test_checkpoint_system.py
    local exit_code=$?
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        log_success "Checkpoint system tests completed successfully!"
    else
        log_error "Checkpoint system tests failed with exit code $exit_code"
    fi
    
    return $exit_code
}

# Inference mode tests
dnne_test_inference() {
    log_info "🔍 Running DNNE Inference Mode Tests"
    echo "================================================================"
    
    check_project_root
    activate_environment
    
    echo ""
    log_info "Running comprehensive training + inference tests..."
    echo ""
    
    cd "$PROJECT_ROOT"
    python claude_scripts/test_mnist_inference_complete.py
    local exit_code=$?
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        log_success "Inference mode tests completed successfully!"
    else
        log_error "Inference mode tests failed with exit code $exit_code"
    fi
    
    return $exit_code
}

# Cartpole performance benchmark tests
dnne_test_cartpole_performance() {
    log_info "🏎️  Running DNNE Cartpole Performance Benchmark"
    echo "================================================================"
    
    check_project_root
    activate_environment
    
    echo ""
    log_info "Running Cartpole performance benchmark vs IsaacGymEnvs..."
    echo ""
    
    cd "$PROJECT_ROOT"
    python claude_scripts/benchmark_cartpole_performance.py
    local exit_code=$?
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        log_success "Cartpole performance benchmark completed successfully!"
    else
        log_error "Cartpole performance benchmark failed with exit code $exit_code"
    fi
    
    return $exit_code
}

# Verbose mode
dnne_test_verbose() {
    dnne_test_pytest "Verbose Test Output" "tests-dnne/ -vvv -s --tb=long" "60"
}

# Dependencies check only
dnne_test_deps() {
    log_info "🔍 Checking DNNE Test Dependencies"
    echo "================================================================"
    
    check_project_root
    activate_environment
    setup_test_config
    
    cd "$PROJECT_ROOT"
    python tests-dnne/check_dependencies.py
    local exit_code=$?
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        log_success "All dependencies are satisfied!"
    else
        log_error "Some dependencies are missing"
    fi
    
    return $exit_code
}

# Help function
dnne_test_help() {
    echo "DNNE Test Suite Commands"
    echo "================================================================"
    echo ""
    echo "Main Commands:"
    echo "  dnne-test              Run all tests (unit + integration + coverage)"
    echo "  dnne-test-unit         Run only unit tests (fast, 30s timeout)"
    echo "  dnne-test-integration  Run only integration tests (slower, 2min timeout)"
    echo "  dnne-test-quick        Run tests with short timeout (10s, skip slow tests)"
    echo ""
    echo "Specialized Commands:"
    echo "  dnne-test-coverage            Run tests with coverage report"
    echo "  dnne-test-ml                  Run only ML node tests"
    echo "  dnne-test-robotics            Run only robotics/Isaac Gym tests"
    echo "  dnne-test-export              Run only export system tests"
    echo "  dnne-test-rl                  Run comprehensive RL tests (Cartpole PPO)"
    echo "  dnne-test-checkpoint          Run checkpoint system tests"
    echo "  dnne-test-cartpole-performance Run Cartpole performance benchmark vs IsaacGymEnvs"
    echo ""
    echo "Debug Commands:"
    echo "  dnne-test-verbose      Run with maximum verbosity"
    echo "  dnne-test-deps         Check dependencies only"
    echo "  dnne-test-help         Show this help"
    echo ""
    echo "Configuration:"
    echo "  DNNE_TEST_DATA_PATH    Set custom data directory (default: ./data)"
    echo "  DNNE_TEST_DOWNLOAD     Enable/disable downloads (default: true)"
    echo ""
    echo "Examples:"
    echo "  dnne-test                           # Run all tests"
    echo "  dnne-test-unit                      # Quick unit tests only"
    echo "  DNNE_TEST_DOWNLOAD=false dnne-test # Run without downloading data"
    echo ""
}

# Export functions for use in other scripts
export -f check_project_root
export -f activate_environment
export -f setup_test_config
export -f install_test_deps
export -f check_dependencies
export -f dnne_test_main
export -f dnne_test_pytest
export -f dnne_test
export -f dnne_test_full
export -f dnne_test_unit
export -f dnne_test_integration
export -f dnne_test_quick
export -f dnne_test_coverage
export -f dnne_test_ml
export -f dnne_test_robotics
export -f dnne_test_export
export -f dnne_test_rl_comprehensive
export -f dnne_test_checkpoint
export -f dnne_test_inference
export -f dnne_test_verbose
export -f dnne_test_deps
export -f dnne_test_help

# If script is run directly, show help
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    dnne_test_help
fi