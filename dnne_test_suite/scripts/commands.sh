#!/bin/bash
# DNNE Test Suite Claude Commands
# Convenient commands for running DNNE tests with different configurations

# Removed set -e to prevent early exit during test execution

# Project root directory (where this script's parent directory is located)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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
    if [ ! -f "$PROJECT_ROOT/main.py" ] || [ ! -d "$PROJECT_ROOT/dnne_test_suite" ]; then
        log_error "Must run from DNNE-UI project root directory"
        log_error "Expected files: main.py, dnne_test_suite/"
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
    
    # Source dnne_config_reader helper
    source "$(dirname "${BASH_SOURCE[0]}")/../../dnne_config_reader.sh"
    
    # Get conda configuration from dnne_config
    CONDA_PATH=$(get_dnne_config "exported.conda.conda_path")
    CONDA_ENV=$(get_dnne_config "exported.conda.conda_env")
    
    # Activate environment
    source $CONDA_PATH/bin/activate $CONDA_ENV || {
        log_error "Failed to activate conda environment $CONDA_ENV"
        log_error "Please ensure conda is installed and $CONDA_ENV environment exists"
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
    python dnne_test_suite/core/check_dependencies.py || {
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
    bash dnne_test_suite/core/run_tests.sh
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
    dnne_test_main "Full Test Suite (Unit + Integration)"
}

# Unit tests only
dnne_test_unit() {
    dnne_test_pytest "Unit Tests Only" "dnne_test_suite/core/unit/" "30"
}

# Integration tests only
dnne_test_integration() {
    dnne_test_pytest "Integration Tests Only" "dnne_test_suite/core/integration/" "120"
}

# Quick tests (unit tests only, shorter timeouts)
dnne_test_quick() {
    dnne_test_pytest "Quick Tests (unit only, 10s timeout)" "dnne_test_suite/core/unit/" "10"
}

# Full test suite (everything)
dnne_test_full() {
    log_info "🚀 Running Full DNNE Test Suite"
    echo "================================================================"
    
    # Run main tests (unit + integration)
    dnne_test_main "Complete Test Suite (Unit + Integration)"
    local main_result=$?
    
    # Check if agent server is available for telemetry test
    echo ""
    log_info "Checking for telemetry test availability..."
    if nc -z 172.22.160.1 8768 2>/dev/null; then
        log_info "Running telemetry tests..."
        dnne_test_telemetry
        local telemetry_result=$?
        
        if [ $telemetry_result -ne 0 ]; then
            log_warning "Telemetry tests failed or skipped"
            # Don't fail the whole suite if telemetry tests fail
            # since they require special setup
        fi
    else
        log_info "Skipping telemetry tests (agent server test port not available)"
        log_info "To enable: restart agent server with --enable-test-port"
    fi
    
    # Return the main test result
    return $main_result
}

# Coverage report
dnne_test_coverage() {
    dnne_test_pytest "Tests with Coverage Report" "dnne_test_suite/core/ --cov=custom_nodes --cov=export_system --cov-report=term-missing --cov-report=html" "60"
}

# ML tests only
dnne_test_ml() {
    dnne_test_pytest "ML Node Tests Only" "dnne_test_suite/core/ -m ml" "30"
}

# Robotics tests only
dnne_test_robotics() {
    dnne_test_pytest "Robotics Tests Only" "dnne_test_suite/core/ -m robotics" "60"
}

# Export system tests only
dnne_test_export() {
    dnne_test_pytest "Export System Tests Only" "dnne_test_suite/core/ -m export" "30"
}

# DNNE Agent tests
dnne_test_agent() {
    log_info "🤖 Running DNNE Agent System Tests"
    echo "================================================================"
    
    check_project_root
    activate_environment
    setup_test_config
    
    log_info "Running agent system tests..."
    echo ""
    
    cd "$PROJECT_ROOT"
    python dnne_test_suite/specialized/dnne_agent_test.py
    local exit_code=$?
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        log_success "Agent system tests completed successfully!"
    elif [ $exit_code -eq 1 ]; then
        log_warning "Agent server not running - please start dnne_agent_server.py on Windows"
    else
        log_error "Agent system tests failed!"
    fi
    
    return $exit_code
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
    python dnne_test_suite/specialized/cartpole_ppo_comprehensive.py
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
    python dnne_test_suite/specialized/checkpoint_system.py
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
    python dnne_test_suite/specialized/mnist_inference_complete.py
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

# Performance comparison test
dnne_test_performance() {
    log_info "📊 Running DNNE Performance Comparison Test"
    echo "================================================================"
    
    check_project_root
    activate_environment
    
    echo ""
    
    # Parse command line arguments for epochs and visual flags
    local epochs_flag=""
    local visual_flag=""
    local epochs_value=""
    
    # Parse all arguments passed to the function
    while [[ $# -gt 0 ]]; do
        case $1 in
            --epochs)
                epochs_value="$2"
                epochs_flag="--epochs $2"
                shift 2
                ;;
            --visual)
                visual_flag="--visual"
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    # Use default epochs if not specified
    if [ -z "$epochs_value" ]; then
        epochs_value="40"
        epochs_flag="--epochs 40"
    fi
    
    # Log what we're running
    if [ -n "$visual_flag" ]; then
        log_info "Running performance profiler (detailed mode, $epochs_value epochs, VISUAL MODE)..."
    else
        log_info "Running performance profiler (detailed mode, $epochs_value epochs)..."
    fi
    
    echo ""
    
    cd "$PROJECT_ROOT"
    
    # Run performance profiler and capture output
    local output_file="/tmp/dnne_performance_test_output.txt"
    python claude_scripts/profiling/performance_profiler.py --mode detailed $epochs_flag $visual_flag 2>&1 | tee "$output_file"
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        log_error "Performance profiler failed with exit code $exit_code"
        return $exit_code
    fi
    
    # Extract relative performance value from output
    local relative_perf=$(grep -oP "Relative Performance: \K[0-9]+\.[0-9]+" "$output_file" | tail -1)
    
    if [ -z "$relative_perf" ]; then
        log_error "Could not extract relative performance value from output"
        return 1
    fi
    
    echo ""
    log_info "Relative Performance: ${relative_perf}x"
    
    # Validate performance is within acceptable range (0.5 < x < 1.5)
    local perf_valid=$(python -c "print('1' if 0.5 < $relative_perf < 1.5 else '0')")
    
    echo ""
    if [ "$perf_valid" = "1" ]; then
        log_success "Performance test PASSED! Relative performance (${relative_perf}x) is within acceptable range (0.5 - 1.5)"
        rm -f "$output_file"
        return 0
    else
        log_error "Performance test FAILED! Relative performance (${relative_perf}x) is outside acceptable range (0.5 - 1.5)"
        rm -f "$output_file"
        return 1
    fi
}

# Verbose mode
dnne_test_verbose() {
    dnne_test_pytest "Verbose Test Output" "dnne_test_suite/core/ -vvv -s --tb=long" "60"
}

# Dependencies check only
dnne_test_deps() {
    log_info "🔍 Checking DNNE Test Dependencies"
    echo "================================================================"
    
    check_project_root
    activate_environment
    setup_test_config
    
    cd "$PROJECT_ROOT"
    python dnne_test_suite/core/check_dependencies.py
    local exit_code=$?
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        log_success "All dependencies are satisfied!"
    else
        log_error "Some dependencies are missing"
    fi
    
    return $exit_code
}

# Telemetry tests
dnne_test_telemetry() {
    log_info "📊 Running DNNE Telemetry Tests"
    echo "================================================================"
    
    check_project_root
    activate_environment
    
    # Check if agent server is running with test port
    echo "Checking agent server test port..."
    if ! nc -z 172.22.160.1 8768 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Agent server test port not accessible${NC}"
        echo "Make sure agent server is running with --enable-test-port"
        echo ""
        echo "To restart with test port:"
        echo "  Use MCP: mcp__dnne-ui__util_restart_dnne with agent_server_extra_args='--enable-test-port'"
        echo ""
        return 1
    fi
    
    echo -e "${GREEN}✅ Agent server test port is accessible${NC}"
    echo ""
    
    # Track results
    local PASSED=0
    local FAILED=0
    
    # Function to run a test
    run_telemetry_test() {
        local test_type=$1
        local description=$2
        
        echo -e "${YELLOW}Running $test_type test: $description${NC}"
        echo "-------------------------------------"
        
        if python dnne_test_suite/specialized/test_telemetry.py --test-type "$test_type"; then
            echo -e "${GREEN}✅ $test_type test PASSED${NC}"
            echo ""
            ((PASSED++))
        else
            echo -e "${RED}❌ $test_type test FAILED${NC}"
            echo ""
            ((FAILED++))
        fi
        
        # Small delay between tests
        sleep 2
    }
    
    # Run all telemetry tests
    run_telemetry_test "basic" "Core telemetry pipeline with SUMMARY validation"
    
    # Note: Long test takes 40 seconds
    echo -e "${YELLOW}Note: Long test will take ~40 seconds...${NC}"
    run_telemetry_test "long" "35-second aggregation interval test"
    
    run_telemetry_test "ratelimit" "Violation rate limiting (10/sec) test"
    
    run_telemetry_test "aggregation" "Telemetry aggregation test"
    
    # Run overhead test (separate script)
    echo -e "${YELLOW}Running OVERHEAD test: Performance impact measurement${NC}"
    echo "-------------------------------------"
    if python dnne_test_suite/specialized/telemetry_overhead_test.py; then
        echo -e "${GREEN}✅ OVERHEAD test PASSED${NC}"
        echo ""
        ((PASSED++))
    else
        echo -e "${RED}❌ OVERHEAD test FAILED${NC}"
        echo ""
        ((FAILED++))
    fi
    
    # Summary
    echo "====================================="
    echo "Telemetry Test Suite Summary"
    echo "====================================="
    echo -e "${GREEN}Passed: $PASSED${NC}"
    echo -e "${RED}Failed: $FAILED${NC}"
    
    if [ $FAILED -eq 0 ]; then
        echo -e "\n${GREEN}✅ All telemetry tests passed!${NC}"
        return 0
    else
        echo -e "\n${RED}❌ Some tests failed. Please review the output above.${NC}"
        return 1
    fi
}

# Help function
dnne_test_help() {
    echo "DNNE Test Suite Commands"
    echo "================================================================"
    echo ""
    echo "Main Commands:"
    echo "  dnne_test              Run all tests (unit + integration + coverage)"
    echo "  dnne_test-unit         Run only unit tests (fast, 30s timeout)"
    echo "  dnne_test-integration  Run only integration tests (slower, 2min timeout)"
    echo "  dnne_test-quick        Run tests with short timeout (10s, skip slow tests)"
    echo ""
    echo "Specialized Commands:"
    echo "  dnne_test-coverage            Run tests with coverage report"
    echo "  dnne_test-ml                  Run only ML node tests"
    echo "  dnne_test-robotics            Run only robotics/Isaac Gym tests"
    echo "  dnne_test-export              Run only export system tests"
    echo "  dnne_test-rl                  Run comprehensive RL tests (Cartpole PPO)"
    echo "  dnne_test-checkpoint          Run checkpoint system tests"
    echo "  dnne_test-cartpole-performance Run Cartpole performance benchmark vs IsaacGymEnvs"
    echo ""
    echo "Debug Commands:"
    echo "  dnne_test-verbose      Run with maximum verbosity"
    echo "  dnne_test-deps         Check dependencies only"
    echo "  dnne_test-help         Show this help"
    echo ""
    echo "Configuration:"
    echo "  DNNE_TEST_DATA_PATH    Set custom data directory (default: ./data)"
    echo "  DNNE_TEST_DOWNLOAD     Enable/disable downloads (default: true)"
    echo ""
    echo "Examples:"
    echo "  dnne_test                           # Run all tests"
    echo "  dnne_test-unit                      # Quick unit tests only"
    echo "  DNNE_TEST_DOWNLOAD=false dnne_test # Run without downloading data"
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
export -f dnne_test_agent
export -f dnne_test_rl_comprehensive
export -f dnne_test_checkpoint
export -f dnne_test_inference
export -f dnne_test_verbose
export -f dnne_test_deps
export -f dnne_test_telemetry
export -f dnne_test_help

# If script is run directly, show help
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    dnne_test_help
fi