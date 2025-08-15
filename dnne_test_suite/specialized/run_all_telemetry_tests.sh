#!/bin/bash
# Run all telemetry tests

echo "====================================="
echo "DNNE Telemetry Test Suite"
echo "====================================="
echo

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0

# Function to run a test
run_test() {
    local test_type=$1
    local description=$2
    
    echo -e "${YELLOW}Running $test_type test: $description${NC}"
    echo "-------------------------------------"
    
    if python test_telemetry.py --test-type "$test_type"; then
        echo -e "${GREEN}✅ $test_type test PASSED${NC}\n"
        ((PASSED++))
    else
        echo -e "${RED}❌ $test_type test FAILED${NC}\n"
        ((FAILED++))
    fi
    
    # Small delay between tests
    sleep 2
}

# Run all tests
run_test "basic" "Core telemetry pipeline with SUMMARY validation"

# Note: Long test takes 40 seconds
echo -e "${YELLOW}Note: Long test will take ~40 seconds...${NC}"
run_test "long" "35-second aggregation interval test"

run_test "ratelimit" "Violation rate limiting (10/sec) test"

# Run overhead test (separate script)
echo -e "${YELLOW}Running OVERHEAD test: Performance impact measurement${NC}"
echo "-------------------------------------"
if python /home/asantanna/DNNE/DNNE-UI/dnne_test_suite/specialized/telemetry_overhead_test.py; then
    echo -e "${GREEN}✅ OVERHEAD test PASSED${NC}\n"
    ((PASSED++))
else
    echo -e "${RED}❌ OVERHEAD test FAILED${NC}\n"
    ((FAILED++))
fi

# Summary
echo "====================================="
echo "Test Suite Summary"
echo "====================================="
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}✅ All telemetry tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}❌ Some tests failed. Please review the output above.${NC}"
    exit 1
fi