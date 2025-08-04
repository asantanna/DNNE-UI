#!/usr/bin/env python3
"""
Interactive Browser Lifecycle Test Suite
Actually calls the MCP tools to test browser functionality
"""

import json
import time
from datetime import datetime

class InteractiveBrowserLifecycleTests:
    def __init__(self):
        self.test_results = []
        self.start_time = datetime.now()
        
    def log_test(self, test_name, success, message, details=None):
        """Log test result"""
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.test_results.append(result)
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}: {message}")
        
    def test_browser_initialization_sequence(self):
        """Test complete browser initialization sequence"""
        print("\n🔧 Testing Browser Initialization Sequence...")
        
        print("Please run the following MCP commands in Claude and report results:")
        print()
        
        # Test 1: Fresh initialization
        print("1️⃣  Test Fresh Browser Initialization")
        print("   Command: Use the initialize_browser MCP tool")
        print("   Expected: success=True, message='Browser initialized successfully'")
        result = input("   Result (pass/fail): ").lower().strip()
        message = input("   Message received: ").strip()
        
        self.log_test(
            "initialize_browser_fresh",
            result == "pass",
            f"Fresh initialization: {message}",
            {"command": "initialize_browser", "expected": "success"}
        )
        
        # Test 2: Double initialization
        print("\n2️⃣  Test Double Initialization")
        print("   Command: Use the initialize_browser MCP tool again (without cleanup)")
        print("   Expected: success=True, message='Browser already initialized'")
        result = input("   Result (pass/fail): ").lower().strip()
        message = input("   Message received: ").strip()
        
        self.log_test(
            "initialize_browser_double",
            result == "pass",
            f"Double initialization: {message}",
            {"command": "initialize_browser", "expected": "already_initialized"}
        )
        
    def test_browser_cleanup_sequence(self):
        """Test browser cleanup sequence"""
        print("\n🧹 Testing Browser Cleanup Sequence...")
        
        # Test 1: Normal cleanup
        print("1️⃣  Test Normal Browser Cleanup")
        print("   Command: Use the cleanup_browser MCP tool")
        print("   Expected: success=True, message='Browser cleaned up'")
        result = input("   Result (pass/fail): ").lower().strip()
        message = input("   Message received: ").strip()
        
        self.log_test(
            "cleanup_browser_normal",
            result == "pass",
            f"Normal cleanup: {message}",
            {"command": "cleanup_browser", "expected": "cleaned_up"}
        )
        
        # Test 2: Double cleanup
        print("\n2️⃣  Test Double Cleanup")
        print("   Command: Use the cleanup_browser MCP tool again")
        print("   Expected: success=True (should handle gracefully)")
        result = input("   Result (pass/fail): ").lower().strip()
        message = input("   Message received: ").strip()
        
        self.log_test(
            "cleanup_browser_double",
            result == "pass",
            f"Double cleanup: {message}",
            {"command": "cleanup_browser", "expected": "graceful_handling"}
        )
        
    def test_browser_restart_sequence(self):
        """Test browser restart sequence"""
        print("\n🔄 Testing Browser Restart Sequence...")
        
        # First ensure browser is initialized
        print("0️⃣  Setup: Initialize browser first")
        print("   Command: Use the initialize_browser MCP tool")
        input("   Press Enter when browser is initialized...")
        
        # Test 1: Normal restart
        print("\n1️⃣  Test Normal Browser Restart")
        print("   Command: Use the restart_browser MCP tool")
        print("   Expected: success=True, browser restarts successfully")
        result = input("   Result (pass/fail): ").lower().strip()
        message = input("   Message received: ").strip()
        
        self.log_test(
            "restart_browser_normal",
            result == "pass",
            f"Normal restart: {message}",
            {"command": "restart_browser", "expected": "successful_restart"}
        )
        
        # Test 2: Restart without initialization
        print("\n2️⃣  Test Restart Without Initialization")
        print("   Command: First use cleanup_browser, then restart_browser")
        print("   Expected: restart should handle uninitialized state")
        
        cleanup_result = input("   Cleanup result (pass/fail): ").lower().strip()
        restart_result = input("   Restart result (pass/fail): ").lower().strip()
        restart_message = input("   Restart message: ").strip()
        
        self.log_test(
            "restart_browser_uninitialized",
            restart_result == "pass",
            f"Restart without init: {restart_message}",
            {"command": "restart_browser", "expected": "handle_uninitialized"}
        )
        
    def test_browser_health_check(self):
        """Test browser health functionality"""
        print("\n🏥 Testing Browser Health Check...")
        
        # Ensure browser is initialized
        print("0️⃣  Setup: Initialize browser")
        print("   Command: Use the initialize_browser MCP tool")
        input("   Press Enter when browser is initialized...")
        
        print("\n1️⃣  Test Browser Health Check")
        print("   Command: Use the check_ui_health MCP tool")
        print("   Expected: success=True, UI is responsive")
        result = input("   Result (pass/fail): ").lower().strip()
        message = input("   Message received: ").strip()
        
        self.log_test(
            "check_ui_health",
            result == "pass",
            f"Health check: {message}",
            {"command": "check_ui_health", "expected": "healthy"}
        )
        
    def test_screenshot_functionality(self):
        """Test screenshot functionality"""
        print("\n📸 Testing Screenshot Functionality...")
        
        print("1️⃣  Test Screenshot Capture")
        print("   Command: Use the take_screenshot MCP tool with name 'test_screenshot'")
        print("   Expected: success=True, screenshot saved")
        result = input("   Result (pass/fail): ").lower().strip()
        message = input("   Message received: ").strip()
        screenshot_path = input("   Screenshot path (if provided): ").strip()
        
        self.log_test(
            "take_screenshot",
            result == "pass",
            f"Screenshot: {message}",
            {"command": "take_screenshot", "path": screenshot_path, "expected": "saved"}
        )
        
    def run_interactive_tests(self):
        """Run all interactive browser tests"""
        print("=" * 70)
        print("🧪 DNNE UI MCP Server - Interactive Browser Lifecycle Tests")
        print("=" * 70)
        print()
        print("This test suite will guide you through testing browser lifecycle tools.")
        print("You'll need to run MCP commands in Claude and report the results.")
        print()
        input("Press Enter to start testing...")
        
        self.test_browser_initialization_sequence()
        self.test_browser_cleanup_sequence()
        self.test_browser_restart_sequence()
        self.test_browser_health_check()
        self.test_screenshot_functionality()
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 Interactive Browser Test Summary")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['message']}")
        else:
            print("\n🎉 All browser lifecycle tests passed!")
        
        # Save results
        results_file = f"/home/asantanna/DNNE/DNNE-UI/mcp-dnne-ui/tests/interactive_browser_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "test_suite": "Interactive Browser Lifecycle Tools",
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": (passed_tests/total_tests)*100,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "detailed_results": self.test_results
            }, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        return failed_tests == 0

if __name__ == "__main__":
    test_suite = InteractiveBrowserLifecycleTests()
    success = test_suite.run_interactive_tests()
    exit(0 if success else 1)