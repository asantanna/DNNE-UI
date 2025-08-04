#!/usr/bin/env python3
"""Comprehensive test suite for DNNE UI MCP Server"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dnne_ui_mcp_server import DNNE_UI_MCPServer
from browser_controller import BrowserController
from utils.helpers import format_mcp_response
from utils.state_manager import StateManager

class TestResult:
    """Track test results"""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.duration = 0
        
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        result = f"{status} - {self.name} ({self.duration:.2f}s)"
        if self.error:
            result += f"\n    Error: {self.error}"
        return result

class DNNEMCPTestSuite:
    """Test suite for DNNE UI MCP Server"""
    
    def __init__(self, run_browser_tests: bool = False):
        self.server = None
        self.browser = None
        self.run_browser_tests = run_browser_tests
        self.results: List[TestResult] = []
        
    async def setup(self):
        """Set up test environment"""
        print("Setting up test environment...")
        self.server = DNNE_UI_MCPServer()
        
        if self.run_browser_tests:
            self.browser = BrowserController()
            await self.browser.initialize()
            self.server.browser_controller = self.browser
            # Update error diagnostics
            self.server.error_diagnostics.browser = self.browser
    
    async def teardown(self):
        """Clean up test environment"""
        print("Cleaning up test environment...")
        if self.browser:
            await self.browser.cleanup()
        
        # Clear test state
        if self.server and self.server.state_manager:
            self.server.state_manager.clear_session_state()
    
    async def run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test with timing and error handling"""
        result = TestResult(test_name)
        print(f"\nRunning: {test_name}")
        
        start_time = asyncio.get_event_loop().time()
        try:
            await test_func()
            result.passed = True
        except Exception as e:
            result.error = str(e)
            print(f"  Error: {e}")
        finally:
            result.duration = asyncio.get_event_loop().time() - start_time
        
        self.results.append(result)
        return result
    
    # Core functionality tests
    async def test_server_initialization(self):
        """Test server can be initialized"""
        assert self.server is not None
        assert self.server.server is not None
        assert self.server.state_manager is not None
        print("  ✓ Server initialized with all components")
    
    async def test_state_management(self):
        """Test state persistence and recovery"""
        # Test state updates
        self.server.state_manager.update("test_key", "test_value")
        assert self.server.state_manager.get("test_key") == "test_value"
        print("  ✓ State update works")
        
        # Test counter
        count = self.server.state_manager.increment_counter("test_counter")
        assert count == 1
        count = self.server.state_manager.increment_counter("test_counter")
        assert count == 2
        print("  ✓ Counter increment works")
        
        # Test operation recording
        self.server.state_manager.record_operation("test_op", success=True)
        last_op = self.server.state_manager.get("last_operation")
        assert last_op["name"] == "test_op"
        assert last_op["success"] == True
        print("  ✓ Operation recording works")
    
    async def test_error_handling(self):
        """Test error diagnostics and handling"""
        from utils.error_handler import ErrorDiagnostics, RecoverableError
        
        diag = ErrorDiagnostics()
        
        # Test error formatting
        error = RecoverableError("Test error")
        response = diag.format_error_response(error, "test_operation")
        
        assert response["success"] == False
        assert response["error"] == "Test error"
        assert response["error_type"] == "RecoverableError"
        assert response["operation"] == "test_operation"
        print("  ✓ Error formatting works")
        
        # Test troubleshooting suggestions
        connection_error = Exception("Connection failed")
        suggestions = diag.get_troubleshooting_suggestions(
            connection_error, 
            {"ui_state": {"has_dialog": False}}
        )
        assert len(suggestions) > 0
        assert any("DNNE server" in s for s in suggestions)
        print("  ✓ Troubleshooting suggestions generated")
    
    async def test_retry_logic(self):
        """Test retry with exponential backoff"""
        from utils.helpers import retry_with_backoff
        
        attempt_count = 0
        
        async def failing_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Test failure")
            return "success"
        
        result = await retry_with_backoff(
            failing_func,
            max_retries=3,
            initial_delay=0.1,
            retry_on=(ConnectionError,)
        )
        
        assert result == "success"
        assert attempt_count == 3
        print(f"  ✓ Retry logic works (took {attempt_count} attempts)")
    
    # Browser-dependent tests
    async def test_browser_health_check(self):
        """Test browser health monitoring"""
        if not self.browser:
            print("  ⚠ Skipped (browser tests disabled)")
            return
        
        healthy = await self.browser.health_check()
        assert healthy == True
        print("  ✓ Browser health check passed")
    
    async def test_browser_restart(self):
        """Test browser restart capability"""
        if not self.browser:
            print("  ⚠ Skipped (browser tests disabled)")
            return
        
        # Restart browser
        success = await self.browser.restart_browser()
        assert success == True
        
        # Verify it's working after restart
        healthy = await self.browser.health_check()
        assert healthy == True
        print("  ✓ Browser restart successful")
    
    async def test_screenshot_capture(self):
        """Test screenshot functionality"""
        if not self.browser:
            print("  ⚠ Skipped (browser tests disabled)")
            return
        
        screenshot_path = await self.browser.take_screenshot("test_screenshot")
        assert screenshot_path is not None
        assert Path(screenshot_path).exists()
        print(f"  ✓ Screenshot saved: {screenshot_path}")
    
    async def test_dialog_handling(self):
        """Test unexpected dialog handling"""
        if not self.browser:
            print("  ⚠ Skipped (browser tests disabled)")
            return
        
        # This would need a way to trigger a dialog
        # For now, just test the method exists
        handled = await self.browser.handle_unexpected_dialog()
        # Should be False since no dialog is present
        assert handled == False
        print("  ✓ Dialog handler works (no dialog present)")
    
    async def test_ui_element_detection(self):
        """Test UI element visibility checks"""
        if not self.browser:
            print("  ⚠ Skipped (browser tests disabled)")
            return
        
        # Check for sidebar buttons
        visible = await self.browser.is_visible(".side-bar-button")
        print(f"  ✓ Sidebar detection: {visible}")
        
        # Check for menu
        menu_visible = await self.browser.is_visible(".comfyui-menu")
        print(f"  ✓ Menu detection: {menu_visible}")
    
    async def run_all_tests(self):
        """Run all tests and report results"""
        print("\n" + "="*60)
        print("DNNE UI MCP Server Test Suite")
        print("="*60)
        
        await self.setup()
        
        try:
            # Core tests (always run)
            await self.run_test("Server Initialization", self.test_server_initialization)
            await self.run_test("State Management", self.test_state_management)
            await self.run_test("Error Handling", self.test_error_handling)
            await self.run_test("Retry Logic", self.test_retry_logic)
            
            # Browser tests (optional)
            await self.run_test("Browser Health Check", self.test_browser_health_check)
            await self.run_test("Browser Restart", self.test_browser_restart)
            await self.run_test("Screenshot Capture", self.test_screenshot_capture)
            await self.run_test("Dialog Handling", self.test_dialog_handling)
            await self.run_test("UI Element Detection", self.test_ui_element_detection)
            
        finally:
            await self.teardown()
        
        # Report results
        print("\n" + "="*60)
        print("Test Results Summary")
        print("="*60)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        
        for result in self.results:
            print(result)
        
        print("\n" + "-"*60)
        print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
        
        if failed == 0:
            print("✅ All tests passed!")
        else:
            print(f"❌ {failed} test(s) failed")
        
        return failed == 0

async def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DNNE UI MCP Server")
    parser.add_argument(
        "--with-browser",
        action="store_true",
        help="Run browser-dependent tests (requires DNNE server)"
    )
    args = parser.parse_args()
    
    if args.with_browser:
        print("Running tests WITH browser (requires DNNE server running)")
    else:
        print("Running tests WITHOUT browser (core functionality only)")
    
    suite = DNNEMCPTestSuite(run_browser_tests=args.with_browser)
    success = await suite.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())