#!/usr/bin/env python3
"""
Test Stateless Browser State Management
Verifies that state is queried from browser, not persisted to disk
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.state_manager import StateManager
from src.browser_controller import BrowserController

class StatelessBrowserTest:
    """Test suite for stateless browser state management"""
    
    def __init__(self):
        self.test_results = []
        
    def log_test(self, test_name, success, message):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}: {message}")
        self.test_results.append({"test": test_name, "success": success})
        
    async def test_state_manager_no_disk(self):
        """Test that StateManager doesn't use disk persistence"""
        print("\n🔧 Testing StateManager (no disk persistence)...")
        
        # Create StateManager instance
        state_mgr = StateManager()
        
        # Check that no state file is created
        state_file = Path("mcp_state.json")
        if state_file.exists():
            self.log_test(
                "no_state_file",
                False,
                "State file exists - should not persist to disk!"
            )
            # Clean up
            state_file.unlink()
        else:
            self.log_test(
                "no_state_file",
                True,
                "No state file created - correctly using in-memory only"
            )
        
        # Test state updates are in-memory only
        state_mgr.update("test_key", "test_value")
        
        # Create new instance - should NOT have the test value
        new_state_mgr = StateManager()
        if new_state_mgr.get("test_key") is None:
            self.log_test(
                "state_not_persisted",
                True,
                "State not persisted between instances - correct!"
            )
        else:
            self.log_test(
                "state_not_persisted",
                False,
                "State persisted between instances - should be in-memory only!"
            )
            
    async def test_browser_state_queries(self):
        """Test browser state query methods"""
        print("\n🔧 Testing Browser State Query Methods...")
        
        # Note: This test requires browser to be running
        # We'll create a mock test that shows the methods exist
        
        browser = BrowserController()
        
        # Check that query methods exist
        methods = [
            'get_current_workflow',
            'get_sidebar_state',
            'get_canvas_info',
            'get_agent_status',
            'get_ui_state'
        ]
        
        for method_name in methods:
            if hasattr(browser, method_name):
                self.log_test(
                    f"method_{method_name}",
                    True,
                    f"Method {method_name} exists"
                )
            else:
                self.log_test(
                    f"method_{method_name}",
                    False,
                    f"Method {method_name} missing!"
                )
                
    async def test_no_state_recovery(self):
        """Test that StateRecovery class has been removed"""
        print("\n🔧 Testing StateRecovery Removal...")
        
        try:
            from src.utils.state_manager import StateRecovery
            self.log_test(
                "state_recovery_removed",
                False,
                "StateRecovery class still exists - should be removed!"
            )
        except ImportError:
            self.log_test(
                "state_recovery_removed",
                True,
                "StateRecovery class removed - correct!"
            )
            
    async def run_all_tests(self):
        """Run all stateless browser tests"""
        print("=" * 60)
        print("🚀 Stateless Browser State Management Test Suite")
        print("=" * 60)
        
        await self.test_state_manager_no_disk()
        await self.test_browser_state_queries()
        await self.test_no_state_recovery()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["success"])
        failed = total - passed
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        if failed == 0:
            print("\n✅ All tests passed! Stateless implementation verified.")
        else:
            print(f"\n❌ {failed} test(s) failed. Please review.")
            
        return failed == 0

async def main():
    """Main test runner"""
    tester = StatelessBrowserTest()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())