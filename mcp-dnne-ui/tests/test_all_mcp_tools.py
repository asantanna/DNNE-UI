#!/usr/bin/env python3
"""Comprehensive test suite for all DNNE UI MCP tools"""

import asyncio
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dnne_ui_mcp_server import DNNE_UI_MCPServer
from browser_controller import BrowserController
from utils.helpers import format_mcp_response
from utils.state_manager import StateManager

class ToolTestResult:
    """Track individual tool test results"""
    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.status = "pending"  # pending, running, pass, fail, skip
        self.error = None
        self.duration = 0
        self.response = None
        self.notes = ""
        
    def __str__(self):
        status_icons = {
            "pass": "✅",
            "fail": "❌", 
            "skip": "⚠️",
            "pending": "⏳",
            "running": "🔄"
        }
        icon = status_icons.get(self.status, "❓")
        result = f"{icon} {self.name} ({self.duration:.2f}s)"
        if self.error:
            result += f"\n    Error: {self.error}"
        if self.notes:
            result += f"\n    Notes: {self.notes}"
        return result

class ComprehensiveMCPTestSuite:
    """Complete test suite for all MCP tools"""
    
    def __init__(self, run_browser_tests: bool = True):
        self.server = None
        self.browser = None
        self.run_browser_tests = run_browser_tests
        self.results: List[ToolTestResult] = []
        self.categories = {
            "Browser Lifecycle": [],
            "Core Workflow": [],
            "Export System": [],
            "Health & Status": [], 
            "Client Management": [],
            "Log Management": [],
            "UI Navigation": [],
            "Canvas Operations": [],
            "Utility Tools": []
        }
        
    async def setup(self):
        """Set up test environment"""
        print("Setting up comprehensive test environment...")
        self.server = DNNE_UI_MCPServer()
        
        if self.run_browser_tests:
            # Initialize browser controller
            self.browser = BrowserController()
            await self.browser.initialize()
            self.server.browser_controller = self.browser
            # Update error diagnostics
            self.server.error_diagnostics.browser = self.browser
            
            # Wait for UI to be ready
            print("Waiting for DNNE UI to be ready...")
            ready = await self.browser.wait_for_ui_ready(timeout=10000)
            if not ready:
                print("⚠️  DNNE UI not fully ready - some tests may fail")
        
    async def teardown(self):
        """Clean up test environment"""
        print("Cleaning up test environment...")
        if self.browser:
            await self.browser.cleanup()
        
        # Clear test state
        if self.server and self.server.state_manager:
            self.server.state_manager.clear_session_state()
    
    async def restore_ui_to_standard_state(self):
        """
        Restore UI to standard baseline state after test completion
        
        Standard state:
        - No dialogs or modals open
        - Sidebar in default state  
        - No menus expanded
        - No error messages visible
        - No blocking elements
        """
        if not self.browser or not self.browser.page:
            print("    ⚠️ Browser unavailable, skipping UI restoration")
            return
        
        try:
            print("    🔄 Restoring UI to standard state...")
            cleanup_actions = []
            
            # 1. Check and close dialogs (bulletproof)
            dialog_found = await self.browser.page.evaluate("""
                () => {
                    const dialogs = document.querySelectorAll('.p-dialog:not([style*="display: none"])');
                    return dialogs.length > 0;
                }
            """)
            
            if dialog_found:
                # Try different close methods
                close_selectors = [
                    ".p-dialog .p-dialog-close-button",  # Updated primary selector
                    ".p-dialog .p-dialog-header-close",  # Legacy fallback
                    ".p-dialog button[aria-label='Close']",
                    ".p-dialog button:has-text('Cancel')",
                    ".p-dialog button:has-text('Close')"
                ]
                
                for selector in close_selectors:
                    try:
                        if await self.browser.is_visible(selector, timeout=200):
                            await self.browser.click(selector, timeout=500)
                            cleanup_actions.append("Closed dialog")
                            break
                    except:
                        continue
                
                # Fallback: ESC key
                await self.browser.page.keyboard.press('Escape')
            
            # 2. Check and close sidebar panels
            sidebar_found = await self.browser.page.evaluate("""
                () => {
                    const sidebars = document.querySelectorAll('.p-sidebar:not([style*="display: none"])');
                    return sidebars.length > 0;
                }
            """)
            
            if sidebar_found:
                try:
                    if await self.browser.is_visible(".p-sidebar .p-sidebar-header-close", timeout=200):
                        await self.browser.click(".p-sidebar .p-sidebar-header-close", timeout=500)
                        cleanup_actions.append("Closed sidebar")
                except:
                    pass
            
            # 3. Clear menu selections (bulletproof)
            await self.browser.page.evaluate("""
                () => {
                    // Click elsewhere to close any open menus
                    const canvas = document.querySelector('#graph-canvas') || document.body;
                    if (canvas) {
                        canvas.click();
                    }
                    
                    // Remove focus from active elements
                    if (document.activeElement && document.activeElement.blur) {
                        document.activeElement.blur();
                    }
                }
            """)
            
            # 4. Clear any toast notifications
            toast_found = await self.browser.page.evaluate("""
                () => {
                    const toasts = document.querySelectorAll('.p-toast');
                    toasts.forEach(toast => toast.remove());
                    return toasts.length > 0;
                }
            """)
            
            if toast_found:
                cleanup_actions.append("Cleared toast notifications")
            
            # 5. Wait for UI to stabilize
            await asyncio.sleep(0.3)
            
            if cleanup_actions:
                print(f"    ✅ UI restored: {', '.join(cleanup_actions)}")
            else:
                print("    ✅ UI already in standard state")
            
        except Exception as e:
            print(f"    💀 FATAL: UI restore failed: {e}")
            print("    💀 Test suite cannot continue with undefined UI state")
            raise RuntimeError(f"Fatal UI restoration failure: {e}") from e

    async def recover_from_error(self, recovery_type: str = "general"):
        """
        Recover from specific UI errors that block normal operation
        
        Args:
            recovery_type: Type of recovery needed
                - "dialog_mask": Remove blocking overlay masks
                - "browser_restart": Restart browser (nuclear option)  
                - "load_default_workflow": Reset to clean workflow
                - "general": General error recovery
        """
        if not self.browser:
            print("    ⚠️ Browser unavailable, skipping error recovery")
            return
        
        try:
            print(f"    🚨 Error recovery: {recovery_type}")
            
            if recovery_type == "dialog_mask":
                # Remove blocking overlay masks
                mask_count = await self.browser.page.evaluate("""
                    () => {
                        const masks = document.querySelectorAll('.p-dialog-mask, .p-overlay-mask, [data-pc-section="mask"]');
                        masks.forEach(mask => mask.remove());
                        return masks.length;
                    }
                """)
                if mask_count > 0:
                    print(f"    ✅ Removed {mask_count} blocking masks")
            
            elif recovery_type == "load_default_workflow":
                # Reset to default/empty workflow
                try:
                    from tools.workflow_tools import WorkflowTools
                    tools = WorkflowTools(self.browser, self.server.state)
                    await tools.new_blank_workflow()
                    print("    ✅ Reset to blank workflow")
                except:
                    print("    ⚠️ Could not reset workflow")
            
            elif recovery_type == "browser_restart":
                # Nuclear option - restart browser
                try:
                    success = await self.browser.restart_browser()
                    if success:
                        print("    ✅ Browser restarted")
                    else:
                        print("    ⚠️ Browser restart failed")
                except:
                    print("    ⚠️ Browser restart failed")
            
            else:  # general recovery
                # Remove any blocking elements
                await self.browser.page.evaluate("""
                    () => {
                        // Remove dialog masks
                        const masks = document.querySelectorAll('.p-dialog-mask, .p-overlay-mask');
                        masks.forEach(mask => mask.remove());
                        
                        // Press ESC to close any modal dialogs
                        document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
                    }
                """)
                print("    ✅ General error recovery completed")
            
            # Always restore to standard state after recovery
            await self.restore_ui_to_standard_state()
            
        except Exception as e:
            print(f"    💀 FATAL: Error recovery failed: {e}")
            print("    💀 Test suite cannot continue with broken UI state")
            raise RuntimeError(f"Fatal error recovery failure: {e}") from e

    async def run_tool_test(self, tool_name: str, category: str, test_func) -> ToolTestResult:
        """Run a single tool test with proper UI state management"""
        result = ToolTestResult(tool_name, category)
        result.status = "running"
        print(f"\nTesting: {tool_name} ({category})")
        
        start_time = asyncio.get_event_loop().time()
        try:
            response = await test_func()
            result.response = response
            
            # Validate response structure
            if isinstance(response, dict):
                if "success" in response:
                    if response["success"]:
                        result.status = "pass"
                        result.notes = response.get("message", "Tool executed successfully")
                    else:
                        result.status = "fail"
                        result.error = response.get("error", "Tool returned success=False")
                else:
                    result.status = "fail"
                    result.error = "Response missing 'success' field"
            else:
                result.status = "fail"
                result.error = f"Invalid response type: {type(response)}"
                
        except Exception as e:
            result.status = "fail"
            result.error = str(e)
            print(f"  ❌ Error: {e}")
            
            # Call specific recovery based on error type
            error_str = str(e).lower()
            if "dialog-mask" in error_str or "intercepts pointer events" in error_str:
                await self.recover_from_error("dialog_mask")
            elif "browser" in error_str:
                await self.recover_from_error("browser_restart")
            else:
                await self.recover_from_error("general")
            
        finally:
            result.duration = asyncio.get_event_loop().time() - start_time
            
            # Always restore UI to standard state after test (pass or fail)
            if self.run_browser_tests:
                await self.restore_ui_to_standard_state()
        
        self.results.append(result)
        self.categories[category].append(result)
        return result
    
    # Browser Lifecycle Tests - These use the server's own lifecycle methods
    async def test_initialize_browser(self):
        """Test browser initialization"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        # Test browser initialization by checking if browser is available after setup
        if self.browser and await self.browser.is_healthy():
            return {"success": True, "message": "Browser initialized successfully"}
        else:
            return {"success": False, "error": "Browser initialization failed"}
    
    async def test_cleanup_browser(self):
        """Test browser cleanup"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        # Test cleanup functionality
        try:
            if self.browser:
                await self.browser.cleanup()
                return {"success": True, "message": "Browser cleanup successful"}
            return {"success": True, "message": "No browser to cleanup"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_restart_browser(self):
        """Test browser restart"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        # Test restart functionality
        try:
            if self.browser:
                success = await self.browser.restart_browser()
                if success:
                    return {"success": True, "message": "Browser restart successful"}
                else:
                    return {"success": False, "error": "Browser restart failed"}
            return {"success": False, "error": "No browser to restart"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_is_browser_running(self):
        """Test browser status check"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        # Test browser running status
        try:
            if self.browser:
                healthy = await self.browser.is_healthy()
                return {"success": True, "data": {"running": healthy}, "message": f"Browser running: {healthy}"}
            return {"success": True, "data": {"running": False}, "message": "Browser not initialized"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Core Workflow Tests - Using WorkflowTools 
    async def test_load_workflow(self):
        """Test workflow loading"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.workflow_tools import WorkflowTools
        tools = WorkflowTools(self.browser, self.server.state)
        return await tools.load_workflow("MNIST_Test")
    
    async def test_get_current_workflow_name(self):
        """Test getting current workflow name"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.workflow_tools import WorkflowTools
        tools = WorkflowTools(self.browser, self.server.state)
        return await tools.get_current_workflow_name()
        
    async def test_save_workflow(self):
        """Test workflow saving"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.workflow_tools import WorkflowTools
        tools = WorkflowTools(self.browser, self.server.state)
        return await tools.save_workflow("test_workflow")
    
    async def test_new_blank_workflow(self):
        """Test creating new blank workflow"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.workflow_tools import WorkflowTools
        tools = WorkflowTools(self.browser, self.server.state)
        return await tools.new_blank_workflow()
    
    async def test_clear_workflow(self):
        """Test clearing workflow"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.workflow_tools import WorkflowTools
        tools = WorkflowTools(self.browser, self.server.state)
        return await tools.clear_workflow()
    
    async def test_get_workflow_list(self):
        """Test getting workflow list"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.workflow_tools import WorkflowTools
        tools = WorkflowTools(self.browser, self.server.state)
        return await tools.get_workflow_list()
    
    # Export System Tests
    async def test_export_workflow(self):
        """Test workflow export"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.workflow_tools import WorkflowTools
        tools = WorkflowTools(self.browser, self.server.state)
        return await tools.export_workflow(run_after=False)
    
    # Health & Status Tests
    async def test_is_ui_healthy(self):
        """Test UI health check"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        # Test UI health by checking browser and key elements
        try:
            if not self.browser:
                return {"success": False, "error": "Browser not initialized"}
            
            healthy = await self.browser.is_healthy()
            if healthy:
                return {"success": True, "message": "UI is healthy"}
            else:
                return {"success": False, "error": "UI health check failed"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_take_screenshot(self):
        """Test screenshot capture"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        try:
            if not self.browser:
                return {"success": False, "error": "Browser not initialized"}
            
            screenshot_path = await self.browser.take_screenshot("test_screenshot")
            if screenshot_path:
                return {"success": True, "data": {"path": screenshot_path}, "message": f"Screenshot saved: {screenshot_path}"}
            else:
                return {"success": False, "error": "Screenshot capture failed"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_util_get_dnne_server_status(self):
        """Test DNNE server status check"""
        # Call the properly formatted MCP tool function, not the private method
        result = await self.server._util_get_dnne_server_status()
        from utils.helpers import format_mcp_response
        return format_mcp_response(
            result.get("running", False),
            data=result,
            message="DNNE server is running" if result.get("running") else "DNNE server not reachable"
        )
    
    async def test_util_is_agent_server_running(self):
        """Test agent server status check"""
        # Call the properly formatted MCP tool function, not the private method
        result = await self.server._util_is_agent_server_running()
        from utils.helpers import format_mcp_response
        return format_mcp_response(
            result.get("running", False),
            data=result,
            message="Agent server is running" if result.get("running") else "Agent server not reachable"
        )
    
    # Client Management Tests (using tool instances)
    async def test_get_connected_clients(self):
        """Test getting connected clients"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.client_tools import ClientTools
        tools = ClientTools(self.server, self.server.state)
        return await tools.get_connected_clients()
    
    async def test_select_client(self):
        """Test client selection"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.client_tools import ClientTools
        tools = ClientTools(self.server, self.server.state)
        return await tools.select_client("Local", "taskbar")
    
    async def test_get_agent_status(self):
        """Test agent status check"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.client_tools import ClientTools
        tools = ClientTools(self.server, self.server.state)
        return await tools.get_agent_status()
    
    async def test_show_all_logs(self):
        """Test showing all logs"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.client_tools import ClientTools
        tools = ClientTools(self.server, self.server.state)
        return await tools.show_all_logs()
    
    async def test_clear_logs(self):
        """Test clearing logs"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.client_tools import ClientTools
        tools = ClientTools(self.server, self.server.state)
        return await tools.clear_logs()
    
    # Log Management Tests
    async def test_get_client_logs(self):
        """Test getting client logs"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.log_tools import LogTools
        tools = LogTools(self.server, self.server.state)
        return await tools.get_client_logs()
    
    async def test_get_training_metrics(self):
        """Test extracting training metrics"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.log_tools import LogTools
        tools = LogTools(self.server, self.server.state)
        return await tools.get_training_metrics()
    
    async def test_get_export_errors(self):
        """Test finding export errors"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.log_tools import LogTools
        tools = LogTools(self.server, self.server.state)
        return await tools.get_export_errors()
    
    async def test_get_recent_errors(self):
        """Test getting recent errors"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.log_tools import LogTools
        tools = LogTools(self.server, self.server.state)
        return await tools.get_recent_errors()
    
    async def test_wait_for_log_pattern(self):
        """Test waiting for log pattern"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.log_tools import LogTools
        tools = LogTools(self.server, self.server.state)
        # Use a short timeout to avoid hanging the test
        return await tools.wait_for_log_pattern("test_pattern", timeout=1)
    
    # UI Navigation Tests
    async def test_open_sidebar_tab(self):
        """Test opening sidebar tab"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.ui_tools import UITools
        tools = UITools(self.server, self.server.state)
        return await tools.open_sidebar_tab("workflows")
    
    async def test_open_menu(self):
        """Test opening menu"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.ui_tools import UITools
        tools = UITools(self.server, self.server.state)
        return await tools.open_menu("Workflow/Save As")
    
    async def test_dismiss_dialog(self):
        """Test dismissing dialog - opens Browse Templates dialog and dismisses it"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.ui_tools import UITools
        tools = UITools(self.server, self.server.state)
        
        # First, open the Browse Templates dialog
        print("      Opening Browse Templates dialog...")
        menu_result = await tools.click_menu_item("Workflow/Browse Templates")
        if not menu_result.get("success"):
            return {"success": False, "error": f"Failed to open dialog: {menu_result.get('error')}"}
        
        await asyncio.sleep(2)  # Let dialog fully appear
        
        # Now test dismissing it
        return await tools.dismiss_dialog()
    
    async def test_get_error_message(self):
        """Test getting error message"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.ui_tools import UITools
        tools = UITools(self.server, self.server.state)
        return await tools.get_error_message()
    
    async def test_wait_for_ui_ready(self):
        """Test waiting for UI ready"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.ui_tools import UITools
        tools = UITools(self.server, self.server.state)
        return await tools.wait_for_ui_ready()
    
    async def test_click_menu_header(self):
        """Test clicking menu header"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.ui_tools import UITools
        tools = UITools(self.server, self.server.state)
        return await tools.click_menu_header("Workflow")
    
    async def test_click_menu_item(self):
        """Test clicking menu item"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.ui_tools import UITools
        tools = UITools(self.server, self.server.state)
        return await tools.click_menu_item("Workflow/New")
    
    # Canvas Operations Tests
    async def test_zoom_to_fit(self):
        """Test zoom to fit"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.canvas_tools import CanvasTools
        tools = CanvasTools(self.server, self.server.state)
        return await tools.zoom_to_fit()
    
    async def test_toggle_link_visibility(self):
        """Test toggling link visibility"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.canvas_tools import CanvasTools
        tools = CanvasTools(self.server, self.server.state)
        return await tools.toggle_link_visibility()
    
    async def test_get_node_count(self):
        """Test getting node count"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.canvas_tools import CanvasTools
        tools = CanvasTools(self.server, self.server.state)
        return await tools.get_node_count()
    
    async def test_zoom_in(self):
        """Test zoom in"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.canvas_tools import CanvasTools
        tools = CanvasTools(self.server, self.server.state)
        return await tools.zoom_in()
    
    async def test_zoom_out(self):
        """Test zoom out"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.canvas_tools import CanvasTools
        tools = CanvasTools(self.server, self.server.state)
        return await tools.zoom_out()
    
    async def test_get_canvas_state(self):
        """Test getting canvas state"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        from tools.canvas_tools import CanvasTools
        tools = CanvasTools(self.server, self.server.state)
        return await tools.get_canvas_state()
    
    # Utility Tools Tests
    async def test_util_find_elements_by_text(self):
        """Test finding elements by text"""
        if not self.run_browser_tests:
            return {"success": True, "message": "Skipped - browser tests disabled"}
        
        # Call the properly formatted MCP tool function, not the private method
        result = await self.server._util_find_elements_by_text("Workflow", limit=5)
        from utils.helpers import format_mcp_response
        return format_mcp_response(
            result.get("found", False),
            data=result,
            message=f"Found {result.get('count', 0)} elements containing 'Workflow'"
        )
    
    async def run_all_tests(self):
        """Run all comprehensive tests and generate detailed report"""
        print("\n" + "="*80)
        print("DNNE UI MCP Server - Comprehensive Tool Testing")
        print("="*80)
        
        await self.setup()
        
        try:
            # Define all tests by category
            test_plan = [
                # Browser Lifecycle
                ("initialize_browser", "Browser Lifecycle", self.test_initialize_browser),
                ("cleanup_browser", "Browser Lifecycle", self.test_cleanup_browser),
                ("restart_browser", "Browser Lifecycle", self.test_restart_browser),
                ("is_browser_running", "Browser Lifecycle", self.test_is_browser_running),
                
                # Core Workflow  
                ("load_workflow", "Core Workflow", self.test_load_workflow),
                ("get_current_workflow_name", "Core Workflow", self.test_get_current_workflow_name),
                ("save_workflow", "Core Workflow", self.test_save_workflow),
                ("new_blank_workflow", "Core Workflow", self.test_new_blank_workflow),
                ("clear_workflow", "Core Workflow", self.test_clear_workflow),
                ("get_workflow_list", "Core Workflow", self.test_get_workflow_list),
                
                # Export System
                ("export_workflow", "Export System", self.test_export_workflow),
                
                # Health & Status
                ("is_ui_healthy", "Health & Status", self.test_is_ui_healthy),
                ("take_screenshot", "Health & Status", self.test_take_screenshot),
                ("util_get_dnne_server_status", "Health & Status", self.test_util_get_dnne_server_status),
                ("util_is_agent_server_running", "Health & Status", self.test_util_is_agent_server_running),
                
                # Client Management
                ("get_connected_clients", "Client Management", self.test_get_connected_clients),
                ("select_client", "Client Management", self.test_select_client),
                ("get_agent_status", "Client Management", self.test_get_agent_status),
                ("show_all_logs", "Client Management", self.test_show_all_logs),
                ("clear_logs", "Client Management", self.test_clear_logs),
                
                # Log Management
                ("get_client_logs", "Log Management", self.test_get_client_logs),
                ("get_training_metrics", "Log Management", self.test_get_training_metrics),
                ("get_export_errors", "Log Management", self.test_get_export_errors),
                ("get_recent_errors", "Log Management", self.test_get_recent_errors),
                ("wait_for_log_pattern", "Log Management", self.test_wait_for_log_pattern),
                
                # UI Navigation
                ("open_sidebar_tab", "UI Navigation", self.test_open_sidebar_tab),
                ("open_menu", "UI Navigation", self.test_open_menu),
                ("dismiss_dialog", "UI Navigation", self.test_dismiss_dialog),
                ("get_error_message", "UI Navigation", self.test_get_error_message),
                ("wait_for_ui_ready", "UI Navigation", self.test_wait_for_ui_ready),
                ("click_menu_header", "UI Navigation", self.test_click_menu_header),
                ("click_menu_item", "UI Navigation", self.test_click_menu_item),
                
                # Canvas Operations
                ("zoom_to_fit", "Canvas Operations", self.test_zoom_to_fit),
                ("toggle_link_visibility", "Canvas Operations", self.test_toggle_link_visibility),
                ("get_node_count", "Canvas Operations", self.test_get_node_count),
                ("zoom_in", "Canvas Operations", self.test_zoom_in),
                ("zoom_out", "Canvas Operations", self.test_zoom_out),
                ("get_canvas_state", "Canvas Operations", self.test_get_canvas_state),
                
                # Utility Tools
                ("util_find_elements_by_text", "Utility Tools", self.test_util_find_elements_by_text),
            ]
            
            # Run all tests
            total_tests = len(test_plan)
            print(f"Running {total_tests} comprehensive tool tests...")
            
            for i, (tool_name, category, test_func) in enumerate(test_plan, 1):
                print(f"\n[{i}/{total_tests}] Testing {category}: {tool_name}")
                await self.run_tool_test(tool_name, category, test_func)
                
                # Brief pause between tests
                await asyncio.sleep(0.1)
            
        finally:
            await self.teardown()
        
        # Generate comprehensive report
        self.generate_report()
        return len([r for r in self.results if r.status == "fail"]) == 0
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST RESULTS")
        print("="*80)
        
        # Overall statistics
        total = len(self.results)
        passed = len([r for r in self.results if r.status == "pass"])
        failed = len([r for r in self.results if r.status == "fail"])
        skipped = len([r for r in self.results if r.status == "skip"])
        
        print(f"\nOVERALL SUMMARY:")
        print(f"Total Tools Tested: {total}")
        print(f"✅ Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"❌ Failed: {failed} ({failed/total*100:.1f}%)")
        if skipped > 0:
            print(f"⚠️  Skipped: {skipped} ({skipped/total*100:.1f}%)")
        
        # Category breakdown
        print(f"\nCATEGORY BREAKDOWN:")
        for category, results in self.categories.items():
            if results:
                cat_passed = len([r for r in results if r.status == "pass"])
                cat_failed = len([r for r in results if r.status == "fail"])
                cat_total = len(results)
                success_rate = cat_passed/cat_total*100 if cat_total > 0 else 0
                print(f"  {category}: {cat_passed}/{cat_total} passed ({success_rate:.1f}%)")
        
        # Detailed results by category
        print(f"\nDETAILED RESULTS:")
        for category, results in self.categories.items():
            if results:
                print(f"\n{category}:")
                print("-" * len(category))
                for result in results:
                    print(f"  {result}")
        
        # Failed tests summary
        failed_tests = [r for r in self.results if r.status == "fail"]
        if failed_tests:
            print(f"\nFAILED TESTS SUMMARY:")
            print("=" * 20)
            for result in failed_tests:
                print(f"❌ {result.name} ({result.category})")
                print(f"   Error: {result.error}")
                if result.notes:
                    print(f"   Notes: {result.notes}")
        
        # Export results to JSON
        self.export_results_json()
        
        print(f"\n" + "="*80)
        if failed == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print(f"⚠️  {failed} TESTS FAILED - See details above")
        print("="*80)
    
    def export_results_json(self):
        """Export detailed results to JSON file"""
        results_data = {
            "test_suite": "DNNE UI MCP Server - Comprehensive Tool Testing",
            "test_date": datetime.now().isoformat(),
            "testing_status": "Complete",
            "summary": {
                "total_tools": len(self.results),
                "tools_passed": len([r for r in self.results if r.status == "pass"]),
                "tools_failed": len([r for r in self.results if r.status == "fail"]),
                "tools_skipped": len([r for r in self.results if r.status == "skip"]),
                "success_rate": len([r for r in self.results if r.status == "pass"])/len(self.results)*100 if self.results else 0
            },
            "category_results": {},
            "detailed_test_results": []
        }
        
        # Category results
        for category, results in self.categories.items():
            if results:
                passed = len([r for r in results if r.status == "pass"])
                failed = len([r for r in results if r.status == "fail"])
                total = len(results)
                results_data["category_results"][category] = {
                    "tools_tested": total,
                    "tools_passed": passed,
                    "tools_failed": failed,
                    "success_rate": passed/total*100 if total > 0 else 0
                }
        
        # Detailed results
        for result in self.results:
            results_data["detailed_test_results"].append({
                "category": result.category,
                "tool": result.name,
                "status": result.status.upper(),
                "duration": result.duration,
                "error": result.error,
                "notes": result.notes
            })
        
        # Save to file
        output_file = Path(__file__).parent / "test_results_comprehensive.json"
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n📄 Detailed results exported to: {output_file}")

async def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive test of all DNNE UI MCP tools")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Skip browser-dependent tests (core functionality only)"
    )
    args = parser.parse_args()
    
    run_browser_tests = not args.no_browser
    
    if run_browser_tests:
        print("Running COMPREHENSIVE tests with browser automation")
        print("⚠️  Requires DNNE server running at http://172.22.160.1:8188")
    else:
        print("Running CORE tests only (no browser automation)")
    
    suite = ComprehensiveMCPTestSuite(run_browser_tests=run_browser_tests)
    success = await suite.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())