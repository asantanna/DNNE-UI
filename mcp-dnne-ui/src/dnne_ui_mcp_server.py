#!/usr/bin/env python3
"""
DNNE UI MCP Server - Browser automation for DNNE UI testing

Naming Convention:
- Functions WITHOUT 'util_' prefix: Query through UI only (test what users see)
- Functions WITH 'util_' prefix: Query servers directly (get ground truth)

Use UI functions for testing the user interface.
Use util functions for debugging and verification.
"""

import asyncio
import aiohttp
import logging
import os
import sys
from typing import Optional, Dict, Any, List
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from mcp.server import FastMCP

try:
    from .browser_controller import BrowserController
    from .utils.helpers import setup_logging, get_env_var, format_mcp_response
    from .utils.js_defs import *
    from .utils.state_manager import StateManager
    from .utils.error_handler import ErrorDiagnostics, with_error_handling
except ImportError:
    # For direct script execution
    from browser_controller import BrowserController
    from utils.helpers import setup_logging, get_env_var, format_mcp_response
    from utils.js_defs import *
    from utils.state_manager import StateManager
    from utils.error_handler import ErrorDiagnostics, with_error_handling

# Load environment variables
load_dotenv()

# Setup logging
setup_logging(get_env_var("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

class DNNE_UI_MCPServer:
    """MCP Server for DNNE UI automation"""
    
    def __init__(self):
        """Initialize the DNNE UI MCP Server"""
        self.server = FastMCP(
            name="dnne-ui",
            instructions="DNNE UI automation server for controlling the DNNE interface via browser automation"
        )
        self.browser_controller: Optional[BrowserController] = None
        self.dnne_url = get_env_var("DNNE_URL", "http://172.22.160.1:8188")
        self.headless = get_env_var("BROWSER_HEADLESS", "false").lower() == "true"
        
        # State management - in-memory only, no disk persistence
        self.state_manager = StateManager()
        self.state = self.state_manager.state
        
        # Error diagnostics
        self.error_diagnostics = ErrorDiagnostics()
        
        # Register all tools
        self._register_tools()
        
    def _register_tools(self):
        """Register all MCP tools with the server"""
        
        # Lifecycle tools
        async def initialize_browser() -> Dict[str, Any]:
            """Initialize the browser and navigate to DNNE UI"""
            try:
                # Check if browser exists AND is healthy
                if self.browser_controller:
                    # Check if browser process is active
                    if self.browser_controller.is_playwright_browser_process_active():
                        # Check if browser window is available
                        if self.browser_controller.is_browser_window_available():
                            # Check if it's responsive to JavaScript
                            if await self.browser_controller.is_javascript_executable():
                                return format_mcp_response(True, message="Browser already initialized and healthy")
                            else:
                                logger.info("Browser window exists but not responsive to JavaScript, cleaning up...")
                        else:
                            logger.info("Browser process active but window closed, cleaning up...")
                    else:
                        logger.info("Browser controller exists but process not active, cleaning up...")
                    
                    # Browser exists but not healthy
                    await self.browser_controller.cleanup()
                    self.browser_controller = None
                
                # Create new browser instance
                self.browser_controller = BrowserController(
                    dnne_url=self.dnne_url,
                    headless=self.headless
                )
                await self.browser_controller.initialize()
                return format_mcp_response(True, message="Browser initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize browser: {e}")
                self.browser_controller = None  # Clear reference on failure
                return format_mcp_response(False, error=str(e))
        
        self.server.add_tool(
            initialize_browser,
            name="initialize_browser",
            description="Initialize the browser and navigate to DNNE UI"
        )
        
        async def shut_down_browser_automation() -> Dict[str, Any]:
            """Shut down browser automation and free all resources"""
            try:
                if self.browser_controller:
                    await self.browser_controller.cleanup()
                    self.browser_controller = None
                return format_mcp_response(True, message="Browser automation shut down successfully")
            except Exception as e:
                logger.error(f"Failed to shut down browser automation: {e}")
                return format_mcp_response(False, error=str(e))
        
        self.server.add_tool(
            shut_down_browser_automation,
            name="shut_down_browser_automation",
            description="Shut down browser automation and free all resources"
        )
        
        async def is_browser_running() -> Dict[str, Any]:
            """Check if browser window is available"""
            if not self.browser_controller:
                return format_mcp_response(False, data={
                    "process_active": False,
                    "window_available": False
                }, message="Browser not initialized")
            
            process_active = self.browser_controller.is_playwright_browser_process_active()
            window_available = self.browser_controller.is_browser_window_available()
            
            if not process_active:
                message = "Browser process not active"
            elif not window_available:
                message = "Browser process active but window closed"
            else:
                message = "Browser window available"
            
            return format_mcp_response(
                window_available,
                data={
                    "process_active": process_active,
                    "window_available": window_available
                },
                message=message
            )
        
        self.server.add_tool(
            is_browser_running,
            name="is_browser_running",
            description="Check if browser window is available"
        )
        
        async def restart_browser() -> Dict[str, Any]:
            """Restart browser for recovery"""
            try:
                # Always clean up existing browser first if it exists
                if self.browser_controller:
                    try:
                        await self.browser_controller.cleanup()
                    except Exception as cleanup_error:
                        logger.warning(f"Error during browser cleanup: {cleanup_error}")
                    self.browser_controller = None
                
                # Now initialize a fresh browser
                return await initialize_browser()
                    
            except Exception as e:
                logger.error(f"Failed to restart browser: {e}")
                self.browser_controller = None
                return format_mcp_response(False, error=str(e))
        
        self.server.add_tool(
            restart_browser,
            name="restart_browser",
            description="Restart the browser for recovery"
        )
        
        # Workflow management tools
        async def load_workflow(name: str) -> Dict[str, Any]:
            """
            Load a workflow from the workflows sidebar
            
            Args:
                name: Name of the workflow file (e.g., "MNIST_Test.json")
            """
            # Use the workflow tools implementation which handles edge cases better
            try:
                from .tools.workflow_tools import WorkflowTools
            except ImportError:
                from tools.workflow_tools import WorkflowTools
            tools = WorkflowTools(self, self.state)
            return await tools.load_workflow(name)
        
        self.server.add_tool(
            load_workflow,
            name="load_workflow",
            description="Load a workflow from the workflows sidebar"
        )
        
        async def get_current_workflow_name() -> Dict[str, Any]:
            """Get the name of the currently loaded workflow"""
            try:
                if not self.browser_controller:
                    return format_mcp_response(False, error="Browser not initialized")
                
                # Delegate to WorkflowTools which has more comprehensive browser query
                try:
                    from .tools.workflow_tools import WorkflowTools
                except ImportError:
                    from tools.workflow_tools import WorkflowTools
                tools = WorkflowTools(self, self.state)
                return await tools.get_current_workflow_name()
                
            except Exception as e:
                logger.error(f"Failed to get workflow name: {e}")
                return format_mcp_response(False, error=str(e))
        
        self.server.add_tool(
            get_current_workflow_name,
            name="get_current_workflow_name",
            description="Get the name of the currently loaded workflow"
        )
        
        async def export_workflow(run_after: bool = False) -> Dict[str, Any]:
            """
            Export the current workflow
            
            Args:
                run_after: Whether to run the workflow after export
            """
            try:
                if not self.browser_controller:
                    return format_mcp_response(False, error="Browser not initialized")
                
                # Delegate to WorkflowTools which has the comprehensive implementation
                try:
                    from .tools.workflow_tools import WorkflowTools
                except ImportError:
                    from tools.workflow_tools import WorkflowTools
                tools = WorkflowTools(self, self.state)
                return await tools.export_workflow(run_after)
                
            except Exception as e:
                logger.error(f"Failed to export workflow: {e}")
                return format_mcp_response(False, error=str(e))
        
        self.server.add_tool(
            export_workflow,
            name="export_workflow",
            description="Export the current workflow"
        )
        
        async def take_screenshot(name: str = "dnne_ui") -> Dict[str, Any]:
            """
            Take a screenshot of the DNNE UI
            
            Args:
                name: Name for the screenshot file
            """
            try:
                if not self.browser_controller:
                    return format_mcp_response(False, error="Browser not initialized")
                
                path = await self.browser_controller.take_screenshot(name)
                
                if path:
                    return format_mcp_response(
                        True,
                        data={"path": path},
                        message=f"Screenshot saved to {path}"
                    )
                else:
                    return format_mcp_response(False, error="Failed to take screenshot")
                    
            except Exception as e:
                logger.error(f"Failed to take screenshot: {e}")
                return format_mcp_response(False, error=str(e))
        
        self.server.add_tool(
            take_screenshot,
            name="take_screenshot",
            description="Take a screenshot of the DNNE UI"
        )
        
        async def is_ui_healthy() -> Dict[str, Any]:
            """Check if the DNNE UI is healthy and responsive"""
            try:
                # First check if browser is even initialized
                if not self.browser_controller:
                    return format_mcp_response(
                        False, 
                        error="Browser not initialized",
                        data={"browser_initialized": False}
                    )
                
                # Check if browser process is active
                if not self.browser_controller.is_playwright_browser_process_active():
                    return format_mcp_response(
                        False, 
                        error="Browser process not active. Use initialize_browser.",
                        data={"browser_process": False}
                    )
                
                # Check if browser window is available
                if not self.browser_controller.is_browser_window_available():
                    return format_mcp_response(
                        False,
                        error="Browser window closed. Use restart_browser.",
                        data={"browser_process": True, "window_available": False}
                    )
                
                # Check if JavaScript is executable
                if not await self.browser_controller.is_javascript_executable():
                    return format_mcp_response(
                        False,
                        error="Browser window not responsive. Use restart_browser.",
                        data={"browser_process": True, "window_available": True, "js_executable": False}
                    )
                
                issues = []
                
                # Check for key UI elements
                if not await self.browser_controller.is_visible(SIDEBAR_BUTTON):
                    issues.append("Sidebar buttons not visible")
                
                if not await self.browser_controller.is_visible(COMFYUI_MENU):
                    issues.append("Menu bar not visible")
                
                # Check for error dialogs
                if await self.browser_controller.is_visible(DIALOG):
                    issues.append("Error dialog is open")
                
                # Get agent status from browser
                agent_status = await self.browser_controller.get_agent_status()
                if not agent_status["agent_connected"]:
                    issues.append("Agent is disconnected")
                
                # Get comprehensive UI state
                ui_state = await self.browser_controller.get_ui_state()
                
                healthy = len(issues) == 0
                
                return format_mcp_response(
                    healthy,
                    data={
                        "healthy": healthy,
                        "issues": issues,
                        "ui_state": ui_state
                    },
                    message="UI is healthy" if healthy else f"UI has {len(issues)} issues"
                )
                
            except Exception as e:
                logger.error(f"Failed to check UI health: {e}")
                return format_mcp_response(False, error=str(e))
        
        self.server.add_tool(
            is_ui_healthy,
            name="is_ui_healthy",
            description="Check if the DNNE UI is healthy and responsive"
        )
        
        # Additional workflow tools
        async def save_workflow(name: Optional[str] = None) -> Dict[str, Any]:
            """
            Save the current workflow
            
            Args:
                name: Optional name for save-as operation
            """
            try:
                if not self.browser_controller:
                    return format_mcp_response(False, error="Browser not initialized")
                
                try:
                    from .tools.workflow_tools import WorkflowTools
                except ImportError:
                    from tools.workflow_tools import WorkflowTools
                tools = WorkflowTools(self, self.state)
                return await tools.save_workflow(name)
                
            except Exception as e:
                logger.error(f"Failed to save workflow: {e}")
                return format_mcp_response(False, error=str(e))
        
        self.server.add_tool(
            save_workflow,
            name="save_workflow",
            description="Save the current workflow (optionally with a new name)"
        )
        
        async def new_blank_workflow() -> Dict[str, Any]:
            """Create a new blank workflow"""
            try:
                if not self.browser_controller:
                    return format_mcp_response(False, error="Browser not initialized")
                
                try:
                    from .tools.workflow_tools import WorkflowTools
                except ImportError:
                    from tools.workflow_tools import WorkflowTools
                tools = WorkflowTools(self, self.state)
                return await tools.new_blank_workflow()
                
            except Exception as e:
                logger.error(f"Failed to create new workflow: {e}")
                return format_mcp_response(False, error=str(e))
        
        self.server.add_tool(
            new_blank_workflow,
            name="new_blank_workflow",
            description="Create a new blank workflow"
        )
        
        async def clear_workflow() -> Dict[str, Any]:
            """Clear the current workflow"""
            try:
                if not self.browser_controller:
                    return format_mcp_response(False, error="Browser not initialized")
                
                try:
                    from .tools.workflow_tools import WorkflowTools
                except ImportError:
                    from tools.workflow_tools import WorkflowTools
                tools = WorkflowTools(self, self.state)
                return await tools.clear_workflow()
                
            except Exception as e:
                logger.error(f"Failed to clear workflow: {e}")
                return format_mcp_response(False, error=str(e))
        
        self.server.add_tool(
            clear_workflow,
            name="clear_workflow",
            description="Clear the current workflow"
        )
        
        async def get_workflow_list() -> Dict[str, Any]:
            """Get list of available workflows"""
            try:
                if not self.browser_controller:
                    return format_mcp_response(False, error="Browser not initialized")
                
                try:
                    from .tools.workflow_tools import WorkflowTools
                except ImportError:
                    from tools.workflow_tools import WorkflowTools
                tools = WorkflowTools(self, self.state)
                return await tools.get_workflow_list()
                
            except Exception as e:
                logger.error(f"Failed to get workflow list: {e}")
                return format_mcp_response(False, error=str(e))
        
        self.server.add_tool(
            get_workflow_list,
            name="get_workflow_list",
            description="Get list of available workflows"
        )
        
        # Register all additional tools from tool modules
        try:
            from .tools.register_all_tools import register_all_additional_tools
        except ImportError:
            from tools.register_all_tools import register_all_additional_tools
        
        register_all_additional_tools(self)
        
        # Register utility tools that bypass UI for ground truth
        self._register_utility_tools()
    
    def _register_utility_tools(self):
        """Register utility tools that query servers directly (bypass UI)"""
        
        async def util_is_agent_server_running() -> Dict[str, Any]:
            """Utility: Check agent server health directly (bypasses UI)"""
            result = await self._util_is_agent_server_running()
            return format_mcp_response(
                result.get("running", False),
                data=result,
                message="Agent server is running" if result.get("running") else "Agent server not reachable"
            )
        
        self.server.add_tool(
            util_is_agent_server_running,
            name="util_is_agent_server_running",
            description="Utility: Check agent server health directly (bypasses UI)"
        )
        
        async def util_get_dnne_server_status() -> Dict[str, Any]:
            """Utility: Get DNNE server status directly (bypasses UI)"""
            result = await self._util_get_dnne_server_status()
            return format_mcp_response(
                result.get("running", False),
                data=result,
                message="DNNE server is running" if result.get("running") else "DNNE server not reachable"
            )
        
        self.server.add_tool(
            util_get_dnne_server_status,
            name="util_get_dnne_server_status",
            description="Utility: Get DNNE server status directly (bypasses UI)"
        )
        
        async def util_find_elements_by_text(text: str, limit: int = 10) -> Dict[str, Any]:
            """Utility: Find DOM elements by text content for debugging"""
            result = await self._util_find_elements_by_text(text, limit)
            return format_mcp_response(
                result.get("found", False),
                data=result,
                message=f"Found {result.get('count', 0)} elements containing '{text}'"
            )
        
        self.server.add_tool(
            util_find_elements_by_text,
            name="util_find_elements_by_text",
            description="Utility: Find DOM elements by text content for debugging"
        )
    
    async def _util_find_elements_by_text(self, text: str, limit: int = 10) -> Dict[str, Any]:
        """
        Utility: Find DOM elements containing specific text (bypasses normal selectors)
        Useful for debugging when selectors aren't working
        
        Args:
            text: Text to search for in elements
            limit: Maximum number of results to return (default 10)
        
        Returns:
            Dict with found elements and their selectors
        """
        if not self.browser_controller or not self.browser_controller.is_browser_window_available():
            return {"found": False, "error": "Browser not available", "count": 0, "elements": []}
        
        try:
            # Escape the search text for JavaScript
            escaped_text = text.replace('\\', '\\\\').replace('"', '\\"')
            
            results = await self.browser_controller.page.evaluate(f"""
                () => {{
                    const searchText = "{escaped_text}";
                    const maxResults = {limit};
                    const allElements = document.querySelectorAll('*');
                    const results = [];
                    let count = 0;
                    
                    for (let el of allElements) {{
                        if (count >= maxResults) break;
                        
                        // Skip script and style elements
                        if (el.tagName === 'SCRIPT' || el.tagName === 'STYLE') continue;
                        
                        // Check if element directly contains the text (not just children)
                        const hasText = el.textContent && el.textContent.includes(searchText);
                        const hasDirectText = el.childNodes && Array.from(el.childNodes).some(
                            node => node.nodeType === 3 && node.textContent.includes(searchText)
                        );
                        
                        if (hasText) {{
                            results.push({{
                                tagName: el.tagName.toLowerCase(),
                                className: el.className || '',
                                id: el.id || '',
                                selector: el.id ? `#${{el.id}}` : 
                                          el.className ? `.${{el.className.split(' ').filter(c => c).join('.')}}` : 
                                          el.tagName.toLowerCase(),
                                text: el.textContent.substring(0, 200),
                                hasDirectText: hasDirectText
                            }});
                            count++;
                        }}
                    }}
                    
                    return {{
                        found: results.length > 0,
                        count: results.length,
                        elements: results
                    }};
                }}
            """)
            
            return results
            
        except Exception as e:
            return {"found": False, "error": str(e), "count": 0, "elements": []}
    
    async def _util_is_agent_server_running(self) -> Dict[str, Any]:
        """
        Utility: Check agent server health directly (bypasses UI)
        Queries http://172.22.160.1:8769/health
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://172.22.160.1:8769/health', timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            "running": True,
                            "healthy": data.get("status") == "healthy",
                            "uptime": data.get("uptime"),
                            "connections": data.get("connections"),
                            "data": data
                        }
                    else:
                        return {"running": False, "error": f"HTTP {resp.status}"}
        except asyncio.TimeoutError:
            return {"running": False, "error": "Connection timeout"}
        except Exception as e:
            return {"running": False, "error": str(e)}
    
    async def _util_get_dnne_server_status(self) -> Dict[str, Any]:
        """
        Utility: Get DNNE server status directly (bypasses UI)
        Queries http://172.22.160.1:8188/api/agent/clients
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://172.22.160.1:8188/api/agent/clients', timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            "running": True,
                            "clients": data.get("clients", []),
                            "connection_status": data.get("connection_status"),
                            "data": data
                        }
                    else:
                        return {"running": False, "error": f"HTTP {resp.status}"}
        except asyncio.TimeoutError:
            return {"running": False, "error": "Connection timeout"}
        except Exception as e:
            return {"running": False, "error": str(e)}
    
    def run(self):
        """Run the MCP server"""
        logger.info("Starting DNNE UI MCP Server")
        logger.info(f"DNNE URL: {self.dnne_url}")
        
        # FastMCP handles stdio transport
        self.server.run("stdio")

def main():
    """Main entry point"""
    server = DNNE_UI_MCPServer()
    server.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)