#!/usr/bin/env python3
"""DNNE UI MCP Server - High-level automation for DNNE UI"""

import asyncio
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
    from .utils.selectors import *
    from .utils.state_manager import StateManager
    from .utils.error_handler import ErrorDiagnostics, with_error_handling
except ImportError:
    # For direct script execution
    from browser_controller import BrowserController
    from utils.helpers import setup_logging, get_env_var, format_mcp_response
    from utils.selectors import *
    from utils.state_manager import StateManager
    from utils.error_handler import ErrorDiagnostics, with_error_handling

# Load environment variables
load_dotenv()

# Setup logging
setup_logging(get_env_var("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

class DNNEUIMCPServer:
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
        
        # State management with persistence
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
                if not self.browser_controller:
                    self.browser_controller = BrowserController(
                        dnne_url=self.dnne_url,
                        headless=self.headless
                    )
                    await self.browser_controller.initialize()
                    return format_mcp_response(True, message="Browser initialized successfully")
                else:
                    return format_mcp_response(True, message="Browser already initialized")
            except Exception as e:
                logger.error(f"Failed to initialize browser: {e}")
                return format_mcp_response(False, error=str(e))
        
        self.server.add_tool(
            initialize_browser,
            name="initialize_browser",
            description="Initialize the browser and navigate to DNNE UI"
        )
        
        async def cleanup_browser() -> Dict[str, Any]:
            """Clean up browser resources"""
            try:
                if self.browser_controller:
                    await self.browser_controller.cleanup()
                    self.browser_controller = None
                return format_mcp_response(True, message="Browser cleaned up")
            except Exception as e:
                logger.error(f"Failed to cleanup browser: {e}")
                return format_mcp_response(False, error=str(e))
        
        self.server.add_tool(
            cleanup_browser,
            name="cleanup_browser",
            description="Clean up browser resources"
        )
        
        async def restart_browser() -> Dict[str, Any]:
            """Restart browser for recovery"""
            try:
                if self.browser_controller:
                    success = await self.browser_controller.restart_browser()
                    if success:
                        # Update error diagnostics with new browser
                        self.error_diagnostics.browser = self.browser_controller
                        return format_mcp_response(True, message="Browser restarted successfully")
                    else:
                        return format_mcp_response(False, error="Browser restart failed")
                else:
                    # Initialize if not existing
                    return await initialize_browser()
                    
            except Exception as e:
                logger.error(f"Failed to restart browser: {e}")
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
            try:
                if not self.browser_controller:
                    return format_mcp_response(False, error="Browser not initialized")
                
                # Open workflows sidebar if needed
                if not self.state["sidebar_open"]:
                    await self.browser_controller.click(WORKFLOWS_TAB)
                    await asyncio.sleep(1)  # Wait for sidebar animation
                    self.state["sidebar_open"] = True
                
                # Click on the workflow
                workflow_selector = get_workflow_selector(name)
                success = await self.browser_controller.click(workflow_selector)
                
                if success:
                    self.state["current_workflow"] = name
                    return format_mcp_response(
                        True, 
                        data={"workflow_name": name},
                        message=f"Loaded workflow: {name}"
                    )
                else:
                    return format_mcp_response(
                        False, 
                        error=f"Failed to find workflow: {name}"
                    )
                    
            except Exception as e:
                logger.error(f"Failed to load workflow: {e}")
                return format_mcp_response(False, error=str(e))
        
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
                
                # Try to get from window title or tab
                title = await self.browser_controller.evaluate("document.title")
                
                # Extract workflow name from title if present
                if title and "Unsaved Workflow" not in title:
                    self.state["current_workflow"] = title
                
                return format_mcp_response(
                    True,
                    data={"workflow_name": self.state["current_workflow"]}
                )
                
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
                
                # TODO: Set run_after checkbox if needed
                # This will need the actual selector once identified
                
                # Click export button
                success = await self.browser_controller.click(EXPORT_BUTTON)
                
                if not success:
                    return format_mcp_response(False, error="Failed to click export button")
                
                # Wait for export to complete (adjust timeout as needed)
                await asyncio.sleep(3)
                
                # Check for errors
                error_visible = await self.browser_controller.is_visible(DIALOG)
                if error_visible:
                    error_text = await self.browser_controller.get_text(DIALOG_CONTENT)
                    return format_mcp_response(
                        False,
                        error=f"Export failed: {error_text}"
                    )
                
                return format_mcp_response(
                    True,
                    message="Workflow exported successfully"
                )
                
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
        
        async def check_ui_health() -> Dict[str, Any]:
            """Check if the DNNE UI is healthy and responsive"""
            try:
                if not self.browser_controller:
                    return format_mcp_response(False, error="Browser not initialized")
                
                issues = []
                
                # Check for key UI elements
                if not await self.browser_controller.is_visible(".side-bar-button"):
                    issues.append("Sidebar buttons not visible")
                
                if not await self.browser_controller.is_visible(".comfyui-menu"):
                    issues.append("Menu bar not visible")
                
                # Check for error dialogs
                if await self.browser_controller.is_visible(DIALOG):
                    issues.append("Error dialog is open")
                
                # Check agent status
                agent_text = await self.browser_controller.get_text(".status-bar")
                if agent_text and "Disconnected" in agent_text:
                    issues.append("Agent is disconnected")
                
                healthy = len(issues) == 0
                
                return format_mcp_response(
                    healthy,
                    data={"healthy": healthy, "issues": issues},
                    message="UI is healthy" if healthy else f"UI has {len(issues)} issues"
                )
                
            except Exception as e:
                logger.error(f"Failed to check UI health: {e}")
                return format_mcp_response(False, error=str(e))
        
        self.server.add_tool(
            check_ui_health,
            name="check_ui_health",
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
                tools = WorkflowTools(self.browser_controller, self.state)
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
                tools = WorkflowTools(self.browser_controller, self.state)
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
                tools = WorkflowTools(self.browser_controller, self.state)
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
                tools = WorkflowTools(self.browser_controller, self.state)
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
    
    def run(self):
        """Run the MCP server"""
        logger.info("Starting DNNE UI MCP Server")
        logger.info(f"DNNE URL: {self.dnne_url}")
        
        # FastMCP handles stdio transport
        self.server.run("stdio")

def main():
    """Main entry point"""
    server = DNNEUIMCPServer()
    server.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)