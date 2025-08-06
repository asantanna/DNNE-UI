"""Browser lifecycle management tools for DNNE UI MCP Server"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from browser_controller import BrowserController
from utils.helpers import format_mcp_response

if TYPE_CHECKING:
    from dnne_ui_mcp_server import DNNE_UI_MCPServer

logger = logging.getLogger(__name__)


class LifecycleTools:
    """Tools for managing browser lifecycle in DNNE UI"""
    
    def __init__(self, server: "DNNE_UI_MCPServer"):
        """
        Initialize lifecycle tools
        
        Args:
            server: DNNE_UI_MCPServer instance for browser access
        """
        self.server = server
    
    async def initialize_browser(self) -> Dict[str, Any]:
        """Initialize the browser and navigate to DNNE UI"""
        try:
            # Check if browser exists AND is healthy
            if self.server.browser_controller:
                # Check if browser process is active
                if self.server.browser_controller.is_playwright_browser_process_active():
                    # Check if browser window is available
                    if self.server.browser_controller.is_browser_window_available():
                        # Check if it's responsive to JavaScript
                        if await self.server.browser_controller.is_javascript_executable():
                            return format_mcp_response(True, message="Browser already initialized and healthy")
                        else:
                            logger.info("Browser window exists but not responsive to JavaScript, cleaning up...")
                    else:
                        logger.info("Browser process active but window closed, cleaning up...")
                else:
                    logger.info("Browser controller exists but process not active, cleaning up...")
                
                # Browser exists but not healthy
                await self.server.browser_controller.cleanup()
                self.server.browser_controller = None
            
            # Create new browser instance
            self.server.browser_controller = BrowserController(
                dnne_url=self.server.dnne_url,
                headless=self.server.headless
            )
            await self.server.browser_controller.initialize()
            return format_mcp_response(True, message="Browser initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            self.server.browser_controller = None  # Clear reference on failure
            return format_mcp_response(False, error=str(e))
    
    async def shut_down_browser_automation(self) -> Dict[str, Any]:
        """Shut down browser automation and free all resources"""
        try:
            if self.server.browser_controller:
                await self.server.browser_controller.cleanup()
                self.server.browser_controller = None
            return format_mcp_response(True, message="Browser automation shut down successfully")
        except Exception as e:
            logger.error(f"Failed to shut down browser automation: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def is_browser_running(self) -> Dict[str, Any]:
        """Check if browser window is available"""
        if not self.server.browser_controller:
            return format_mcp_response(False, data={
                "process_active": False,
                "window_available": False
            }, message="Browser not initialized")
        
        process_active = self.server.browser_controller.is_playwright_browser_process_active()
        window_available = self.server.browser_controller.is_browser_window_available()
        
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
    
    async def restart_browser(self) -> Dict[str, Any]:
        """Restart browser for recovery"""
        try:
            # Always clean up existing browser first if it exists
            if self.server.browser_controller:
                try:
                    await self.server.browser_controller.cleanup()
                except Exception as cleanup_error:
                    logger.warning(f"Error during browser cleanup: {cleanup_error}")
                self.server.browser_controller = None
            
            # Now initialize a fresh browser
            return await self.initialize_browser()
                
        except Exception as e:
            logger.error(f"Failed to restart browser: {e}")
            self.server.browser_controller = None
            return format_mcp_response(False, error=str(e))