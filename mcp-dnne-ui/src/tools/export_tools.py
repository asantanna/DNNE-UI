"""Export operation tools for DNNE UI MCP Server"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import format_mcp_response
from utils.js_defs import *
from utils.timing_constants import ANIMATION_DELAY, EXPORT_TIMEOUT
from utils.js_snippets import run_js_snippet_in_browser

logger = logging.getLogger(__name__)

class ExportTools:
    """Tools for export operations in DNNE UI"""
    
    def __init__(self, server):
        """
        Initialize export tools
        
        Args:
            server: DNNE_UI_MCPServer instance for dynamic browser access
        """
        self.server = server
    
    @property
    def browser(self):
        """Get browser controller dynamically from server"""
        return self.server.browser_controller
    
    async def take_screenshot(self, name: str = "dnne_ui") -> Dict[str, Any]:
        """
        Take a screenshot of the DNNE UI
        
        Args:
            name: Name for the screenshot file
        
        Returns:
            MCP response with screenshot path
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            path = await self.browser.take_screenshot(name)
            
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
    
    async def get_export_status(self) -> Dict[str, Any]:
        """
        Check the current export status
        
        Returns:
            MCP response with export status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            # Check for export in progress indicators
            # This could be a progress bar, spinner, or status message
            
            # Check for error dialog
            error_visible = await self.browser.is_visible(DIALOG)
            if error_visible:
                error_text = await self.browser.get_text(DIALOG_CONTENT)
                return format_mcp_response(
                    False,
                    data={
                        "status": "error",
                        "message": error_text
                    }
                )
            
            # Check for success indicators
            # Look for status messages in the UI
            status_text = await self.browser.evaluate("""
                () => {
                    // Look for status messages in various places
                    const statusBar = document.querySelector('.status-bar');
                    if (statusBar) {
                        const text = statusBar.textContent;
                        if (text.includes('Export')) {
                            return text;
                        }
                    }
                    
                    // Check for toast notifications
                    const toast = document.querySelector('.p-toast-message');
                    if (toast) {
                        return toast.textContent;
                    }
                    
                    return null;
                }
            """)
            
            if status_text:
                return format_mcp_response(
                    True,
                    data={
                        "status": "completed" if "success" in status_text.lower() else "in_progress",
                        "message": status_text
                    }
                )
            
            return format_mcp_response(
                True,
                data={
                    "status": "idle",
                    "message": "No export in progress"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get export status: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def set_run_after_export(self, enabled: bool) -> Dict[str, Any]:
        """
        Set whether to run the workflow after export
        
        Args:
            enabled: Whether to enable run after export
        
        Returns:
            MCP response with success status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Setting run after export to: {enabled}")
            
            # Use the defined selector for the run after export checkbox
            checkbox_selector = RUN_AFTER_EXPORT
            
            checkbox_exists = await self.browser.is_visible(checkbox_selector)
            if not checkbox_exists:
                return format_mcp_response(
                    False,
                    error="Run after export checkbox not found"
                )
            
            # Check if checkbox is disabled
            is_disabled = await run_js_snippet_in_browser(
                self.browser, 
                "is_checkbox_disabled", 
                {"selector": checkbox_selector}
            )
            
            if is_disabled:
                return format_mcp_response(
                    False,
                    error="Cannot set run after export - checkbox is disabled (Local export selected)"
                )
            
            # Get current state
            is_checked = await run_js_snippet_in_browser(
                self.browser,
                "is_checkbox_checked",
                {"selector": checkbox_selector}
            )
            
            # Toggle if needed
            if is_checked != enabled:
                await self.browser.click(checkbox_selector)
                await asyncio.sleep(ANIMATION_DELAY)
            
            return format_mcp_response(
                True,
                data={"run_after_export": enabled},
                message=f"Run after export set to: {enabled}"
            )
            
        except Exception as e:
            logger.error(f"Failed to set run after export: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def wait_for_export_completion(self, timeout: int = EXPORT_TIMEOUT) -> Dict[str, Any]:
        """
        Wait for export operation to complete
        
        Args:
            timeout: Maximum time to wait in milliseconds
        
        Returns:
            MCP response with completion status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Waiting for export to complete (timeout: {timeout}ms)")
            
            start_time = asyncio.get_event_loop().time()
            timeout_seconds = timeout / 1000
            
            while (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
                # Check for completion indicators
                
                # Check for error dialog
                error_visible = await self.browser.is_visible(DIALOG)
                if error_visible:
                    error_text = await self.browser.get_text(DIALOG_CONTENT)
                    return format_mcp_response(
                        False,
                        error=f"Export failed: {error_text}"
                    )
                
                # Check for success message
                success = await self.browser.evaluate("""
                    () => {
                        // Check for success toast
                        const toast = document.querySelector('.p-toast-message-success');
                        if (toast && toast.textContent.toLowerCase().includes('export')) {
                            return true;
                        }
                        
                        // Check status bar
                        const statusBar = document.querySelector('.status-bar');
                        if (statusBar && statusBar.textContent.toLowerCase().includes('export complete')) {
                            return true;
                        }
                        
                        return false;
                    }
                """)
                
                if success:
                    return format_mcp_response(
                        True,
                        message="Export completed successfully"
                    )
                
                # Wait a bit before checking again
                await asyncio.sleep(1)  # Keep 1s for polling interval
            
            return format_mcp_response(
                False,
                error=f"Export timed out after {timeout}ms"
            )
            
        except Exception as e:
            logger.error(f"Failed while waiting for export: {e}")
            return format_mcp_response(False, error=str(e))