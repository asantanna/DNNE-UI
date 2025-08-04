"""Enhanced error handling with diagnostics for DNNE UI MCP Server"""

import asyncio
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from ..browser_controller import BrowserController

logger = logging.getLogger(__name__)

class ErrorDiagnostics:
    """Collect and format error diagnostics"""
    
    def __init__(self, browser_controller: Optional["BrowserController"] = None):
        self.browser = browser_controller
        self.error_dir = Path("error_diagnostics")
        self.error_dir.mkdir(exist_ok=True)
        
    async def capture_error_state(
        self, 
        error: Exception,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Capture comprehensive error state including screenshot
        
        Args:
            error: The exception that occurred
            operation: Name of the operation that failed
            context: Additional context information
            
        Returns:
            Dictionary with error diagnostics
        """
        timestamp = datetime.now().isoformat()
        error_id = f"{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        diagnostics = {
            "error_id": error_id,
            "timestamp": timestamp,
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        # Capture screenshot if browser is available
        if self.browser and self.browser.page:
            try:
                screenshot_path = self.error_dir / f"{error_id}_screenshot.png"
                await self.browser.page.screenshot(path=str(screenshot_path))
                diagnostics["screenshot"] = str(screenshot_path)
                logger.info(f"Error screenshot saved: {screenshot_path}")
            except Exception as e:
                logger.warning(f"Failed to capture error screenshot: {e}")
                diagnostics["screenshot"] = None
        
        # Capture UI state if possible
        if self.browser and self.browser.page:
            try:
                ui_state = await self.browser.evaluate("""
                    () => {
                        return {
                            url: window.location.href,
                            title: document.title,
                            has_dialog: document.querySelector('.p-dialog') !== null,
                            has_error: document.querySelector('.p-toast-message-error') !== null,
                            sidebar_open: document.querySelector('.sidebar-content-container') !== null,
                            canvas_nodes: window.app?.canvas?.graph?.nodes?.length || 0
                        };
                    }
                """)
                diagnostics["ui_state"] = ui_state
            except Exception as e:
                logger.warning(f"Failed to capture UI state: {e}")
                diagnostics["ui_state"] = None
        
        # Save diagnostics to file
        try:
            diagnostics_file = self.error_dir / f"{error_id}_diagnostics.json"
            with open(diagnostics_file, 'w') as f:
                json.dump(diagnostics, f, indent=2, default=str)
            logger.info(f"Error diagnostics saved: {diagnostics_file}")
            diagnostics["diagnostics_file"] = str(diagnostics_file)
        except Exception as e:
            logger.error(f"Failed to save diagnostics: {e}")
        
        return diagnostics
    
    def format_error_response(
        self,
        error: Exception,
        operation: str,
        diagnostics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format a comprehensive error response for MCP
        
        Args:
            error: The exception that occurred
            operation: Name of the operation that failed
            diagnostics: Optional diagnostics data
            
        Returns:
            Formatted error response
        """
        response = {
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "operation": operation
        }
        
        if diagnostics:
            response["error_id"] = diagnostics.get("error_id")
            response["screenshot"] = diagnostics.get("screenshot")
            response["diagnostics_file"] = diagnostics.get("diagnostics_file")
            
            # Add troubleshooting suggestions based on error type
            suggestions = self.get_troubleshooting_suggestions(error, diagnostics)
            if suggestions:
                response["suggestions"] = suggestions
        
        return response
    
    def get_troubleshooting_suggestions(
        self,
        error: Exception,
        diagnostics: Dict[str, Any]
    ) -> list[str]:
        """
        Generate troubleshooting suggestions based on error type
        
        Args:
            error: The exception that occurred
            diagnostics: Error diagnostics
            
        Returns:
            List of troubleshooting suggestions
        """
        suggestions = []
        error_msg = str(error).lower()
        
        # Connection errors
        if "connection" in error_msg or "navigate" in error_msg:
            suggestions.append("Ensure DNNE server is running on Windows")
            suggestions.append("Check firewall settings allow connection on port 8188")
            suggestions.append("Verify URL is correct (default: http://172.22.160.1:8188)")
        
        # Element not found errors
        if "selector" in error_msg or "element" in error_msg or "timeout" in error_msg:
            suggestions.append("UI may have changed - verify selectors are up to date")
            suggestions.append("Increase timeout values for slow connections")
            suggestions.append("Check if a dialog is blocking the UI")
        
        # Export errors
        if "export" in error_msg or "widget" in error_msg:
            suggestions.append("Check workflow is valid before exporting")
            suggestions.append("Verify all required nodes are present")
            suggestions.append("Clear any error dialogs before retrying")
        
        # Browser errors
        if "browser" in error_msg or "page" in error_msg:
            suggestions.append("Try restarting the browser with cleanup_browser() then initialize_browser()")
            suggestions.append("Check system resources (memory/CPU)")
            suggestions.append("Ensure Chromium is properly installed")
        
        # Dialog present
        if diagnostics.get("ui_state", {}).get("has_dialog"):
            suggestions.append("A dialog is open - use dismiss_dialog() to close it")
        
        # Error toast present
        if diagnostics.get("ui_state", {}).get("has_error"):
            suggestions.append("An error message is visible - use get_error_message() to read it")
        
        return suggestions


class RecoverableError(Exception):
    """Exception that indicates the operation can be retried"""
    pass


class NonRecoverableError(Exception):
    """Exception that indicates the operation cannot be retried"""
    pass


async def with_error_handling(
    operation_name: str,
    func,
    browser_controller: Optional["BrowserController"] = None,
    max_retries: int = 3,
    capture_diagnostics: bool = True
):
    """
    Execute a function with comprehensive error handling
    
    Args:
        operation_name: Name of the operation for logging
        func: Async function to execute
        browser_controller: Optional browser controller for diagnostics
        max_retries: Maximum retry attempts
        capture_diagnostics: Whether to capture error diagnostics
        
    Returns:
        Result of the function or error response
    """
    from .helpers import retry_with_backoff
    
    diagnostics_handler = ErrorDiagnostics(browser_controller) if capture_diagnostics else None
    
    try:
        # Execute with retry logic
        result = await retry_with_backoff(
            func,
            max_retries=max_retries,
            retry_on=(RecoverableError, TimeoutError, ConnectionError)
        )
        return result
        
    except Exception as e:
        logger.error(f"Operation '{operation_name}' failed: {e}")
        
        # Capture diagnostics if enabled
        if diagnostics_handler:
            try:
                diagnostics = await diagnostics_handler.capture_error_state(
                    error=e,
                    operation=operation_name
                )
                return diagnostics_handler.format_error_response(e, operation_name, diagnostics)
            except Exception as diag_error:
                logger.error(f"Failed to capture diagnostics: {diag_error}")
        
        # Return basic error response if diagnostics failed
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "operation": operation_name
        }