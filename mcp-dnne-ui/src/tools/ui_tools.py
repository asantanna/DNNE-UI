"""UI navigation and interaction tools for DNNE UI MCP Server"""

import asyncio
import logging
from typing import Dict, Any, Optional
try:
    from ..utils.helpers import format_mcp_response, parse_menu_path
    from ..utils.selectors import *
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.helpers import format_mcp_response, parse_menu_path
    from utils.selectors import *

logger = logging.getLogger(__name__)

class UITools:
    """Tools for UI navigation and interaction in DNNE UI"""
    
    def __init__(self, browser_controller, state: Dict[str, Any]):
        """
        Initialize UI tools
        
        Args:
            browser_controller: BrowserController instance
            state: Shared state dictionary
        """
        self.browser = browser_controller
        self.state = state
    
    async def open_sidebar_tab(self, tab: str) -> Dict[str, Any]:
        """
        Open a specific sidebar tab
        
        Args:
            tab: Tab name (workflows, nodes, models, queue)
        
        Returns:
            MCP response with success status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Opening sidebar tab: {tab}")
            
            # Map tab names to selectors
            tab_selectors = {
                "workflows": WORKFLOWS_TAB,
                "nodes": NODE_LIBRARY_TAB,
                "node_library": NODE_LIBRARY_TAB,
                "models": MODEL_LIBRARY_TAB,  # To be removed
                "queue": QUEUE_TAB  # To be removed
            }
            
            tab_lower = tab.lower()
            if tab_lower not in tab_selectors:
                return format_mcp_response(
                    False,
                    error=f"Unknown tab: {tab}. Valid options: workflows, nodes"
                )
            
            selector = tab_selectors[tab_lower]
            
            # Click the tab
            success = await self.browser.click(selector)
            
            if success:
                # Wait for sidebar animation
                await asyncio.sleep(0.5)
                self.state["sidebar_open"] = True
                self.state["sidebar_tab"] = tab_lower
                
                return format_mcp_response(
                    True,
                    data={"tab": tab_lower},
                    message=f"Opened {tab} sidebar"
                )
            else:
                return format_mcp_response(
                    False,
                    error=f"Failed to open {tab} sidebar"
                )
                
        except Exception as e:
            logger.error(f"Failed to open sidebar tab: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def open_menu(self, path: str) -> Dict[str, Any]:
        """
        Open a menu item by path
        
        Args:
            path: Menu path like "Workflow/Save As" or "Edit/Undo"
        
        Returns:
            MCP response with success status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Opening menu: {path}")
            
            # Parse the menu path
            menu_parts, final_item = parse_menu_path(path)
            
            if not menu_parts:
                return format_mcp_response(
                    False,
                    error=f"Invalid menu path: {path}"
                )
            
            # Map top-level menu names to indices
            menu_indices = {
                "workflow": 1,
                "edit": 2,
                "dnne": 3,
                "help": 4  # If it exists
            }
            
            # Get the top menu index
            top_menu = menu_parts[0].lower()
            if top_menu not in menu_indices:
                return format_mcp_response(
                    False,
                    error=f"Unknown menu: {menu_parts[0]}"
                )
            
            menu_index = menu_indices[top_menu]
            
            # Click the top menu
            menu_selector = get_menu_item_selector(menu_index)
            await self.browser.click(f"{menu_selector} .p-menubar-item-label")
            await asyncio.sleep(0.3)
            
            # If there are nested menus, navigate them
            # For now, we'll handle single-level submenus
            
            # Find and click the final item
            success = await self.browser.evaluate(f"""
                () => {{
                    const items = document.querySelectorAll('.p-menubar-submenu .p-menubar-item-label');
                    for (let item of items) {{
                        if (item.textContent?.trim() === '{final_item}') {{
                            item.click();
                            return true;
                        }}
                    }}
                    return false;
                }}
            """)
            
            if success:
                return format_mcp_response(
                    True,
                    data={"path": path},
                    message=f"Opened menu: {path}"
                )
            else:
                # Close menu if item not found
                await self.browser.evaluate("document.body.click()")
                return format_mcp_response(
                    False,
                    error=f"Menu item not found: {final_item}"
                )
                
        except Exception as e:
            logger.error(f"Failed to open menu: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def dismiss_dialog(self) -> Dict[str, Any]:
        """
        Dismiss any open dialog or error message
        
        Returns:
            MCP response with dialog info
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info("Dismissing dialog")
            
            # Check if dialog is visible
            dialog_visible = await self.browser.is_visible(DIALOG)
            
            if not dialog_visible:
                return format_mcp_response(
                    True,
                    data={"dialog_type": "none"},
                    message="No dialog to dismiss"
                )
            
            # Get dialog type/title
            dialog_title = await self.browser.get_text(f"{DIALOG} .p-dialog-title")
            dialog_type = "unknown"
            
            if dialog_title:
                if "error" in dialog_title.lower():
                    dialog_type = "error"
                elif "warning" in dialog_title.lower():
                    dialog_type = "warning"
                elif "confirm" in dialog_title.lower():
                    dialog_type = "confirmation"
                else:
                    dialog_type = "info"
            
            # Try to close via X button
            close_button = f"{DIALOG} {DIALOG_CLOSE}"
            close_exists = await self.browser.is_visible(close_button)
            
            if close_exists:
                await self.browser.click(close_button)
            else:
                # Try OK/Cancel buttons
                ok_button = f"{DIALOG_FOOTER} button:has-text('OK'), {DIALOG_FOOTER} button:has-text('Close')"
                cancel_button = f"{DIALOG_FOOTER} button:has-text('Cancel')"
                
                if await self.browser.is_visible(ok_button):
                    await self.browser.click(ok_button)
                elif await self.browser.is_visible(cancel_button):
                    await self.browser.click(cancel_button)
                else:
                    # Try pressing Escape
                    await self.browser.evaluate("""
                        document.dispatchEvent(new KeyboardEvent('keydown', {
                            key: 'Escape',
                            bubbles: true
                        }));
                    """)
            
            await asyncio.sleep(0.5)
            
            # Verify dialog is gone
            still_visible = await self.browser.is_visible(DIALOG)
            
            if not still_visible:
                return format_mcp_response(
                    True,
                    data={
                        "dialog_type": dialog_type,
                        "dialog_title": dialog_title
                    },
                    message=f"Dismissed {dialog_type} dialog"
                )
            else:
                return format_mcp_response(
                    False,
                    error="Failed to dismiss dialog"
                )
                
        except Exception as e:
            logger.error(f"Failed to dismiss dialog: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def get_error_message(self) -> Dict[str, Any]:
        """
        Get the current error dialog message if any
        
        Returns:
            MCP response with error details
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info("Getting error message")
            
            # Check for dialog
            dialog_visible = await self.browser.is_visible(DIALOG)
            
            if not dialog_visible:
                # Check for toast notifications
                toast_visible = await self.browser.is_visible(".p-toast-message-error")
                
                if toast_visible:
                    toast_text = await self.browser.get_text(".p-toast-message-error")
                    return format_mcp_response(
                        True,
                        data={
                            "has_error": True,
                            "title": "Toast Error",
                            "message": toast_text,
                            "type": "toast"
                        }
                    )
                
                return format_mcp_response(
                    True,
                    data={
                        "has_error": False,
                        "title": None,
                        "message": None
                    },
                    message="No error dialog present"
                )
            
            # Get dialog details
            title = await self.browser.get_text(f"{DIALOG} .p-dialog-title")
            message = await self.browser.get_text(DIALOG_CONTENT)
            
            # Determine if it's an error
            is_error = False
            if title and "error" in title.lower():
                is_error = True
            elif message and any(word in message.lower() for word in ["error", "failed", "exception"]):
                is_error = True
            
            return format_mcp_response(
                True,
                data={
                    "has_error": is_error,
                    "title": title,
                    "message": message,
                    "type": "dialog"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get error message: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def wait_for_ui_ready(self, timeout: int = 10) -> Dict[str, Any]:
        """
        Wait for the UI to be fully loaded and ready
        
        Args:
            timeout: Maximum time to wait in seconds
        
        Returns:
            MCP response with ready status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Waiting for UI to be ready (timeout: {timeout}s)")
            
            start_time = asyncio.get_event_loop().time()
            
            # Wait for key elements
            ready = await self.browser.wait_for_ui_ready(timeout * 1000)
            
            if ready:
                elapsed = asyncio.get_event_loop().time() - start_time
                return format_mcp_response(
                    True,
                    data={
                        "ready": True,
                        "load_time": elapsed
                    },
                    message=f"UI ready in {elapsed:.2f} seconds"
                )
            else:
                return format_mcp_response(
                    False,
                    data={
                        "ready": False,
                        "load_time": timeout
                    },
                    error=f"UI not ready after {timeout} seconds"
                )
                
        except Exception as e:
            logger.error(f"Failed while waiting for UI: {e}")
            return format_mcp_response(False, error=str(e))