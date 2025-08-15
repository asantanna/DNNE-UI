"""UI navigation and interaction tools for DNNE UI MCP Server"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import format_mcp_response, parse_menu_path
from utils.js_defs import *
from utils.timing_constants import (
    MENU_TIMEOUT, SELECTOR_TIMEOUT, ANIMATION_DELAY
)
from utils.js_snippets import (
    js_is_dropdown_open,
    js_get_dropdown_items,
    js_get_dropdown_item_details,
    js_click_dropdown_item_by_index,
    js_get_button_state,
    js_execute_custom_code
)

logger = logging.getLogger(__name__)

class UITools:
    """Tools for UI navigation and interaction in DNNE UI"""
    
    def __init__(self, server):
        """
        Initialize UI tools
        
        Args:
            server: DNNE_UI_MCPServer instance for dynamic browser access
        """
        self.server = server
    
    @property
    def browser(self):
        """Get browser controller dynamically from server"""
        return self.server.browser_controller
    
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
            
            # Use centralized tab selectors
            tab_lower = tab.lower()
            if tab_lower not in TAB_SELECTORS:
                return format_mcp_response(
                    False,
                    error=f"Unknown tab: {tab}. Valid options: workflows, nodes"
                )
            
            selector = TAB_SELECTORS[tab_lower]
            
            # Click the tab
            success = await self.browser.click(selector)
            
            if success:
                # Wait for sidebar animation
                await asyncio.sleep(ANIMATION_DELAY)
                
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
            
            # Try to close via X button - fail fast if not found
            close_button = f"{DIALOG} {DIALOG_CLOSE}"
            close_exists = await self.browser.is_visible(close_button)
            
            if not close_exists:
                return format_mcp_response(
                    False, 
                    error=f"Dialog close button not found using selector '{close_button}'. Dialog structure may have changed."
                )
            
            # Click the close button
            success = await self.browser.click(close_button)
            if not success:
                return format_mcp_response(
                    False,
                    error=f"Failed to click dialog close button '{close_button}'"
                )
            
            await asyncio.sleep(ANIMATION_DELAY)
            
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
    # get_error_message has been removed - use take_screenshot to view error dialogs
    
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
    
    async def click_menu_header(self, menu_name: str) -> Dict[str, Any]:
        """
        Click a menu header to open/close the menu
        
        Args:
            menu_name: Name of menu (e.g., "Workflow", "Edit", "DNNE")
        
        Returns:
            MCP response with menu state
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Clicking menu header: {menu_name}")
            
            # Use centralized menu indices
            menu_index = MENU_INDICES.get(menu_name.lower())
            if not menu_index:
                return format_mcp_response(
                    False,
                    error=f"Unknown menu: {menu_name}"
                )
            
            # Get menu selector
            menu_selector = f"{get_menu_item_selector(menu_index)} .p-menubar-item-label"
            
            # Check if submenu is currently visible
            was_open = await self.browser.is_visible(MENU_SUBMENU)
            
            # Click menu header
            success = await self.browser.click(menu_selector)
            
            if success:
                # Wait a bit for animation
                await asyncio.sleep(ANIMATION_DELAY)
                
                # Check new state
                is_open = await self.browser.is_visible(MENU_SUBMENU)
                
                return format_mcp_response(
                    True,
                    data={
                        "menu": menu_name,
                        "was_open": was_open,
                        "is_open": is_open,
                        "toggled": was_open != is_open
                    },
                    message=f"Menu '{menu_name}' is now {'open' if is_open else 'closed'}"
                )
            else:
                return format_mcp_response(
                    False,
                    error=f"Failed to click menu header: {menu_name}"
                )
                
        except Exception as e:
            logger.error(f"Failed to click menu header: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def click_menu_item(self, path: str) -> Dict[str, Any]:
        """
        Click a menu item by path (e.g., 'Workflow/Save As')
        
        This function properly handles menu state - checks if submenu is
        already visible before clicking menu header.
        
        Args:
            path: Menu path like "Workflow/Save As" or "Edit/Undo"
        
        Returns:
            MCP response with success status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Clicking menu item: {path}")
            
            # Parse the menu path
            menu_parts, final_item = parse_menu_path(path)
            
            if not menu_parts:
                return format_mcp_response(
                    False,
                    error=f"Invalid menu path: {path}"
                )
            
            # Get the top menu index from centralized mapping
            top_menu = menu_parts[0].lower()
            if top_menu not in MENU_INDICES:
                return format_mcp_response(
                    False,
                    error=f"Unknown menu: {menu_parts[0]}"
                )
            
            menu_index = MENU_INDICES[top_menu]
            
            # Check if submenu is already visible
            submenu_visible = await self.browser.is_visible(MENU_SUBMENU)
            
            if not submenu_visible:
                # Open menu first
                menu_selector = get_menu_item_selector(menu_index)
                await self.browser.click(f"{menu_selector} .p-menubar-item-label")
                # Wait for submenu to appear
                await self.browser.wait_for_selector(MENU_SUBMENU, timeout=MENU_TIMEOUT)
                await asyncio.sleep(ANIMATION_DELAY)  # Short pause for animation
            
            # Get submenu items for the current menu from centralized mapping
            if top_menu not in SUBMENU_ITEMS:
                return format_mcp_response(
                    False,
                    error=f"No submenu items defined for menu: {top_menu}"
                )
            
            menu_item_indices = SUBMENU_ITEMS[top_menu]
            
            # Find item by index
            item_index = menu_item_indices.get(final_item.lower())
            
            if not item_index:
                return format_mcp_response(
                    False,
                    error=f"Menu item not mapped: {final_item}. Available items: {list(menu_item_indices.keys())}"
                )
            
            # Click by nth-child selector
            item_selector = get_submenu_item_selector(item_index)
            success = await self.browser.click(item_selector)
            
            if success:
                return format_mcp_response(
                    True,
                    data={"path": path},
                    message=f"Clicked menu item: {path}"
                )
            else:
                return format_mcp_response(
                    False,
                    error=f"Failed to click menu item selector: {item_selector}"
                )
                
        except Exception as e:
            logger.error(f"Failed to click menu item: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def click_droplist(self, path: str) -> Dict[str, Any]:
        """
        Click a dropdown to open it (for testing/screenshots)
        
        Args:
            path: Dropdown path like "taskbar/client" or "log_window/filter"
                  Format: location/control
                  
        Returns:
            MCP response with dropdown state
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Clicking dropdown: {path}")
            
            # Parse the path
            parts = path.split('/')
            if len(parts) != 2:
                return format_mcp_response(
                    False,
                    error=f"Invalid droplist path: {path}. Use format 'location/control' (e.g., 'taskbar/client')"
                )
            
            location = parts[0].lower()
            control = parts[1].lower()
            
            # Get the dropdown selector from centralized mapping
            if location not in DROPDOWN_SELECTORS:
                return format_mcp_response(
                    False,
                    error=f"Invalid location: {location}. Valid locations: {', '.join(dropdown_selectors.keys())}"
                )
            
            if control not in DROPDOWN_SELECTORS[location]:
                return format_mcp_response(
                    False,
                    error=f"Invalid control '{control}' for location '{location}'. Valid controls: {', '.join(DROPDOWN_SELECTORS[location].keys())}"
                )
            
            dropdown_selector = DROPDOWN_SELECTORS[location][control]
            
            # Check if dropdown exists
            dropdown_exists = await self.browser.is_visible(dropdown_selector)
            if not dropdown_exists:
                return format_mcp_response(
                    False,
                    error=f"Dropdown not found at {location}/{control} with selector: {dropdown_selector}"
                )
            
            # Check if dropdown is already open by looking for dropdown items
            items_visible_before = await js_is_dropdown_open(self.browser, DROPDOWN_ITEM_SELECTORS)
            
            # Click to toggle dropdown
            logger.debug(f"Clicking dropdown: {dropdown_selector}")
            await self.browser.click(dropdown_selector)
            await asyncio.sleep(ANIMATION_DELAY)  # Wait for animation
            
            # Check new state and count items using same selectors as click_droplist_item
            dropdown_state = await js_get_dropdown_items(self.browser, DROPDOWN_ITEM_SELECTORS)
            
            return format_mcp_response(
                True,
                data={
                    "location": location,
                    "control": control,
                    "was_open": items_visible_before,
                    "is_open": dropdown_state["is_open"],
                    "toggled": items_visible_before != dropdown_state["is_open"],
                    "item_count": dropdown_state["item_count"],
                    "items": dropdown_state["items"]
                },
                message=f"Dropdown '{control}' at '{location}' is now {'open' if dropdown_state['is_open'] else 'closed'}"
            )
                
        except Exception as e:
            logger.error(f"Failed to click dropdown: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def click_droplist_item(self, path: str, item: str) -> Dict[str, Any]:
        """
        Click a dropdown list item
        
        Args:
            path: Dropdown path like "taskbar/client" or "log_window/filter"
                  Format: location/control
            item: The item to select (e.g., "Local", "Tardigrade")
        
        Returns:
            MCP response with success status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Clicking droplist item: {path} -> {item}")
            
            # Parse the path
            parts = path.split('/')
            if len(parts) != 2:
                return format_mcp_response(
                    False,
                    error=f"Invalid droplist path: {path}. Use format 'location/control' (e.g., 'taskbar/client')"
                )
            
            location = parts[0].lower()
            control = parts[1].lower()
            
            # Get the dropdown selector from centralized mapping
            if location not in DROPDOWN_SELECTORS:
                return format_mcp_response(
                    False,
                    error=f"Invalid location: {location}. Valid locations: {', '.join(dropdown_selectors.keys())}"
                )
            
            if control not in DROPDOWN_SELECTORS[location]:
                return format_mcp_response(
                    False,
                    error=f"Invalid control '{control}' for location '{location}'. Valid controls: {', '.join(DROPDOWN_SELECTORS[location].keys())}"
                )
            
            dropdown_selector = DROPDOWN_SELECTORS[location][control]
            
            # Check if dropdown exists
            dropdown_exists = await self.browser.is_visible(dropdown_selector)
            if not dropdown_exists:
                return format_mcp_response(
                    False,
                    error=f"Dropdown not found at {location} with selector: {dropdown_selector}"
                )
            
            # Open dropdown (if not already open)
            dropdown_open = await js_is_dropdown_open(self.browser, DROPDOWN_ITEM_SELECTORS)
            if not dropdown_open:
                logger.debug(f"Opening dropdown: {dropdown_selector}")
                await self.browser.click(dropdown_selector)
                await asyncio.sleep(ANIMATION_DELAY)  # Wait for dropdown animation

            # Import normalize function
            try:
                from ..utils.helpers import normalize_ui_text
            except ImportError:
                from utils.helpers import normalize_ui_text
            
            # Normalize the search text
            normalized_search = normalize_ui_text(item)
            
            # Get all dropdown items and their indices
            items_data = await js_get_dropdown_item_details(self.browser, DROPDOWN_ITEM_SELECTORS)
            
            # Find matching item using normalized comparison
            match_found = None
            logger.debug(f"Looking for normalized: '{normalized_search}' in {len(items_data)} items")
            for item_data in items_data:
                normalized_item = normalize_ui_text(item_data["text"])
                logger.debug(f"  Comparing '{normalized_item}' (from '{item_data['text']}') with '{normalized_search}'")
                if normalized_item == normalized_search:
                    match_found = item_data
                    logger.debug(f"  Match found!")
                    break
            
            # Import constants for tieredmenu
            from utils.js_defs import TIEREDMENU_ITEM, TIEREDMENU_ITEM_LINK
            
            # Click the matched item
            success = False
            if match_found:
                # Special handling for tieredmenu items - need to click the link inside
                if match_found['selector'] == TIEREDMENU_ITEM:
                    # For tieredmenu, click the link element inside the item
                    link_selector = f"{match_found['selector']}:nth-of-type({match_found['index'] + 1}) {TIEREDMENU_ITEM_LINK}"
                    success = await self.browser.click(link_selector)
                else:
                    # Use the standard click method for other dropdown types
                    success = await js_click_dropdown_item_by_index(
                        self.browser,
                        match_found['selector'],
                        match_found['index']
                    )
            
            if success:
                return format_mcp_response(
                    True,
                    data={
                        "location": location,
                        "selected": item
                    },
                    message=f"Selected '{item}' from {location} dropdown"
                )
            else:
                # Close dropdown if selection failed
                await self.browser.click(dropdown_selector)
                return format_mcp_response(
                    False,
                    error=f"Item not found in dropdown: {item}"
                )
                
        except Exception as e:
            logger.error(f"Failed to click droplist item: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def click_button(self, location: str) -> Dict[str, Any]:
        """
        Click a button using location/control convention
        
        Args:
            location: Button location path like "taskbar/export", "canvas/zoom_in", "dialog/confirm"
                     Format: location/control
                     Available locations: taskbar, canvas, dialog, sidebar, log_window
        
        Returns:
            MCP response with success status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Clicking button: {location}")
            
            # Parse the location path
            parts = location.split('/')
            if len(parts) != 2:
                return format_mcp_response(
                    False,
                    error=f"Invalid button location: {location}. Use format 'location/control' (e.g., 'taskbar/export')"
                )
            
            location_part = parts[0].lower()
            control = parts[1].lower()
            
            # Get the button selector from centralized mapping
            if location_part not in BUTTON_SELECTORS:
                return format_mcp_response(
                    False,
                    error=f"Invalid location: {location_part}. Valid locations: {', '.join(BUTTON_SELECTORS.keys())}"
                )
            
            if control not in BUTTON_SELECTORS[location_part]:
                return format_mcp_response(
                    False,
                    error=f"Invalid control '{control}' for location '{location_part}'. Valid controls: {', '.join(BUTTON_SELECTORS[location_part].keys())}"
                )
            
            button_selector = BUTTON_SELECTORS[location_part][control]
            
            # Get button state using the js_get_button_state function
            button_state = await js_get_button_state(self.browser, button_selector)
            
            if not button_state.get('exists'):
                return format_mcp_response(
                    False,
                    error=f"Button not found at {location_part}/{control} with selector: {button_selector}"
                )
            
            if not button_state.get('visible'):
                return format_mcp_response(
                    False,
                    error=f"Button at {location_part}/{control} exists but is not visible"
                )
            
            if button_state.get('disabled'):
                return format_mcp_response(
                    False,
                    data={
                        "location": location_part,
                        "control": control,
                        "disabled": True,
                        "text": button_state.get('text')
                    },
                    error=f"Button '{control}' at '{location_part}' is disabled"
                )
            
            # Click the button
            success = await self.browser.click(button_selector)
            
            if success:
                # Wait for any animation or action to complete
                await asyncio.sleep(ANIMATION_DELAY)
                
                return format_mcp_response(
                    True,
                    data={
                        "location": location_part,
                        "control": control,
                        "clicked": True,
                        "text": button_state.get('text')
                    },
                    message=f"Clicked '{control}' button at '{location_part}'"
                )
            else:
                return format_mcp_response(
                    False,
                    error=f"Failed to click button at {location_part}/{control}"
                )
                
        except Exception as e:
            logger.error(f"Failed to click button: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def get_checkbox_state(self, path: str) -> Dict[str, Any]:
        """
        Get the current state of a checkbox
        
        Args:
            path: Checkbox path like "runner_args_dlg/override"
                  Format: location/control
        
        Returns:
            MCP response with checkbox state (checked, disabled, visible)
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Getting checkbox state: {path}")
            
            # Parse the path
            parts = path.split('/')
            if len(parts) != 2:
                return format_mcp_response(
                    False,
                    error=f"Invalid checkbox path: {path}. Use format 'location/control' (e.g., 'runner_args_dlg/override')"
                )
            
            location = parts[0].lower()
            control = parts[1].lower()
            
            # Get the checkbox selector from centralized mapping
            from utils.js_defs import CHECKBOX_SELECTORS
            
            if location not in CHECKBOX_SELECTORS:
                return format_mcp_response(
                    False,
                    error=f"Invalid location: {location}. Valid locations: {', '.join(CHECKBOX_SELECTORS.keys())}"
                )
            
            if control not in CHECKBOX_SELECTORS[location]:
                return format_mcp_response(
                    False,
                    error=f"Invalid control '{control}' for location '{location}'. Valid controls: {', '.join(CHECKBOX_SELECTORS[location].keys())}"
                )
            
            checkbox_selector = CHECKBOX_SELECTORS[location][control]
            
            # Check if checkbox exists
            exists = await self.browser.is_visible(checkbox_selector)
            if not exists:
                return format_mcp_response(
                    False,
                    error=f"Checkbox not found at {location}/{control} with selector: {checkbox_selector}"
                )
            
            # Get checkbox state
            state = await self.browser.evaluate(f"""
                () => {{
                    const checkbox = document.querySelector('{checkbox_selector}');
                    if (!checkbox) return null;
                    
                    // Handle both input[type="checkbox"] and elements with checkbox role
                    const isInput = checkbox.tagName.toLowerCase() === 'input';
                    
                    return {{
                        checked: isInput ? checkbox.checked : checkbox.getAttribute('aria-checked') === 'true',
                        disabled: checkbox.disabled || checkbox.hasAttribute('disabled') || checkbox.getAttribute('aria-disabled') === 'true',
                        visible: checkbox.offsetParent !== null && window.getComputedStyle(checkbox).display !== 'none'
                    }};
                }}
            """)
            
            if state is None:
                return format_mcp_response(
                    False,
                    error=f"Failed to get state for checkbox at {location}/{control}"
                )
            
            return format_mcp_response(
                True,
                data={
                    "location": location,
                    "control": control,
                    "checked": state.get('checked', False),
                    "disabled": state.get('disabled', False),
                    "visible": state.get('visible', True)
                },
                message=f"Checkbox '{control}' at '{location}' is {'checked' if state.get('checked') else 'unchecked'}"
            )
            
        except Exception as e:
            logger.error(f"Failed to get checkbox state: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def click_checkbox(self, path: str) -> Dict[str, Any]:
        """
        Click a checkbox to toggle its state
        
        Args:
            path: Checkbox path like "runner_args_dlg/override"
                  Format: location/control
        
        Returns:
            MCP response with new checkbox state
            
        Fails if:
            - Checkbox doesn't exist
            - Checkbox is not visible
            - Checkbox is disabled
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Clicking checkbox: {path}")
            
            # First get the current state to validate
            state_result = await self.get_checkbox_state(path)
            
            if not state_result['success']:
                return state_result  # Pass through the error
            
            # Data is merged directly into the response, not under 'data' key
            current_state = state_result
            
            # Check if checkbox is disabled
            if current_state.get('disabled', False):
                return format_mcp_response(
                    False,
                    error=f"Cannot click checkbox '{path}' - it is disabled"
                )
            
            # Check if checkbox is visible
            if not current_state.get('visible', True):
                return format_mcp_response(
                    False,
                    error=f"Cannot click checkbox '{path}' - it is not visible"
                )
            
            # Get the selector to click
            parts = path.split('/')
            location = parts[0].lower()
            control = parts[1].lower()
            
            from utils.js_defs import CHECKBOX_SELECTORS
            checkbox_selector = CHECKBOX_SELECTORS[location][control]
            
            # Click the checkbox
            await self.browser.click(checkbox_selector)
            await asyncio.sleep(ANIMATION_DELAY)
            
            # Get the new state
            new_state_result = await self.get_checkbox_state(path)
            
            if not new_state_result['success']:
                return format_mcp_response(
                    False,
                    error=f"Clicked checkbox but failed to verify new state"
                )
            
            new_state = new_state_result.get('checked', False)
            
            return format_mcp_response(
                True,
                data={
                    "location": location,
                    "control": control,
                    "checked": new_state
                },
                message=f"Checkbox '{control}' at '{location}' is now {'checked' if new_state else 'unchecked'}"
            )
            
        except Exception as e:
            logger.error(f"Failed to click checkbox: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def get_input_text(self, path: str) -> Dict[str, Any]:
        """
        Get the current text value of an input field
        
        Args:
            path: Input path like "runner_args_dlg/cmd_line"
                  Format: location/control
        
        Returns:
            MCP response with input value and state
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Getting input text: {path}")
            
            # Parse the path
            parts = path.split('/')
            if len(parts) != 2:
                return format_mcp_response(
                    False,
                    error=f"Invalid input path: {path}. Use format 'location/control' (e.g., 'runner_args_dlg/cmd_line')"
                )
            
            location = parts[0].lower()
            control = parts[1].lower()
            
            # Get the input selector from centralized mapping
            from utils.js_defs import INPUT_SELECTORS
            
            if location not in INPUT_SELECTORS:
                return format_mcp_response(
                    False,
                    error=f"Invalid location: {location}. Valid locations: {', '.join(INPUT_SELECTORS.keys())}"
                )
            
            if control not in INPUT_SELECTORS[location]:
                return format_mcp_response(
                    False,
                    error=f"Invalid control '{control}' for location '{location}'. Valid controls: {', '.join(INPUT_SELECTORS[location].keys())}"
                )
            
            input_selector = INPUT_SELECTORS[location][control]
            
            # Check if input exists
            exists = await self.browser.is_visible(input_selector)
            if not exists:
                return format_mcp_response(
                    False,
                    error=f"Input field not found at {location}/{control} with selector: {input_selector}"
                )
            
            # Get input state and value
            state = await self.browser.evaluate(f"""
                () => {{
                    const input = document.querySelector('{input_selector}');
                    if (!input) return null;
                    
                    return {{
                        value: input.value || '',
                        disabled: input.disabled || input.hasAttribute('disabled'),
                        readonly: input.readOnly || input.hasAttribute('readonly'),
                        visible: input.offsetParent !== null && window.getComputedStyle(input).display !== 'none'
                    }};
                }}
            """)
            
            if state is None:
                return format_mcp_response(
                    False,
                    error=f"Failed to get state for input at {location}/{control}"
                )
            
            return format_mcp_response(
                True,
                data={
                    "location": location,
                    "control": control,
                    "value": state.get('value', ''),
                    "disabled": state.get('disabled', False),
                    "readonly": state.get('readonly', False),
                    "visible": state.get('visible', True)
                },
                message=f"Input '{control}' at '{location}' has value: {state.get('value', '')}"
            )
            
        except Exception as e:
            logger.error(f"Failed to get input text: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def enter_input_text(self, path: str, text: str, clear_first: bool = True) -> Dict[str, Any]:
        """
        Enter text into an input field
        
        Args:
            path: Input path like "runner_args_dlg/cmd_line"
                  Format: location/control
            text: Text to enter
            clear_first: Whether to clear existing text first (default: True)
        
        Returns:
            MCP response with the final text value
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Entering text into: {path}")
            
            # First get the current state to validate
            state_result = await self.get_input_text(path)
            
            if not state_result['success']:
                return state_result  # Pass through the error
            
            # Data is merged directly into the response, not under 'data' key
            current_state = state_result
            
            # Check if input is disabled
            if current_state.get('disabled', False):
                return format_mcp_response(
                    False,
                    error=f"Cannot enter text in '{path}' - input is disabled"
                )
            
            # Check if input is readonly
            if current_state.get('readonly', False):
                return format_mcp_response(
                    False,
                    error=f"Cannot enter text in '{path}' - input is readonly"
                )
            
            # Check if input is visible
            if not current_state.get('visible', True):
                return format_mcp_response(
                    False,
                    error=f"Cannot enter text in '{path}' - input is not visible"
                )
            
            # Get the selector
            parts = path.split('/')
            location = parts[0].lower()
            control = parts[1].lower()
            
            from utils.js_defs import INPUT_SELECTORS
            input_selector = INPUT_SELECTORS[location][control]
            
            # Focus the input
            await self.browser.click(input_selector)
            await asyncio.sleep(0.1)
            
            # Clear the input if requested
            if clear_first:
                # Select all and delete
                await self.browser.evaluate(f"""
                    () => {{
                        const input = document.querySelector('{input_selector}');
                        if (input) {{
                            input.select();
                            input.value = '';
                            // Trigger input event for frameworks that listen to it
                            input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        }}
                    }}
                """)
                await asyncio.sleep(0.1)
            
            # Type the new text
            await self.browser.type_text(input_selector, text)
            await asyncio.sleep(0.1)
            
            # Get the final value
            final_value = await self.browser.evaluate(f"""
                () => {{
                    const input = document.querySelector('{input_selector}');
                    return input ? input.value : null;
                }}
            """)
            
            if final_value is None:
                return format_mcp_response(
                    False,
                    error=f"Failed to verify text entry in '{path}'"
                )
            
            return format_mcp_response(
                True,
                data={
                    "location": location,
                    "control": control,
                    "value": final_value
                },
                message=f"Entered text in '{control}' at '{location}'"
            )
            
        except Exception as e:
            logger.error(f"Failed to enter text: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def run_javascript(self, code: str, return_result: bool = True) -> Dict[str, Any]:
        """Execute JavaScript code in the current browser context
        
        Args:
            code: JavaScript code to execute
            return_result: Whether to return the result (default True)
        
        Returns:
            MCP response with result or success status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.debug(f"Executing JavaScript: {code[:100]}...")
            
            if return_result:
                result = await js_execute_custom_code(self.browser, code)
                return format_mcp_response(
                    True,
                    data={"result": result},
                    message="JavaScript executed successfully"
                )
            else:
                await js_execute_custom_code(self.browser, code)
                return format_mcp_response(
                    True,
                    message="JavaScript executed successfully (no return value)"
                )
                
        except Exception as e:
            logger.error(f"Failed to execute JavaScript: {e}")
            return format_mcp_response(False, error=str(e))