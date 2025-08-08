"""Client and agent management tools for DNNE UI MCP Server"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import format_mcp_response
from utils.js_defs import *
from utils.timing_constants import ANIMATION_DELAY

logger = logging.getLogger(__name__)

class ClientTools:
    """Tools for managing clients and agents in DNNE UI"""
    
    def __init__(self, server):
        """
        Initialize client tools
        
        Args:
            server: DNNE_UI_MCPServer instance for dynamic browser access
        """
        self.server = server
    
    @property
    def browser(self):
        """Get browser controller dynamically from server"""
        return self.server.browser_controller
    
    async def get_connected_clients(self) -> Dict[str, Any]:
        """
        Get list of all connected clients
        
        Returns:
            MCP response with client list
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info("Getting connected clients")
            
            # Use UITools to get client list from dropdown
            from .ui_tools import UITools
            ui_tools = UITools(self.server)
            
            # Try to open client dropdown
            dropdown_result = await ui_tools.click_droplist("taskbar/client")
            
            if dropdown_result.get("success"):
                clients = dropdown_result.get("items", [])
                # Normalize client names to remove emojis
                try:
                    from ..utils.helpers import normalize_ui_text
                except ImportError:
                    from utils.helpers import normalize_ui_text
                clean_clients = [normalize_ui_text(client) for client in clients]
                
                # Filter out "Local" as it's not a connected client
                connected_clients = [client for client in clean_clients if client != "Local"]
                
                # Close dropdown if it was opened
                if dropdown_result.get("toggled"):
                    await ui_tools.click_droplist("taskbar/client")
                
                return format_mcp_response(
                    True,
                    data={
                        "clients": connected_clients,
                        "count": len(connected_clients)
                    },
                    message=f"Found {len(connected_clients)} connected clients"
                )
            else:
                # No dropdown, check status bar for client count
                status_text = await self.browser.get_text(STATUS_BAR)
                if status_text and "Clients:" in status_text:
                    import re
                    match = re.search(r'Clients:\s*(\d+)', status_text)
                    if match:
                        count = int(match.group(1))
                        return format_mcp_response(
                            True,
                            data={
                                "clients": [],
                                "count": count
                            },
                            message=f"{count} clients connected (names not available)"
                        )
                
                return format_mcp_response(
                    True,
                    data={"clients": [], "count": 0},
                    message="No clients connected"
                )
                
        except Exception as e:
            logger.error(f"Failed to get connected clients: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def select_client(self, name: str, location: str = "taskbar") -> Dict[str, Any]:
        """
        Select a specific client from the specified location
        
        Args:
            name: Client name to select (e.g., "Local", "Tardigrade", "Agent 1", "All Clients")
            location: Where to select from - "taskbar" or "log_window" (default: "taskbar")
        
        Returns:
            MCP response with success status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Selecting client: {name} from {location}")
            
            # Use UITools for dropdown interaction
            from .ui_tools import UITools
            ui_tools = UITools(self.server)
            
            # Determine dropdown path based on location
            if location == "taskbar":
                dropdown_path = "taskbar/client"
            elif location == "log_window":
                dropdown_path = "log_window/filter"
            else:
                return format_mcp_response(
                    False,
                    error=f"Invalid location: {location}. Use 'taskbar' or 'log_window'"
                )
            
            # Open dropdown
            open_result = await ui_tools.click_droplist(dropdown_path)
            if not open_result.get("success"):
                return format_mcp_response(
                    False,
                    error=f"Failed to open dropdown: {open_result.get('error')}"
                )
            
            # Check if client exists in the dropdown
            items = open_result.get("items", [])
            logger.debug(f"Dropdown items found: {items}")
            
            # Normalize items to check for existence
            try:
                from ..utils.helpers import normalize_ui_text
            except ImportError:
                from utils.helpers import normalize_ui_text
            normalized_items = [normalize_ui_text(item) for item in items]
            logger.debug(f"Normalized items: {normalized_items}")
            logger.debug(f"Looking for: '{name}'")
            found = name in normalized_items
            logger.debug(f"Found: {found}")
            
            if not found:
                # Close dropdown
                await ui_tools.click_droplist(dropdown_path)
                return format_mcp_response(
                    False,
                    error=f"Client not found: {name}"
                )
            
            # Select the client
            select_result = await ui_tools.click_droplist_item(dropdown_path, name)
            if select_result.get("success"):
                return format_mcp_response(
                    True,
                    data={
                        "selected": name,
                        "location": location
                    },
                    message=f"Selected client '{name}' from {location}"
                )
            else:
                return format_mcp_response(
                    False,
                    error=f"Failed to select client: {select_result.get('error')}"
                )
                
        except Exception as e:
            logger.error(f"Failed to select client: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def get_selected_client(self) -> Dict[str, Any]:
        """
        Get the currently selected client from the taskbar dropdown
        
        Returns:
            MCP response with selected client name
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info("Getting selected client")
            
            # Get the dropdown selector from centralized mapping (same as click_droplist)
            from utils.js_defs import DROPDOWN_SELECTORS
            dropdown_selector = DROPDOWN_SELECTORS["taskbar"]["client"]
            
            # Get the text content of the dropdown (which shows the selected value)
            selected_text = await self.browser.get_text(dropdown_selector)
            
            if not selected_text:
                # Fallback to evaluating JavaScript directly
                selected_text = await self.browser.evaluate(f"""
                    () => {{
                        const dropdown = document.querySelector('{dropdown_selector}');
                        if (dropdown) {{
                            return dropdown.value || dropdown.textContent?.trim() || 'Local';
                        }}
                        return null;
                    }}
                """)
            
            if not selected_text:
                return format_mcp_response(
                    False,
                    error="Could not read selected client from dropdown"
                )
            
            # The selected text should be the client name
            selected_client = selected_text.strip()
            logger.info(f"Currently selected client: {selected_client}")
            
            return format_mcp_response(
                True,
                data={"selected_client": selected_client},
                message=f"Selected client: {selected_client}"
            )
            
        except Exception as e:
            logger.error(f"Failed to get selected client: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """
        Get the agent connection status from the status bar
        
        Returns:
            MCP response with agent status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info("Getting agent status")
            
            # Look for agent status in the status bar
            status_text = await self.browser.get_text(AGENT_STATUS_BOTTOM)
            
            if not status_text:
                return format_mcp_response(
                    False, 
                    error=f"Status bar not found using selector '{AGENT_STATUS_BOTTOM}'. Status bar may not be visible or selector needs updating."
                )
            
            # Parse the status
            connected = False
            status = "unknown"
            
            if "Connected" in status_text:
                connected = True
                status = "connected"
            elif "Disconnected" in status_text:
                connected = False
                status = "disconnected"
            elif "Connecting" in status_text:
                connected = False
                status = "connecting"
            
            # Extract client count if present
            client_count = 0
            import re
            match = re.search(r'Clients:\s*(\d+)', status_text)
            if match:
                client_count = int(match.group(1))
            
            return format_mcp_response(
                True,
                data={
                    "connected": connected,
                    "status": status,
                    "client_count": client_count,
                    "raw_status": status_text
                },
                message=f"Agent is {status}"
            )
                
        except Exception as e:
            logger.error(f"Failed to get agent status: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def show_all_logs(self) -> Dict[str, Any]:
        """
        Click the Show All Logs button to display logs from all clients
        
        Returns:
            MCP response with success status
        """
        logger.info("show_all_logs called")
        return format_mcp_response(
            False,
            error="Not implemented yet"
        )
    
    async def clear_logs(self) -> Dict[str, Any]:
        """
        Clear the log window
        
        Returns:
            MCP response with success status
        """
        logger.info("clear_logs called")
        return format_mcp_response(
            False,
            error="Not implemented yet"
        )