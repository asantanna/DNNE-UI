"""Client and agent management tools for DNNE UI MCP Server"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
try:
    from ..utils.helpers import format_mcp_response
    from ..utils.selectors import *
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.helpers import format_mcp_response
    from utils.selectors import *

logger = logging.getLogger(__name__)

class ClientTools:
    """Tools for managing clients and agents in DNNE UI"""
    
    def __init__(self, server, state: Dict[str, Any]):
        """
        Initialize client tools
        
        Args:
            server: DNNE_UI_MCPServer instance for dynamic browser access
            state: Shared state dictionary
        """
        self.server = server
        self.state = state
    
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
            
            # Use the client dropdown selector
            client_selector = CLIENT_DROPDOWN
            
            # First try to find the dropdown
            dropdown_exists = await self.browser.is_visible(client_selector)
            
            if dropdown_exists:
                # Click to open dropdown
                await self.browser.click(client_selector)
                await asyncio.sleep(0.5)
                
                # Get all client options
                clients = await self.browser.evaluate("""
                    () => {
                        // Try different selector patterns
                        let options = document.querySelectorAll('.p-dropdown-item');
                        if (!options.length) {
                            options = document.querySelectorAll('option');
                        }
                        if (!options.length) {
                            options = document.querySelectorAll('[role="option"]');
                        }
                        
                        return Array.from(options).map(opt => 
                            opt.textContent?.trim() || opt.value || ''
                        ).filter(text => text);
                    }
                """)
                
                # Close dropdown
                await self.browser.click(client_selector)
                
                return format_mcp_response(
                    True,
                    data={
                        "clients": clients or [],
                        "count": len(clients) if clients else 0
                    },
                    message=f"Found {len(clients) if clients else 0} connected clients"
                )
            else:
                # No dropdown, check status bar for client count
                status_text = await self.browser.get_text(".status-bar")
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
            
            # Determine selector based on location
            if location == "taskbar":
                # Client dropdown in taskbar (was export-target-dropdown)
                client_selector = ".export-target-dropdown, .client-dropdown"
            elif location == "log_window":
                # Client selector in log window
                client_selector = ".log-client-dropdown, .client-select"
            else:
                return format_mcp_response(
                    False,
                    error=f"Invalid location: {location}. Use 'taskbar' or 'log_window'"
                )
            
            dropdown_exists = await self.browser.is_visible(client_selector)
            
            if not dropdown_exists:
                return format_mcp_response(
                    False,
                    error="Client dropdown not found"
                )
            
            # Open dropdown
            await self.browser.click(client_selector)
            await asyncio.sleep(0.5)
            
            # Find and click the specific client
            success = await self.browser.evaluate(f"""
                () => {{
                    const options = document.querySelectorAll('.p-dropdown-item, option, [role="option"]');
                    for (let opt of options) {{
                        if (opt.textContent?.trim() === '{name}') {{
                            opt.click();
                            return true;
                        }}
                    }}
                    return false;
                }}
            """)
            
            if success:
                # Update state with selected client
                self.state["selected_client"] = name
                
                return format_mcp_response(
                    True,
                    data={
                        "selected": name,
                        "location": location
                    },
                    message=f"Selected client '{name}' from {location}"
                )
            else:
                # Close dropdown if selection failed
                await self.browser.click(client_selector)
                return format_mcp_response(
                    False,
                    error=f"Client not found: {name}"
                )
                
        except Exception as e:
            logger.error(f"Failed to select client: {e}")
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
            status_text = await self.browser.get_text(STATUS_BAR)
            
            if not status_text:
                return format_mcp_response(
                    False, 
                    error=f"Status bar not found using selector '{STATUS_BAR}'. Status bar may not be visible or selector needs updating."
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