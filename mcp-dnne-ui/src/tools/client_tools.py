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
    
    def __init__(self, browser_controller, state: Dict[str, Any]):
        """
        Initialize client tools
        
        Args:
            browser_controller: BrowserController instance
            state: Shared state dictionary
        """
        self.browser = browser_controller
        self.state = state
    
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
            
            # Look for client dropdown selector
            # The actual selector needs to be determined from the UI
            client_selector = ".client-dropdown, select[name*='client'], .client-selector"
            
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
    
    async def select_client(self, name: str) -> Dict[str, Any]:
        """
        Select a specific client from the dropdown
        
        Args:
            name: Name of the client to select
        
        Returns:
            MCP response with success status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info(f"Selecting client: {name}")
            
            # Find and click client dropdown
            client_selector = ".client-dropdown, select[name*='client'], .client-selector"
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
                self.state["selected_client"] = name
                return format_mcp_response(
                    True,
                    data={"selected": name},
                    message=f"Selected client: {name}"
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
            status_text = await self.browser.get_text(".status-bar")
            
            if not status_text:
                # Try alternative selectors
                status_text = await self.browser.evaluate("""
                    () => {
                        // Look for agent status in various places
                        const statusBar = document.querySelector('.status-bar, [class*="status"]');
                        if (statusBar) {
                            const text = statusBar.textContent;
                            if (text && text.includes('Agent')) {
                                return text;
                            }
                        }
                        
                        // Look for specific agent indicator
                        const agentElement = document.querySelector('[class*="agent"], [data-testid*="agent"]');
                        if (agentElement) {
                            return agentElement.textContent;
                        }
                        
                        return null;
                    }
                """)
            
            if status_text:
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
            else:
                return format_mcp_response(
                    False,
                    error="Agent status not found in UI"
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
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info("Showing all logs")
            
            # Look for Show All Logs button
            show_logs_selector = "button:has-text('Show All Logs'), button:has-text('Show Logs'), .show-all-logs"
            
            button_exists = await self.browser.is_visible(show_logs_selector)
            
            if button_exists:
                await self.browser.click(show_logs_selector)
                await asyncio.sleep(1)  # Wait for logs to load
                
                return format_mcp_response(
                    True,
                    message="Showing all logs"
                )
            else:
                # Try alternative - maybe logs are already visible
                logs_visible = await self.browser.is_visible(".log-panel, .logs-container, [class*='log']")
                
                if logs_visible:
                    return format_mcp_response(
                        True,
                        message="Logs are already visible"
                    )
                else:
                    return format_mcp_response(
                        False,
                        error="Show All Logs button not found"
                    )
                    
        except Exception as e:
            logger.error(f"Failed to show all logs: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def clear_logs(self) -> Dict[str, Any]:
        """
        Clear the log window
        
        Returns:
            MCP response with success status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info("Clearing logs")
            
            # Look for Clear Logs button
            clear_selector = "button:has-text('Clear'), button:has-text('Clear Logs'), .clear-logs"
            
            button_exists = await self.browser.is_visible(clear_selector)
            
            if button_exists:
                await self.browser.click(clear_selector)
                await asyncio.sleep(0.5)
                
                # Check for confirmation dialog
                dialog_visible = await self.browser.is_visible(DIALOG)
                if dialog_visible:
                    confirm_button = f"{DIALOG_FOOTER} button:has-text('Yes'), {DIALOG_FOOTER} button:has-text('Confirm')"
                    await self.browser.click(confirm_button)
                    await asyncio.sleep(0.5)
                
                return format_mcp_response(
                    True,
                    message="Logs cleared"
                )
            else:
                return format_mcp_response(
                    False,
                    error="Clear Logs button not found"
                )
                    
        except Exception as e:
            logger.error(f"Failed to clear logs: {e}")
            return format_mcp_response(False, error=str(e))