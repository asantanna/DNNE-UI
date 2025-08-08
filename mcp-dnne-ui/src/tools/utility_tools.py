"""Utility tools for DNNE UI MCP Server"""

import asyncio
import logging
import sys
import aiohttp
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import format_mcp_response
from utils.js_defs import SIDEBAR_BUTTON, COMFYUI_MENU, DIALOG

if TYPE_CHECKING:
    from dnne_ui_mcp_server import DNNE_UI_MCPServer

logger = logging.getLogger(__name__)


class UtilityTools:
    """Utility tools for DNNE UI operations and health checks"""
    
    def __init__(self, server: "DNNE_UI_MCPServer"):
        """
        Initialize utility tools
        
        Args:
            server: DNNE_UI_MCPServer instance for browser access
        """
        self.server = server
    
    @property
    def browser(self):
        """Get browser controller from server"""
        return self.server.browser_controller
    
    async def util_is_ui_healthy(self) -> Dict[str, Any]:
        """Check if the DNNE UI is healthy and responsive"""
        try:
            # First check if browser is even initialized
            if not self.browser:
                return format_mcp_response(
                    False, 
                    error="Browser not initialized",
                    data={"browser_initialized": False}
                )
            
            # Check if browser process is active
            if not self.browser.is_playwright_browser_process_active():
                return format_mcp_response(
                    False, 
                    error="Browser process not active. Use initialize_browser.",
                    data={"browser_process": False}
                )
            
            # Check if browser window is available
            if not self.browser.is_browser_window_available():
                return format_mcp_response(
                    False,
                    error="Browser window closed. Use restart_browser.",
                    data={"browser_process": True, "window_available": False}
                )
            
            # Check if JavaScript is executable
            if not await self.browser.is_javascript_executable():
                return format_mcp_response(
                    False,
                    error="Browser window not responsive. Use restart_browser.",
                    data={"browser_process": True, "window_available": True, "js_executable": False}
                )
            
            issues = []
            
            # Check for key UI elements
            if not await self.browser.is_visible(SIDEBAR_BUTTON):
                issues.append("Sidebar buttons not visible")
            
            if not await self.browser.is_visible(COMFYUI_MENU):
                issues.append("Menu bar not visible")
            
            # Check for error dialogs
            if await self.browser.is_visible(DIALOG):
                issues.append("Error dialog is open")
            
            # Get agent status from browser
            agent_status = await self.browser.get_status_bar_info()
            if not agent_status["agent_connected"]:
                issues.append("Agent is disconnected")
            
            # Get comprehensive UI state
            ui_state = await self.browser.get_ui_state()
            
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
    
    async def util_is_agent_server_running(self) -> Dict[str, Any]:
        """
        Utility: Check agent server health directly (bypasses UI)
        Queries http://172.22.160.1:8769/health
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://172.22.160.1:8769/health', timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return format_mcp_response(
                            True,
                            data={
                                "running": True,
                                "healthy": data.get("status") == "healthy",
                                "uptime": data.get("uptime"),
                                "connections": data.get("connections"),
                                "raw_data": data
                            },
                            message="Agent server is running"
                        )
                    else:
                        return format_mcp_response(
                            False,
                            error=f"HTTP {resp.status}",
                            message="Agent server not reachable"
                        )
        except asyncio.TimeoutError:
            return format_mcp_response(
                False,
                error="Connection timeout",
                message="Agent server not reachable"
            )
        except Exception as e:
            return format_mcp_response(
                False,
                error=str(e),
                message="Agent server not reachable"
            )
    
    async def util_get_dnne_server_status(self) -> Dict[str, Any]:
        """
        Utility: Get DNNE server status directly (bypasses UI)
        Queries http://172.22.160.1:8188/api/agent/clients
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://172.22.160.1:8188/api/agent/clients', timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return format_mcp_response(
                            True,
                            data={
                                "running": True,
                                "clients": data.get("clients", []),
                                "connection_status": data.get("connection_status"),
                                "raw_data": data
                            },
                            message="DNNE server is running"
                        )
                    else:
                        return format_mcp_response(
                            False,
                            error=f"HTTP {resp.status}",
                            message="DNNE server not reachable"
                        )
        except asyncio.TimeoutError:
            return format_mcp_response(
                False,
                error="Connection timeout",
                message="DNNE server not reachable"
            )
        except Exception as e:
            return format_mcp_response(
                False,
                error=str(e),
                message="DNNE server not reachable"
            )
    
    async def util_restart_dnne(self, restart_agent_server: bool = False, extra_args: str = None) -> Dict[str, Any]:
        """
        Utility: Restart DNNE server (and optionally agent server)
        Uses /remote_command endpoint to trigger restart
        
        Args:
            restart_agent_server: If True, also restart the agent server
            extra_args: Additional command line arguments to pass (e.g., "--verbose DEBUG")
        
        Returns:
            MCP response with restart status
        """
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "command": "restart",
                    "args": {
                        "delay": 3,
                        "reason": "MCP requested restart",
                        "restart_agent_server": restart_agent_server
                    }
                }
                
                # Add extra arguments if provided
                if extra_args:
                    payload["args"]["extra_args"] = extra_args
                
                async with session.post(
                    'http://172.22.160.1:8188/remote_command',
                    json=payload,
                    timeout=10
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return format_mcp_response(
                            True,
                            data=data,
                            message=f"Server restart initiated{' (including agent server)' if restart_agent_server else ''}"
                        )
                    else:
                        return format_mcp_response(
                            False,
                            error=f"HTTP {resp.status}",
                            message="Failed to restart server"
                        )
        except asyncio.TimeoutError:
            return format_mcp_response(
                False,
                error="Connection timeout",
                message="Failed to restart server"
            )
        except Exception as e:
            return format_mcp_response(
                False,
                error=str(e),
                message="Failed to restart server"
            )
    
    async def util_is_DNNE_running(self) -> Dict[str, Any]:
        """
        Utility: Check if DNNE server is running using health endpoint
        Queries http://172.22.160.1:8188/health
        
        Returns:
            MCP response with server health status
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://172.22.160.1:8188/health', timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return format_mcp_response(
                            True,
                            data={
                                "running": True,
                                "healthy": data.get("status") == "healthy",
                                "uptime": data.get("uptime"),
                                "version": data.get("version"),
                                "agent_connected": data.get("agent_connected"),
                                "agent_clients": data.get("agent_clients"),
                                "active_workflows": data.get("active_workflows"),
                                "raw_data": data
                            },
                            message="DNNE server is running"
                        )
                    else:
                        return format_mcp_response(
                            False,
                            error=f"HTTP {resp.status}",
                            message="DNNE server not reachable"
                        )
        except asyncio.TimeoutError:
            return format_mcp_response(
                False,
                error="Connection timeout",
                message="DNNE server not reachable"
            )
        except Exception as e:
            return format_mcp_response(
                False,
                error=str(e),
                message="DNNE server not reachable"
            )
    
    async def util_find_elements_by_text(self, text: str, limit: int = 10) -> Dict[str, Any]:
        """
        Utility: Find DOM elements containing specific text (bypasses normal selectors)
        Useful for debugging when selectors aren't working
        
        Args:
            text: Text to search for in elements
            limit: Maximum number of results to return (default 10)
        
        Returns:
            MCP response with found elements and their selectors
        """
        if not self.browser or not self.browser.is_browser_window_available():
            return format_mcp_response(
                False,
                error="Browser not available",
                data={"count": 0, "elements": []},
                message=f"Found 0 elements containing '{text}'"
            )
        
        try:
            # Escape the search text for JavaScript
            escaped_text = text.replace('\\', '\\\\').replace('"', '\\"')
            
            results = await self.browser.page.evaluate(f"""
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
            
            return format_mcp_response(
                results.get("found", False),
                data=results,
                message=f"Found {results.get('count', 0)} elements containing '{text}'"
            )
            
        except Exception as e:
            return format_mcp_response(
                False,
                error=str(e),
                data={"count": 0, "elements": []},
                message=f"Found 0 elements containing '{text}'"
            )