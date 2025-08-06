#!/usr/bin/env python3
"""
DNNE UI MCP Server - Browser automation for DNNE UI testing

Naming Convention:
- Functions WITHOUT 'util_' prefix: Query through UI only (test what users see)
- Functions WITH 'util_' prefix: Query servers directly (get ground truth)

Use UI functions for testing the user interface.
Use util functions for debugging and verification.
"""

import asyncio
import aiohttp
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
    from .utils.js_defs import *
    from .utils.error_handler import ErrorDiagnostics, with_error_handling
except ImportError:
    # For direct script execution
    from browser_controller import BrowserController
    from utils.helpers import setup_logging, get_env_var, format_mcp_response
    from utils.js_defs import *
    from utils.error_handler import ErrorDiagnostics, with_error_handling

# Load environment variables
load_dotenv()

# Setup logging
setup_logging(get_env_var("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

class DNNE_UI_MCPServer:
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
        
        # Error diagnostics
        self.error_diagnostics = ErrorDiagnostics()
        
        # Register all tools
        self._register_tools()
        
    def _register_tools(self):
        """Register all MCP tools with the server"""
        
        # Register all tools from tool modules
        try:
            from .tools.register_all_tools import register_all_tools
        except ImportError:
            from tools.register_all_tools import register_all_tools
        
        self.registered_tools = register_all_tools(self)
    
    async def _util_find_elements_by_text(self, text: str, limit: int = 10) -> Dict[str, Any]:
        """
        Utility: Find DOM elements containing specific text (bypasses normal selectors)
        Useful for debugging when selectors aren't working
        
        Args:
            text: Text to search for in elements
            limit: Maximum number of results to return (default 10)
        
        Returns:
            Dict with found elements and their selectors
        """
        if not self.browser_controller or not self.browser_controller.is_browser_window_available():
            return {"found": False, "error": "Browser not available", "count": 0, "elements": []}
        
        try:
            # Escape the search text for JavaScript
            escaped_text = text.replace('\\', '\\\\').replace('"', '\\"')
            
            results = await self.browser_controller.page.evaluate(f"""
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
            
            return results
            
        except Exception as e:
            return {"found": False, "error": str(e), "count": 0, "elements": []}
    
    async def _util_is_agent_server_running(self) -> Dict[str, Any]:
        """
        Utility: Check agent server health directly (bypasses UI)
        Queries http://172.22.160.1:8769/health
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://172.22.160.1:8769/health', timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            "running": True,
                            "healthy": data.get("status") == "healthy",
                            "uptime": data.get("uptime"),
                            "connections": data.get("connections"),
                            "data": data
                        }
                    else:
                        return {"running": False, "error": f"HTTP {resp.status}"}
        except asyncio.TimeoutError:
            return {"running": False, "error": "Connection timeout"}
        except Exception as e:
            return {"running": False, "error": str(e)}
    
    async def _util_get_dnne_server_status(self) -> Dict[str, Any]:
        """
        Utility: Get DNNE server status directly (bypasses UI)
        Queries http://172.22.160.1:8188/api/agent/clients
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://172.22.160.1:8188/api/agent/clients', timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            "running": True,
                            "clients": data.get("clients", []),
                            "connection_status": data.get("connection_status"),
                            "data": data
                        }
                    else:
                        return {"running": False, "error": f"HTTP {resp.status}"}
        except asyncio.TimeoutError:
            return {"running": False, "error": "Connection timeout"}
        except Exception as e:
            return {"running": False, "error": str(e)}
    
    def run(self):
        """Run the MCP server"""
        logger.info("Starting DNNE UI MCP Server")
        logger.info(f"DNNE URL: {self.dnne_url}")
        
        # FastMCP handles stdio transport
        self.server.run("stdio")

def main():
    """Main entry point"""
    server = DNNE_UI_MCPServer()
    server.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)