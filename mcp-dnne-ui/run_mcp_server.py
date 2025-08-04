#!/usr/bin/env python3
"""
DNNE UI MCP Server Runner
Entry point for running the DNNE UI MCP server with Claude Desktop
"""

import sys
import os
import asyncio

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dnne_ui_mcp_server import DNNEUIMCPServer

async def main():
    """Main entry point for the MCP server"""
    server = DNNEUIMCPServer()
    
    # Run the server
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())