#!/usr/bin/env python3
"""
DNNE UI MCP Server Runner
Entry point for running the DNNE UI MCP server with Claude Desktop
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dnne_ui_mcp_server import DNNE_UI_MCPServer

def main():
    """Main entry point for the MCP server"""
    server = DNNE_UI_MCPServer()
    
    # Run the server
    server.run()

if __name__ == "__main__":
    main()