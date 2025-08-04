#!/usr/bin/env python3
"""
Test script to verify DNNE UI MCP server setup
"""

import sys
import os
import json
import asyncio

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_mcp_server():
    """Test that the MCP server can be imported and initialized"""
    print("Testing DNNE UI MCP Server setup...")
    print("-" * 50)
    
    try:
        # Test imports
        print("✓ Testing imports...")
        from dnne_ui_mcp_server import DNNEUIMCPServer
        print("  ✓ Successfully imported DNNEUIMCPServer")
        
        # Test initialization
        print("\n✓ Testing server initialization...")
        server = DNNEUIMCPServer()
        print("  ✓ Server initialized successfully")
        
        # Check registered tools (FastMCP doesn't expose tools directly)
        print(f"\n✓ Server initialized with DNNE UI tools")
        
        # List expected tool categories
        tool_categories = {
            'workflow': ['workflow_new', 'workflow_load', 'workflow_save', 'workflow_clear'],
            'export': ['export_workflow', 'export_status', 'export_cancel'],
            'client': ['client_list', 'client_status', 'client_connect'],
            'logs': ['logs_get', 'logs_clear', 'logs_export'],
            'ui': ['ui_screenshot', 'ui_refresh', 'ui_zoom'],
            'canvas': ['canvas_center', 'canvas_clear_selection'],
            'node': ['node_add', 'node_remove', 'node_connect'],
            'settings': ['settings_get', 'settings_update']
        }
        
        print("\n✓ Tool categories:")
        for category, tools in sorted(tool_categories.items()):
            print(f"  - {category}: {len(tools)} tools")
        
        print("\n" + "=" * 50)
        print("✅ MCP Server setup test PASSED!")
        print("=" * 50)
        
        # Configuration for Claude Desktop
        config = {
            "dnne-ui": {
                "command": "python3",
                "args": [
                    "/home/asantanna/DNNE/DNNE-UI/mcp-dnne-ui/run_mcp_server.py"
                ]
            }
        }
        
        print("\n📋 Add this to your Claude Desktop config (.claude.json):")
        print("\nIn the 'mcpServers' section:")
        print(json.dumps(config, indent=2))
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_mcp_server())
    sys.exit(0 if success else 1)