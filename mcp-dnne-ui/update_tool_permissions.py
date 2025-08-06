#!/usr/bin/env python3
"""
Script to extract all MCP tool names and update Claude settings with permissions

This script uses the unified tool registration system to automatically discover
all registered MCP tools and generate permissions for Claude settings.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent))

from src.dnne_ui_mcp_server import DNNE_UI_MCPServer

def update_claude_settings():
    """Extract tool names and update Claude settings file"""
    
    # Create a server instance which registers all tools
    server = DNNE_UI_MCPServer()
    
    # Get all registered tool names from the server
    tool_names = server.registered_tools if hasattr(server, 'registered_tools') else []
    
    print(f"Found {len(tool_names)} MCP tools:")
    for name in sorted(tool_names):
        print(f"  - {name}")
    
    # Settings file path
    settings_path = Path("/home/asantanna/DNNE/DNNE-UI/.claude/settings.local.json")
    
    # Read existing settings
    if settings_path.exists():
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        print(f"\nLoaded existing settings from {settings_path}")
    else:
        settings = {}
        print(f"\nCreating new settings file at {settings_path}")
    
    # Ensure permissions structure exists
    if "permissions" not in settings:
        settings["permissions"] = {"allow": [], "deny": []}
    if "allow" not in settings["permissions"]:
        settings["permissions"]["allow"] = []
    if "deny" not in settings["permissions"]:
        settings["permissions"]["deny"] = []
    
    # Get existing allowed tools (to avoid duplicates)
    existing_allow = set(settings["permissions"]["allow"])
    
    # Add all MCP tools to the allow list
    # Format: "mcp__dnne-ui__<tool_name>"
    for tool_name in tool_names:
        mcp_tool_name = f"mcp__dnne-ui__{tool_name}"
        existing_allow.add(mcp_tool_name)
    
    # Update the allow list
    settings["permissions"]["allow"] = sorted(list(existing_allow))
    
    # Ensure directory exists
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write updated settings
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2, sort_keys=True)
    
    print(f"\nUpdated {settings_path} with permissions for:")
    print(f"  - {len(tool_names)} MCP tools added to allow list")
    print("\nTotal allowed tools:", len(settings["permissions"]["allow"]))
    
    # Show a sample of the permissions
    print("\nSample permissions added:")
    for tool in list(tool_names)[:5]:
        print(f"  - mcp__dnne-ui__{tool}: allow")
    if len(tool_names) > 5:
        print(f"  ... and {len(tool_names) - 5} more MCP tools")

if __name__ == "__main__":
    update_claude_settings()