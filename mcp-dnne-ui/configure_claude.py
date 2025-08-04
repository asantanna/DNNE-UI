#!/usr/bin/env python3
"""
Configure Claude Desktop to use the DNNE UI MCP Server
"""

import json
import shutil
from pathlib import Path

def configure_claude_desktop():
    """Add DNNE UI MCP server to Claude Desktop configuration"""
    
    claude_config_path = Path.home() / ".claude.json"
    backup_path = Path.home() / ".claude.json.backup"
    
    # Create backup
    if claude_config_path.exists():
        shutil.copy2(claude_config_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
    
    # Load existing configuration
    if claude_config_path.exists():
        with open(claude_config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Ensure projects structure exists
    if "projects" not in config:
        config["projects"] = {}
    
    # Get the current project path
    project_path = "/home/asantanna/DNNE/DNNE-UI"
    
    # Ensure project exists in config
    if project_path not in config["projects"]:
        config["projects"][project_path] = {
            "allowedTools": [],
            "history": [],
            "mcpContextUris": [],
            "mcpServers": {},
            "enabledMcpjsonServers": [],
            "disabledMcpjsonServers": [],
            "hasTrustDialogAccepted": True,
            "projectOnboardingSeenCount": 0,
            "hasClaudeMdExternalIncludesApproved": False
        }
    
    # Add the DNNE UI MCP server configuration
    dnne_ui_server_config = {
        "command": "python3",
        "args": [
            "/home/asantanna/DNNE/DNNE-UI/mcp-dnne-ui/src/dnne_ui_mcp_server.py"
        ],
        "env": {
            "DNNE_URL": "http://172.22.160.1:8188",
            "BROWSER_HEADLESS": "false",
            "LOG_LEVEL": "INFO"
        }
    }
    
    # Update the project's MCP servers
    config["projects"][project_path]["mcpServers"]["dnne-ui"] = dnne_ui_server_config
    
    # Write back the configuration
    with open(claude_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Claude Desktop configuration updated!")
    print(f"📁 Project: {project_path}")
    print(f"🔧 MCP Server: dnne-ui")
    print(f"🌐 DNNE URL: http://172.22.160.1:8188")
    print()
    print("Next steps:")
    print("1. Restart Claude Desktop for changes to take effect")
    print("2. Ensure the DNNE UI server is running on http://172.22.160.1:8188")
    print("3. In Claude Desktop, you can now use MCP tools like 'initialize_browser', 'load_workflow', etc.")
    print()
    print("Available MCP tools:")
    tools = [
        "initialize_browser", "cleanup_browser", "restart_browser",
        "load_workflow", "export_workflow", "save_workflow",
        "new_blank_workflow", "clear_workflow", "get_workflow_list",
        "take_screenshot", "check_ui_health", "get_current_workflow_name"
    ]
    for tool in tools:
        print(f"  - {tool}")

if __name__ == "__main__":
    configure_claude_desktop()