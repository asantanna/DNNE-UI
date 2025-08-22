#!/bin/bash

# DNNE UI MCP Server Installation Script for Claude Desktop
# This script installs the DNNE UI MCP server for use with Claude Desktop

echo "═══════════════════════════════════════════════════════════════"
echo "  DNNE UI MCP Server Installation for Claude Desktop"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if claude command exists
if ! command -v claude &> /dev/null; then
    echo "❌ Error: 'claude' command not found."
    echo "   Please ensure Claude Desktop is installed and the CLI is in your PATH."
    echo "   See: https://docs.anthropic.com/en/docs/claude-code"
    exit 1
fi

# Server name (can be customized if needed)
SERVER_NAME="dnne-ui"

# Check if user wants a custom name (useful for multiple installations)
if [ "$1" == "--name" ] && [ -n "$2" ]; then
    SERVER_NAME="$2"
    echo "📝 Using custom server name: $SERVER_NAME"
fi

# Build the command to run the MCP server
# We'll use python directly with the full path to the script
MCP_COMMAND="python"
MCP_SCRIPT="$SCRIPT_DIR/src/dnne_ui_mcp_server.py"

echo "📦 Installing MCP server..."
echo "   Name: $SERVER_NAME"
echo "   Directory: $SCRIPT_DIR"
echo "   Command: python $SCRIPT_DIR/src/dnne_ui_mcp_server.py"
echo ""

# Install the MCP server
# Pass the python command and the script path as an argument
claude mcp add "$SERVER_NAME" "$MCP_COMMAND" "$MCP_SCRIPT"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully installed DNNE UI MCP server!"
    echo ""
    echo "To use in Claude Desktop:"
    echo "  1. Restart Claude Desktop (or reload the conversation)"
    echo "  2. The MCP tools will be available automatically"
    echo ""
    echo "To verify installation:"
    echo "  claude mcp list"
    echo ""
    echo "To uninstall later:"
    echo "  claude mcp remove $SERVER_NAME"
else
    echo ""
    echo "❌ Installation failed. Please check the error message above."
    exit 1
fi