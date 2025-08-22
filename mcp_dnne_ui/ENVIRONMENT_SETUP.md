# Environment Setup Guide

## Conda Environment Usage

This project requires specific conda environments for different purposes:

### MCP_PY310 Environment
- **Purpose**: MCP server and Claude Code integration
- **Python Version**: 3.10.18
- **Key Dependencies**: mcp, playwright, pydantic, aiohttp
- **Usage**: All MCP server operations, testing, and Claude Code sessions

```bash
# Activate MCP environment
source /home/asantanna/miniconda/bin/activate MCP_PY310

# Start Claude Code in this environment
claude-code

====================================================

# Run MCP server MANUALLY (if outside of Claude Code)
python src/dnne_ui_mcp_server.py

# Run tests
python tests/test_all_mcp_tools.py
```

### DNNE_PY38 Environment  
- **Purpose**: Main DNNE project and Isaac Gym integration
- **Python Version**: 3.8.x
- **Key Dependencies**: torch, isaacgym, isaacgymenvs
- **Usage**: DNNE server, neural network training, robotics simulation


## Important Notes

1. **Always start Claude Code in MCP_PY310** - This ensures all MCP dependencies are available
2. **Keep environments separate** - Isaac Gym requires Python 3.8, MCP requires Python 3.10+
3. **Use the correct environment for each task** - MCP operations in MCP_PY310, DNNE operations in DNNE_PY38
