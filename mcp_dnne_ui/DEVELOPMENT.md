# DNNE UI MCP Server - Development Documentation

## Architecture Overview

### Technology Stack
- **Language**: Python 3.10+ (MCP_PY310 conda environment)
- **Browser Automation**: Playwright
- **MCP Framework**: FastMCP
- **Async**: asyncio for non-blocking operations
- **Browser**: Chromium

### Design Decisions

#### Stateless Architecture
- MCP server is completely stateless
- Browser state queried directly when needed
- Avoids synchronization issues between disk and actual state

#### Naming Conventions
- Query functions use `is_` prefix (e.g., `is_ui_healthy()`)
- Utility functions use `util_` prefix for ground truth operations

#### Fail-Fast Design Principles
- **No fallback selectors** - Fail immediately with clear error messages
- **No default values** - Throw NotImplementedError for missing implementations
- **Clear error context** - Include selector paths and expected elements
- **Immediate feedback** - No silent failures or wrong behavior

```

### Important Notes
- Browser runs on Windows, accessed from WSL2
- Default URL: `http://172.22.160.1:8188`

## Error Handling Strategy

### Error Response Format
```python
{
    "success": False,
    "error": "Error message",
    "error_type": "BrowserError",
    "screenshot": "path/to/error_screenshot.png",
    "suggestion": "Try restarting the browser"
}
```

### Running Tests
```bash
# Activate MCP_PY310 environment first
source /home/asantanna/miniconda/bin/activate MCP_PY310

# Run comprehensive test suite
python tests/test_all_mcp_tools.py

# Test results are saved to:
# tests/test_results_comprehensive.json
```

## MCP Integration

### Claude Desktop Configuration
```json
{
  "servers": {
    "dnne-ui": {
      "command": "python",
      "args": ["/path/to/mcp_dnne_ui/run_mcp_server.py"],
      "type": "stdio"
    }
  }
}
```

## Contributing

### Code Style
```bash
# Format code
black src/

# Check linting
ruff check src/

# Type checking
mypy src/
```

## Resources

- [TASKS.md](TASKS.md) - Current task tracking
- [README.md](README.md) - User documentation
- [MCP Documentation](https://github.com/modelcontextprotocol/python-sdk)
- [Playwright Documentation](https://playwright.dev/python/)