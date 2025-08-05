# DNNE UI MCP Server - Development Documentation

## Architecture Overview

### Technology Stack
- **Language**: Python 3.10+ (MCP_PY310 conda environment)
- **Browser Automation**: Playwright (not Puppeteer)
- **MCP Framework**: FastMCP
- **Async**: asyncio for non-blocking operations
- **Browser**: Chromium with `--start-maximized`

### Design Decisions

#### Why Playwright over Puppeteer?
- Native Python support (matches DNNE ecosystem)
- Auto-waiting for elements (improves reliability)
- Better error messages and debugging
- More robust selector strategies
- Connect to existing browser via CDP

#### Stateless Architecture
- **No disk persistence** - MCP server is completely stateless
- Browser state queried directly when needed
- Avoids synchronization issues between disk and actual state
- Three levels of browser state checking:
  - `is_playwright_browser_process_active()` - Process exists
  - `is_browser_window_available()` - Window/page exists
  - `is_javascript_executable()` - Can execute JS

#### Naming Conventions
- Query functions use `is_` prefix (e.g., `is_ui_healthy()`)
- Utility functions use `util_` prefix for ground truth operations
- Tool names match function names for consistency

#### Fail-Fast Design Principles
- **No fallback selectors** - Fail immediately with clear error messages
- **No default values** - Throw NotImplementedError for missing implementations
- **Clear error context** - Include selector paths and expected elements
- **Immediate feedback** - No silent failures or wrong behavior

## Current Implementation Status

### Working Components (Production Ready) ✅
- Browser lifecycle management (init, cleanup, restart)
- Screenshot capture with configurable paths
- Health checks and status monitoring (fixed agent status selector)
- Basic workflow operations (new, clear, list)
- MCP server communication with Claude Desktop
- Menu navigation with submenu state checking
- Dialog dismissal with updated selectors
- Canvas operations (zoom, fit view, link visibility)
- Comprehensive test suite with bulletproof UI restoration

### Known Issues 🔧

#### 1. Client Dropdown Selector (HIGH PRIORITY)
**Problem**: Client dropdown selector not found in UI
**Impact**: Cannot select export targets
**Next Steps**: Identify correct selector from UI inspection

#### 2. Log Button Selectors (HIGH PRIORITY)
**Problem**: Clear Logs and Show All Logs buttons not found
**Impact**: Log management functionality incomplete
**Status**: Buttons may be disabled or have different selectors

#### 2. Export System Slot Corruption (MEDIUM PRIORITY)
**Problem**: Export may fail with slot corruption errors
**Impact**: Core DNNE functionality affected
**Status**: Identified in main DNNE codebase, not MCP-specific

## Project Structure

```
mcp-dnne-ui/
├── src/
│   ├── dnne_ui_mcp_server.py      # Main MCP server
│   ├── browser_controller.py       # Browser automation layer
│   ├── tools/
│   │   ├── workflow_tools.py      # Workflow management
│   │   ├── export_tools.py        # Export operations
│   │   ├── client_tools.py        # Client management
│   │   ├── log_tools.py           # Log operations
│   │   ├── ui_tools.py            # UI navigation
│   │   └── canvas_tools.py        # Canvas operations
│   └── utils/
│       ├── selectors.py           # UI selectors
│       ├── helpers.py             # Helper functions
│       └── state_manager.py       # State management (in-memory only)
├── tests/
├── screenshots/                    # All screenshots go here
├── SELECTORS.md                   # UI selector reference
├── DEVELOPMENT.md                  # This file
├── TASKS.md                       # Task tracking
└── README.md                      # User documentation
```

## Browser Configuration

### Launch Settings
```python
browser_args = [
    '--start-maximized',           # Required for status bar visibility
    '--disable-blink-features=AutomationControlled',
    '--disable-dev-shm-usage'
]
```

### Connection Method
```python
# Connect to existing browser via CDP
browser = playwright.chromium.connect_over_cdp('http://localhost:9222')
```

### Important Notes
- Browser runs on Windows, accessed from WSL2
- Default URL: `http://172.22.160.1:8188`
- Must maintain maximized state (no zoom operations)
- Status bar requires specific viewport handling

## Error Handling Strategy

### Error Categories
```python
class DNNEError(Exception): pass
class BrowserError(DNNEError): pass      # Browser launch/navigation
class ElementNotFoundError(DNNEError): pass  # Missing UI elements
class TimeoutError(DNNEError): pass      # Operation timeouts
class ExportError(DNNEError): pass       # Export failures
class ConnectionError(DNNEError): pass   # DNNE server connection
```

### Recovery Mechanisms
1. **Browser restart** with state preservation
2. **Automatic retry** with exponential backoff
3. **Screenshot on failure** for debugging
4. **Detailed error context** in responses

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

## Testing Framework

### Test Coverage (as of 2025-08-04)
```
Total Tools: 39
Tested: 39 (100%)
Working: 31 (79.5%)
Failed: 8 (20.5%)
```

### Test Categories
- **Unit Tests**: Individual tool functions
- **Integration Tests**: Multi-tool workflows
- **Error Tests**: Failure scenarios
- **Performance Tests**: Timing and responsiveness

### Running Tests
```bash
# Activate MCP_PY310 environment first
source /home/asantanna/miniconda/bin/activate MCP_PY310

# Run comprehensive test suite
python tests/test_all_mcp_tools.py

# Test results are saved to:
# tests/test_results_comprehensive.json
```

## Debugging Techniques

### 1. Browser State Inspection
```python
# Check all browser states
state = {
    "process_active": browser_controller.is_playwright_browser_process_active(),
    "window_available": browser_controller.is_browser_window_available(),
    "js_executable": browser_controller.is_javascript_executable()
}
```

### 2. Screenshot Debugging
```python
# Take screenshot when debugging
await browser_controller.screenshot("debug_state.png")
```

### 3. Element Finding
```python
# Use util function to find elements by text
result = await util_find_elements_by_text("Save As")
```

### 4. Selector Testing
```python
# Test selector directly
element = await page.query_selector('.workflows-tab-button')
visible = await element.is_visible() if element else False
```

## Performance Considerations

### Timeouts
- **Default timeout**: 3000ms (3 seconds)
- **UI is fast** - long timeouts indicate wrong selector
- **Menu animations**: 500ms delay needed
- **Sidebar animations**: 1000ms delay needed

### Optimization Tips
1. Check element visibility before interactions
2. Use specific selectors (data-testid preferred)
3. Batch operations when possible
4. Reuse browser connection

## MCP Integration

### Claude Desktop Configuration
```json
{
  "servers": {
    "dnne-ui": {
      "command": "python",
      "args": ["/path/to/mcp-dnne-ui/run_mcp_server.py"],
      "type": "stdio"
    }
  }
}
```

### Auto-Allow Tools
Update `.claude/settings.local.json`:
```json
{
  "tools": {
    "approvedTools": {
      "mcp__dnne-ui__initialize_browser": true,
      "mcp__dnne-ui__load_workflow": true,
      // ... add all tools
    }
  }
}
```

## Common Pitfalls and Solutions

### Menu Navigation
**Pitfall**: Clicking menu header when submenu is already open closes it
**Solution**: Always check submenu visibility first

### Workflow Loading
**Pitfall**: Sidebar not open when trying to load workflow
**Solution**: Check sidebar state and open if needed

### Screenshot Paths
**Pitfall**: Screenshots saved in wrong directory
**Solution**: Always use absolute MCP path: `mcp-dnne-ui/screenshots/`

### Browser State
**Pitfall**: Assuming browser is ready after launch
**Solution**: Use proper state checking methods

## Future Improvements

### High Priority
1. Complete log analysis tools (5 functions)
2. Fix remaining selector issues
3. Add comprehensive error recovery

### Medium Priority
1. Performance profiling
2. Multi-client testing scenarios
3. Advanced canvas operations

### Low Priority
1. Video capture capability
2. Network request monitoring
3. Custom wait conditions

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

### Pull Request Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Selectors documented
- [ ] Error handling added
- [ ] Screenshots for UI changes

## Resources

- [SELECTORS.md](SELECTORS.md) - Complete UI selector reference
- [TASKS.md](TASKS.md) - Current task tracking
- [README.md](README.md) - User documentation
- [MCP Documentation](https://github.com/modelcontextprotocol/python-sdk)
- [Playwright Documentation](https://playwright.dev/python/)