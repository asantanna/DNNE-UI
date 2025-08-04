# DNNE UI MCP Implementation Plan

## Overview
This document outlines the detailed implementation plan for creating a Model Context Protocol (MCP) server that provides high-level automation for the DNNE UI using Python and Playwright.

## Timeline
**Total Duration**: 4 weeks  
**Start Date**: TBD  
**Target Completion**: TBD

## Phase 1: Project Setup and Infrastructure (Week 1)

### 1.1 Create MCP Server Project Structure
**Duration**: 1 day

- Create directory structure:
  ```
  mcp-dnne-ui/
  ├── src/
  │   ├── __init__.py
  │   ├── dnne_ui_mcp_server.py
  │   ├── browser_controller.py
  │   ├── tools/
  │   │   ├── __init__.py
  │   │   ├── workflow_tools.py
  │   │   ├── export_tools.py
  │   │   ├── client_tools.py
  │   │   ├── log_tools.py
  │   │   ├── ui_tools.py
  │   │   └── canvas_tools.py
  │   └── utils/
  │       ├── __init__.py
  │       ├── selectors.py
  │       └── helpers.py
  ├── tests/
  ├── docs/
  ├── pyproject.toml
  ├── .env.example
  └── README.md
  ```

- Install dependencies:
  ```toml
  [dependencies]
  mcp = ">=1.4.0"
  playwright = ">=1.48.0"
  python-dotenv = ">=1.0.0"
  asyncio = "*"
  typing-extensions = ">=4.0.0"
  ```

### 1.2 Implement Base MCP Server Class
**Duration**: 2 days

```python
from mcp.server.fastmcp import FastMCP
from playwright.sync_api import sync_playwright
import os
from dotenv import load_dotenv

class DNNEUIServer:
    def __init__(self):
        load_dotenv()
        self.mcp = FastMCP("DNNE UI Automation")
        self.browser = None
        self.page = None
        self.dnne_url = os.getenv("DNNE_URL", "http://172.22.160.1:8188")
        self.state = {
            "current_workflow": None,
            "selected_client": None,
            "sidebar_open": False,
            "export_target": "Local"
        }
```

### 1.3 Browser Automation Layer
**Duration**: 2 days

- Browser initialization with proper viewport
- Navigation and page load verification
- Screenshot management
- Element waiting and retry logic
- Selector constants and helpers

## Phase 2: Core Workflow Functions (Week 1-2)

### 2.1 Workflow Management Tools
**Duration**: 3 days

Functions to implement:
- `load_workflow(name: str)` - Load workflow from sidebar
- `save_workflow(name: Optional[str])` - Save current or save as
- `new_blank_workflow()` - Create new empty workflow
- `clear_workflow()` - Clear current workflow
- `get_current_workflow_name()` - Get active workflow name

### 2.2 Export Operations
**Duration**: 2 days

Functions to implement:
- `export_workflow(run_after: bool)` - Main export function
- `set_export_target(target: str)` - Configure export destination
- `get_export_status()` - Check export progress

### 2.3 Testing Phase 2
**Duration**: 1 day

- Unit tests for each workflow function
- Integration test with MNIST_Test.json
- Error scenario testing
- Performance benchmarking

## Phase 3: Client and Log Management (Week 2)

### 3.1 Client Management Tools
**Duration**: 2 days

Functions to implement:
- `get_connected_clients()` - List all clients
- `select_client(name: str)` - Select specific client
- `get_agent_status()` - Parse agent connection status

### 3.2 Log Operations
**Duration**: 2 days

Functions to implement:
- `get_client_logs(client_name: Optional[str])` - Retrieve logs
- `show_all_logs()` - Display all client logs
- `clear_logs()` - Clear log window

### 3.3 Log Analysis Tools
**Duration**: 2 days

Functions to implement:
- `get_training_metrics()` - Extract training data
- `get_export_errors()` - Find export issues
- `get_recent_errors(count: int)` - Recent error messages
- `wait_for_log_pattern(pattern: str, timeout: int)` - Wait for specific output

## Phase 4: UI Navigation and Utilities (Week 2-3)

### 4.1 UI Navigation Tools
**Duration**: 2 days

Functions to implement:
- `open_sidebar_tab(tab: str)` - Navigate sidebar
- `open_menu(path: str)` - Navigate menu items
- `dismiss_dialog()` - Close dialogs
- `get_error_message()` - Extract error text

### 4.2 Canvas Operations
**Duration**: 2 days

Functions to implement:
- `zoom_to_fit()` - Fit workflow to view
- `toggle_link_visibility()` - Show/hide connections
- `get_node_count()` - Count workflow nodes
- `take_screenshot(name: str)` - Capture UI state

### 4.3 Utility Functions
**Duration**: 1 day

Functions to implement:
- `get_workflow_list()` - List available workflows
- `check_ui_health()` - Verify UI state
- `wait_for_ui_ready(timeout: int)` - Wait for load

## Phase 5: Error Handling and Robustness (Week 3)

### 5.1 Add Retry Logic
**Duration**: 2 days

- Exponential backoff implementation
- Configurable retry counts
- Recovery strategies for common failures
- Timeout handling

### 5.2 Enhanced Error Messages
**Duration**: 1 day

- Detailed error context
- Screenshot on failure
- Troubleshooting suggestions
- Error categorization

### 5.3 State Recovery
**Duration**: 2 days

- Browser restart capability
- State persistence
- Unexpected dialog handling
- Connection recovery

## Phase 6: Testing and Documentation (Week 3-4)

### 6.1 Comprehensive Testing
**Duration**: 3 days

- Unit test suite
- Integration tests
- Error scenario coverage
- Performance testing
- Multi-client scenarios

### 6.2 Documentation
**Duration**: 2 days

- Installation guide
- API reference
- Troubleshooting guide
- Example scripts
- Video tutorials (optional)

### 6.3 Claude Integration
**Duration**: 2 days

- MCP configuration for Claude Desktop
- Testing all functions from Claude
- Claude-specific documentation
- Configuration templates

## Phase 7: Deployment and Packaging (Week 4)

### 7.1 Package for Distribution
**Duration**: 2 days

- Python package structure
- setup.py/pyproject.toml configuration
- Installation scripts
- Platform-specific installers

### 7.2 MCP Registration
**Duration**: 1 day

- Claude Desktop configuration
- JSON configuration template
- Environment variable setup
- Installation verification

### 7.3 Final Testing
**Duration**: 2 days

- End-to-end testing
- User acceptance testing
- Performance optimization
- Bug fixes

## Technical Specifications

### Browser Configuration
```python
browser_args = [
    '--start-maximized',
    '--disable-blink-features=AutomationControlled',
    '--disable-dev-shm-usage'
]
```

### Selector Strategy
Priority order:
1. data-testid attributes
2. aria-label attributes
3. Class selectors with specific names
4. Text content matching
5. CSS path as last resort

### State Management
```python
state = {
    "current_workflow": str,
    "selected_client": str,
    "sidebar_open": bool,
    "export_target": str,
    "last_error": str,
    "browser_ready": bool
}
```

### Error Categories
- `BrowserError`: Browser launch/navigation failures
- `ElementNotFoundError`: Missing UI elements
- `TimeoutError`: Operation timeouts
- `ExportError`: Export-specific failures
- `ConnectionError`: DNNE server connection issues

## Success Criteria

### Functional Requirements
- ✅ All 30+ MCP functions implemented and working
- ✅ Status bar always visible in screenshots
- ✅ Successful workflow export with error handling
- ✅ Client log extraction functional
- ✅ Multi-step operations reliable

### Performance Requirements
- Page load < 5 seconds
- Function response < 2 seconds
- Screenshot capture < 1 second
- Export completion detection < 30 seconds

### Quality Requirements
- 90%+ test coverage
- Comprehensive error handling
- Clear documentation
- Intuitive function names

## Risk Mitigation

### Technical Risks
| Risk | Mitigation |
|------|-----------|
| Browser compatibility | Test on Windows 10/11, multiple Chrome versions |
| Network latency | Implement retries, longer timeouts for remote servers |
| UI changes | Use flexible selectors, implement fallbacks |
| Race conditions | Add explicit waits, use Playwright's auto-waiting |
| Memory leaks | Proper browser cleanup, periodic restarts |

### Project Risks
| Risk | Mitigation |
|------|-----------|
| Scope creep | Strict adherence to specification |
| Timeline delays | Buffer time in each phase |
| Integration issues | Early Claude Desktop testing |
| Documentation debt | Document as we code |

## Dependencies

### External Dependencies
- DNNE UI server must be running
- Chrome/Chromium browser available
- Network access to DNNE server
- Python 3.8+ environment

### Python Package Dependencies
- mcp[cli] >= 1.4.0
- playwright >= 1.48.0
- python-dotenv >= 1.0.0
- asyncio (built-in)
- typing-extensions >= 4.0.0

## Monitoring and Metrics

### Development Metrics
- Functions implemented: 0/35
- Tests written: 0/50+
- Documentation pages: 0/10
- Bug count: 0

### Runtime Metrics
- Average function execution time
- Success/failure rates
- Most used functions
- Error frequency by type

## Conclusion

This implementation plan provides a structured approach to building the DNNE UI MCP server. The phased approach allows for iterative development with regular testing and validation. The focus on error handling and robustness ensures the MCP server will be reliable in production use.