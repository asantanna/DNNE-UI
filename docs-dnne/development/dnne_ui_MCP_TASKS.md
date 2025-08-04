# DNNE UI MCP Implementation Tasks

This is a persistent TODO list for tracking the implementation of the DNNE UI MCP server.  
See `dnne_ui_MCP_implementation_plan.md` for detailed specifications.

## Quick Stats
- **Total Tasks**: 75
- **Completed**: 0
- **In Progress**: 0
- **Blocked**: 0

## Phase 1: Project Setup and Infrastructure

### 1.1 Project Structure
- [ ] Create `mcp-dnne-ui/` directory structure
- [ ] Create `src/` subdirectory with module structure
- [ ] Create `tests/` directory
- [ ] Create `docs/` directory
- [ ] Create `pyproject.toml` with project metadata
- [ ] Create `.env.example` with DNNE_URL template
- [ ] Create initial `README.md` with setup instructions
- [ ] Initialize git repository
- [ ] Add `.gitignore` for Python projects

### 1.2 Dependencies and Environment
- [ ] Install `mcp[cli]>=1.4.0`
- [ ] Install `playwright>=1.48.0`
- [ ] Run `playwright install chromium`
- [ ] Install `python-dotenv>=1.0.0`
- [ ] Install `typing-extensions>=4.0.0`
- [ ] Create virtual environment
- [ ] Document dependency installation process

### 1.3 Base MCP Server
- [ ] Create `dnne_ui_mcp_server.py` with FastMCP
- [ ] Implement server initialization
- [ ] Add environment variable loading
- [ ] Create state management dictionary
- [ ] Implement server lifecycle methods
- [ ] Add logging configuration
- [ ] Create basic error handling

### 1.4 Browser Controller
- [ ] Create `browser_controller.py` class
- [ ] Implement browser launch with `--start-maximized`
- [ ] Add page navigation to DNNE URL
- [ ] Implement wait for page ready
- [ ] Add screenshot capability
- [ ] Create element finder helpers
- [ ] Add retry logic for element finding
- [ ] Implement browser cleanup

## Phase 2: Core Workflow Functions

### 2.1 Workflow Management
- [ ] Implement `load_workflow(name: str)`
- [ ] Implement `save_workflow(name: Optional[str])`
- [ ] Implement `new_blank_workflow()`
- [ ] Implement `clear_workflow()`
- [ ] Implement `get_current_workflow_name()`
- [ ] Add tests for workflow management
- [ ] Document workflow functions

### 2.2 Export Operations
- [ ] Implement `export_workflow(run_after: bool)`
- [ ] Implement `set_export_target(target: str)`
- [ ] Implement `get_export_status()`
- [ ] Add export error detection
- [ ] Add tests for export operations
- [ ] Document export functions

### 2.3 Initial Testing
- [ ] Create test script for Phase 2 functions
- [ ] Test with MNIST_Test.json
- [ ] Test with Cartpole_PPO.json
- [ ] Document any issues found
- [ ] Create bug fixes if needed

## Phase 3: Client and Log Management

### 3.1 Client Management
- [ ] Implement `get_connected_clients()`
- [ ] Implement `select_client(name: str)`
- [ ] Implement `get_agent_status()`
- [ ] Add client dropdown interaction
- [ ] Add tests for client management

### 3.2 Log Operations
- [ ] Implement `get_client_logs(client_name: Optional[str])`
- [ ] Implement `show_all_logs()`
- [ ] Implement `clear_logs()`
- [ ] Add log panel interaction
- [ ] Add tests for log operations

### 3.3 Log Analysis
- [ ] Implement `get_training_metrics()`
- [ ] Implement `get_export_errors()`
- [ ] Implement `get_recent_errors(count: int)`
- [ ] Implement `wait_for_log_pattern(pattern: str, timeout: int)`
- [ ] Add regex pattern matching
- [ ] Add tests for log analysis

## Phase 4: UI Navigation and Utilities

### 4.1 UI Navigation
- [ ] Implement `open_sidebar_tab(tab: str)`
- [ ] Implement `open_menu(path: str)`
- [ ] Implement `dismiss_dialog()`
- [ ] Implement `get_error_message()`
- [ ] Add menu path parsing
- [ ] Add tests for UI navigation

### 4.2 Canvas Operations
- [ ] Implement `zoom_to_fit()`
- [ ] Implement `toggle_link_visibility()`
- [ ] Implement `get_node_count()`
- [ ] Implement `take_screenshot(name: str)`
- [ ] Add canvas state detection
- [ ] Add tests for canvas operations

### 4.3 Utility Functions
- [ ] Implement `get_workflow_list()`
- [ ] Implement `check_ui_health()`
- [ ] Implement `wait_for_ui_ready(timeout: int)`
- [ ] Add health check criteria
- [ ] Add tests for utilities

## Phase 5: Error Handling and Robustness

### 5.1 Retry Logic
- [ ] Implement exponential backoff
- [ ] Add configurable retry counts
- [ ] Create recovery strategies
- [ ] Add timeout handling
- [ ] Test retry mechanisms

### 5.2 Enhanced Error Messages
- [ ] Add detailed error context
- [ ] Implement screenshot on failure
- [ ] Add troubleshooting suggestions
- [ ] Create error categorization
- [ ] Test error scenarios

### 5.3 State Recovery
- [ ] Implement browser restart
- [ ] Add state persistence
- [ ] Handle unexpected dialogs
- [ ] Add connection recovery
- [ ] Test recovery mechanisms

## Phase 6: Testing and Documentation

### 6.1 Testing Suite
- [ ] Create unit tests for all functions
- [ ] Create integration tests
- [ ] Add performance tests
- [ ] Test multi-client scenarios
- [ ] Generate test coverage report

### 6.2 Documentation
- [ ] Write installation guide
- [ ] Create API reference
- [ ] Write troubleshooting guide
- [ ] Create example scripts
- [ ] Add inline code documentation

### 6.3 Claude Integration
- [ ] Create MCP configuration JSON
- [ ] Test with Claude Desktop
- [ ] Document Claude-specific usage
- [ ] Create configuration templates
- [ ] Verify all functions work from Claude

## Phase 7: Deployment and Packaging

### 7.1 Packaging
- [ ] Create setup.py/pyproject.toml
- [ ] Add installation scripts
- [ ] Create Windows installer
- [ ] Create Linux installer
- [ ] Test installation process

### 7.2 MCP Registration
- [ ] Configure for Claude Desktop
- [ ] Create JSON template
- [ ] Test registration process
- [ ] Document configuration steps
- [ ] Create troubleshooting guide

### 7.3 Final Testing
- [ ] End-to-end testing
- [ ] User acceptance testing
- [ ] Performance optimization
- [ ] Final bug fixes
- [ ] Release preparation

## Bugs and Issues

### Known Issues
- [ ] Status bar clipping in Puppeteer (resolved with Playwright)
- [ ] IsaacGymEnvs widget mismatch on export
- [ ] ComfyUI remnants in menu (Browse Templates, Export, Export API)
- [ ] Window title says "ComfyUI" instead of "DNNE"

### Blockers
- None currently

## Notes

### Important Decisions
- **Technology**: Python with Playwright (not Node.js with Puppeteer)
- **Architecture**: FastMCP framework for MCP server
- **Browser**: Chromium with `--start-maximized` flag
- **URL**: Default to `http://172.22.160.1:8188` (WSL2 to Windows)

### Lessons Learned
- Playwright auto-waiting is more reliable than Puppeteer
- Status bar visibility requires `--start-maximized`
- Browser must be launched with `no_viewport=True`
- Element selectors should prioritize data-testid and aria-label

### Resources
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Playwright Documentation](https://playwright.dev/python/)
- [DNNE UI MCP Specification](dnne_ui_MCP.md)
- [Implementation Plan](dnne_ui_MCP_implementation_plan.md)

## Progress Log

### 2025-08-04
- Created implementation plan document
- Created this TASKS.md for tracking
- Successfully tested Playwright with DNNE UI
- Validated browser automation approach

---

*Last Updated: 2025-08-04*  
*Next Review: When Phase 1 begins*