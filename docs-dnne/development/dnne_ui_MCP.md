# DNNE UI MCP (Model Context Protocol) Specification

## Overview

This document specifies a DNNE-specific MCP server that provides high-level automation for the DNNE UI. Instead of using low-level Puppeteer commands, this MCP will expose task-oriented functions that encapsulate common DNNE operations.

## Motivation

Currently, AI assistants interact with DNNE UI through generic Puppeteer commands, requiring knowledge of specific selectors, handling timing issues, and managing complex multi-step operations. A DNNE-specific MCP would:

1. **Simplify interactions** - One function call instead of multiple Puppeteer commands
2. **Improve reliability** - Built-in waits, retries, and error handling
3. **Provide better abstractions** - Task-focused rather than DOM-focused
4. **Enable state tracking** - Monitor workflow status, client connections, etc.

## Implementation Technology

Based on our proof of concept testing, we will use:
- **Python** with **Playwright** for browser automation
- MCP server framework for protocol implementation
- Async/await for non-blocking operations

### Why Playwright over Puppeteer?
- Native Python support (matches DNNE's ecosystem)
- Auto-waiting for elements (improves reliability)
- Better error messages and debugging
- More robust selector strategies

## Proposed MCP Functions

### Workflow Management

#### `load_workflow(name: str) -> dict`
Loads a workflow from the workflow list.
- Opens workflows sidebar if needed
- Finds and clicks the workflow by name
- Waits for workflow to load
- Returns: `{"success": bool, "message": str, "workflow_name": str}`

#### `save_workflow(name: Optional[str] = None) -> dict`
Saves the current workflow.
- If name is None, saves current workflow
- If name is provided, performs "Save As"
- Returns: `{"success": bool, "message": str, "saved_as": str}`

#### `new_blank_workflow() -> dict`
Creates a new empty workflow.
- Clears the canvas
- Returns: `{"success": bool, "message": str}`

#### `clear_workflow() -> dict`
Clears the current workflow.
- Returns: `{"success": bool, "message": str}`

#### `get_current_workflow_name() -> dict`
Gets the name of the currently loaded workflow.
- Returns: `{"workflow_name": str}`

### Export Operations

#### `export_workflow(run_after: bool = False) -> dict`
Exports the current workflow.
- Clicks export button
- Optionally sets "run after export" checkbox
- Waits for export completion
- Returns: `{"success": bool, "message": str, "export_path": str, "errors": List[str]}`

#### `set_export_target(target: str) -> dict`
Sets the export destination.
- Options: "Local", "Remote", etc.
- Returns: `{"success": bool, "target": str}`

#### `get_export_status() -> dict`
Checks if an export is in progress or completed.
- Returns: `{"status": str, "progress": float, "message": str}`

### Client/Agent Management

#### `get_connected_clients() -> dict`
Lists all connected clients.
- Clicks client dropdown
- Extracts list of clients
- Returns: `{"clients": List[str], "count": int}`

#### `select_client(name: str) -> dict`
Selects a specific client from the dropdown.
- Returns: `{"success": bool, "selected": str}`

#### `get_client_logs(client_name: Optional[str] = None) -> dict`
Gets logs for a specific client or current selection.
- Opens logs panel
- Selects client if specified
- Reads log content
- Returns: `{"client": str, "logs": str, "error_count": int, "warning_count": int}`

#### `show_all_logs() -> dict`
Shows logs from all clients.
- Clicks "Show All Logs" button
- Returns: `{"success": bool, "logs": str}`

#### `clear_logs() -> dict`
Clears the log window.
- Returns: `{"success": bool}`

#### `get_agent_status() -> dict`
Gets the agent connection status.
- Parses status bar for "Agent: ⚪ Connected"
- Returns: `{"connected": bool, "status": str}`

### Log Analysis

#### `get_training_metrics() -> dict`
Extracts training metrics from logs.
- Parses for epoch, loss, accuracy
- Returns: `{"epoch": int, "loss": float, "accuracy": float, "metrics": dict}`

#### `get_export_errors() -> dict`
Finds export-related errors in logs.
- Searches for widget mismatch, missing nodes, etc.
- Returns: `{"errors": List[dict], "count": int}`

#### `get_recent_errors(count: int = 10) -> dict`
Gets the most recent error messages.
- Returns: `{"errors": List[str], "timestamps": List[str]}`

#### `wait_for_log_pattern(pattern: str, timeout: int = 30) -> dict`
Waits for a specific pattern to appear in logs.
- Returns: `{"found": bool, "match": str, "timeout": bool}`

### UI Navigation

#### `open_sidebar_tab(tab: Literal["workflows", "nodes"]) -> dict`
Opens a specific sidebar tab.
- Clicks the appropriate tab button
- Waits for sidebar to open
- Returns: `{"success": bool, "tab": str}`

#### `open_menu(path: str) -> dict`
Opens a menu item by path.
- Example: "Workflow/Save As"
- Handles multi-level menus
- Returns: `{"success": bool, "path": str}`

#### `dismiss_dialog() -> dict`
Dismisses any open dialog or error message.
- Returns: `{"success": bool, "dialog_type": str}`

#### `get_error_message() -> dict`
Gets the current error dialog message if any.
- Returns: `{"has_error": bool, "title": str, "message": str}`

### Canvas Operations

#### `zoom_to_fit() -> dict`
Fits the workflow to the viewport.
- Returns: `{"success": bool}`

#### `toggle_link_visibility() -> dict`
Toggles connection line visibility.
- Returns: `{"success": bool, "visible": bool}`

#### `get_node_count() -> dict`
Gets the number of nodes in the current workflow.
- Returns: `{"count": int}`

#### `take_screenshot(name: str, full_page: bool = False) -> dict`
Takes a screenshot of the UI.
- Returns: `{"success": bool, "path": str, "dimensions": dict}`

### Utility Functions

#### `get_workflow_list() -> dict`
Lists all available workflows.
- Opens workflows sidebar
- Extracts workflow names
- Returns: `{"workflows": List[str], "count": int}`

#### `check_ui_health() -> dict`
Performs a health check on the UI.
- Verifies key elements are visible
- Checks agent connection
- Returns: `{"healthy": bool, "issues": List[str]}`

#### `wait_for_ui_ready(timeout: int = 10) -> dict`
Waits for the UI to be fully loaded.
- Returns: `{"ready": bool, "load_time": float}`

## Implementation Details

### Error Handling
Each function should:
- Catch and handle Playwright exceptions
- Provide meaningful error messages
- Include recovery attempts where appropriate
- Return structured error information

### Timing and Waits
- Use Playwright's auto-waiting features
- Add explicit waits only when necessary
- Configurable timeouts for long operations
- Progress indicators for multi-step operations

### State Management
The MCP server should maintain:
- Current workflow name
- Selected client
- Export settings
- UI state (sidebar open/closed, etc.)

## Example Usage

```python
# Instead of this (current Puppeteer approach):
await mcp__puppeteer__puppeteer_click({
    selector: '.workflows-tab-button'
})
await mcp__puppeteer__puppeteer_click({
    selector: 'li[aria-label="MNIST_Test.json"] .p-tree-node-content'
})

# We would have this:
await mcp__dnne_ui__load_workflow("MNIST_Test.json")
```

```python
# Complex operation simplified:
result = await mcp__dnne_ui__export_workflow(run_after=True)
if not result["success"]:
    errors = await mcp__dnne_ui__get_export_errors()
    print(f"Export failed: {errors}")
```

## Configuration

The MCP server would be configured in Claude's settings:

```json
{
  "servers": {
    "dnne-ui": {
      "command": "python",
      "args": ["/path/to/dnne_ui_mcp_server.py"],
      "type": "stdio",
      "env": {
        "DNNE_URL": "http://172.22.160.1:8188"
      }
    }
  }
}
```

## Next Steps

1. Create the MCP server skeleton with Playwright integration
2. Implement core functions (load_workflow, export_workflow, etc.)
3. Add error handling and retry logic
4. Test with common DNNE workflows
5. Package for easy installation and configuration

## Notes

### Features to Exclude (ComfyUI remnants)
- Queue operations (DNNE doesn't use a queue)
- Model library functions (to be removed from UI)
- Template browsing (non-functional in DNNE)
- API export (non-functional in DNNE)

### UI Cleanup Tasks Identified
- Remove "Browse Templates" from Workflow menu
- Remove "Export" and "Export (API)" from Workflow menu (redundant with Export button)
- Remove Model Library and Queue buttons from sidebar
- Change window title from "ComfyUI" to "DNNE"