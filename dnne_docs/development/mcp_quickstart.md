# MCP Quickstart for Claude Code

**Purpose**: Get Claude Code instances working with DNNE UI's MCP server with minimum friction.  
**Installation**: See [mcp_dnne_ui/README.md](../../mcp_dnne_ui/README.md) for setup details.

## First Steps (Every Session)

```bash
# 1. Check if browser is running
is_browser_running()

# 2. Initialize if needed (only if not running)
initialize_browser()

# 3. Verify UI is healthy
is_ui_healthy()

# 4. If UI issues, restart DNNE server (runs on Windows, controlled from WSL)
util_restart_dnne()
```

## Core Workflows

### Export Workflow Pattern
```bash
# 1. Select target client (CRITICAL - must be done first!)
# Use "Local" for export-only, or remote client name for export+run
select_client("Tardigrade")

# 2. Export (and optionally run with no args)
export_workflow(run_after=True)  # True = deploy and run (no args), False = export only

# 3. Stop running workflow (remote clients only)
click_button("taskbar/stop")

# 4. Check status
get_status_bar_info()
```

### Load and Modify Workflow
```bash
# 1. Get available workflows
get_workflow_list()

# 2. Load specific workflow
load_workflow("MNIST_Test")

# 3. Make changes via UI...

# 4. Save workflow
save_workflow("MNIST_Test_Modified")
```

### Debug UI Issues
```bash
# 1. Take screenshot to see state
take_screenshot("debug")

# 2. Find elements by text
util_find_elements_by_text("Export")

# 3. Get canvas info
get_canvas_state()
```

## Critical Gotchas

### Browser Not Auto-Initialized
❌ **Wrong**: Calling UI functions without checking browser
✅ **Right**: Always `is_browser_running()` → `initialize_browser()` first

### Deploy and Run Requires Remote Client
❌ **Wrong**: `export_workflow(run_after=True)` without remote client
✅ **Right**: `select_client("Tardigrade")` → `export_workflow(run_after=True)`
✅ **Also Right**: `select_client("Local")` → `export_workflow(run_after=False)` # Export only

**Terminology**: "Export" = save locally, "Deploy" = send to remote client

### Dialog Blocks Everything
❌ **Wrong**: Trying to interact while dialog is open
✅ **Right**: `dismiss_dialog()` first, then continue

### Server Restart Method
❌ **Wrong**: `Bash("pkill -f dnne")` - doesn't work (servers run on Windows)
✅ **Right**: `util_restart_dnne()` - proper Windows restart from WSL

### Agent Server Restart Disconnects Clients
⚠️ **Warning**: Restarting agent server disconnects all clients
❌ **Wrong**: `util_restart_dnne(restart_agent_server=True)` then expect clients to reconnect
✅ **Right**: After agent server restart, manually restart agent clients:
```bash
# In WSL terminal:
python dnne_agent/dnne_agent_client.py
```

### Workflow List Sidebar Handling
ℹ️ **Note**: `get_workflow_list()` automatically handles sidebar state - opens it if needed, gets list, restores previous state

## Tool Categories

### Lifecycle (Browser Control)
- `initialize_browser()` - Start browser automation
- `shut_down_browser_automation()` - Clean shutdown
- `restart_browser()` - Recovery from issues
- `is_browser_running()` - Check status

### Workflow Operations
- `load_workflow(name)` - Load workflow by name
- `save_workflow(name)` - Save current (avoid overwriting standard workflows!)
- `export_workflow(run_after)` - Export to Python (run_after=True deploys and runs)
- `get_workflow_list()` - List available workflows
- `clear_workflow()` - Clear canvas

### UI Interaction
- `select_client(name)` - Choose export target
- `click_button(location)` - Click buttons
- `enter_input_text(path, text)` - Fill inputs
- `click_checkbox(path)` - Toggle checkboxes
- `dismiss_dialog()` - Close dialogs

### Canvas Control
- `zoom_to_fit()` - Fit to viewport
- `get_node_count()` - Count nodes
- `take_screenshot(name)` - Capture UI
- `get_canvas_state()` - Full canvas info

### Utility/Debug
- `util_is_DNNE_running()` - Check server health
- `util_restart_dnne()` - Restart DNNE server
- `util_find_elements_by_text(text)` - Find UI elements
- `get_status_bar_info()` - Connection status
- `get_viewer_client_log()` - View logs (may require log viewer to be visible)

## Supported UI Locations

Valid paths for UI interaction tools:

### Button Locations (`click_button`)
**Taskbar:**
- `taskbar/export` - Main export button
- `taskbar/export_dropdown` - Export dropdown arrow
- `taskbar/stop` - Stop workflow button
- `taskbar/show_logs` - Show logs button

**Canvas:**
- `canvas/zoom_in` - Zoom in button
- `canvas/zoom_out` - Zoom out button
- `canvas/fit_view` - Fit to viewport button
- `canvas/select_mode` - Select mode button

**Dialog:**
- `dialog/close` - Close dialog button
- `dialog/confirm` - Confirm/Yes button
- `dialog/cancel` - Cancel/No button

**Runner Args Dialog:**
- `runner_args_dlg/cancel` - Cancel button
- `runner_args_dlg/accept` - Accept button (text varies)

**Sidebar:**
- `sidebar/close_workflow` - Close workflow button

**Log Window:**
- `log_window/clear` - Clear logs button (if exists)

### Dropdown Locations (`click_droplist`, `click_droplist_item`)
**Taskbar:**
- `taskbar/client` - Client selector dropdown
- `taskbar/export` - Export options (after clicking dropdown arrow)

**Log Window:**
- `log_window/client` - Client selector in log viewer
- `log_window/type` - Log type selector (Run Logs, Telemetry Data, etc.)

### Checkbox Locations (`click_checkbox`, `get_checkbox_state`)
**Runner Args Dialog:**
- `runner_args_dlg/override` - Override checkbox

**Taskbar:**
- `taskbar/custom_args` - Custom args checkbox

**Log Window:**
- `log_window/auto_scroll` - Auto-scroll checkbox

### Input Locations (`enter_input_text`, `get_input_text`)
**Runner Args Dialog:**
- `runner_args_dlg/cmd_line` - Command line arguments input

> **Tip**: Use `util_find_elements_by_text()` to discover elements or `run_javascript()` for unsupported actions.

## Common Patterns

### Recovery Sequence
```bash
# When UI is unresponsive
dismiss_dialog()
restart_browser()
wait_for_ui_ready()

# When server appears dead
util_restart_dnne()  # Basic restart
util_restart_dnne(dnne_extra_args="--debug")  # Debug mode (check dnne_logs/DNNE.log)
util_restart_dnne(restart_agent_server=True)  # Restart both servers
util_restart_dnne(dnne_extra_args="--debug", restart_agent_server=True, agent_server_extra_args="--debug")  # Both in debug

# For telemetry test suites only
util_restart_dnne(restart_agent_server=True, agent_server_extra_args="--enable-test-port")
```

### Export with Custom Args
```bash
# 1. Select remote client
select_client("Tardigrade")

# 2. Open export dialog (click dropdown)
click_button("taskbar/export_dropdown")

# 3. Choose "Deploy and Run" 
click_droplist_item("taskbar/export", "Deploy and Run")

# 4. In runner args dialog:
click_checkbox("runner_args_dlg/override")  # Enables manual arg entry
enter_input_text("runner_args_dlg/cmd_line", "--enable-telemetry 10,11")
click_button("dialog/confirm")
```

### Check Training Progress
```bash
# 1. Select client in log viewer
select_client("Tardigrade", "log_window")

# 2. Get log content (defaults to "Run Logs")
get_viewer_client_log()

# 3. Switch log types as needed:
click_droplist_item("log_window/type", "Telemetry Violations")  # View violations
click_droplist_item("log_window/type", "Telemetry Data")  # View telemetry data
get_viewer_client_log()  # Works for all log types
```

## Quick Debugging

**UI not responding?**
```bash
take_screenshot("stuck")
dismiss_dialog()
```

**Export not working?**
```bash
get_connected_clients()  # Is client connected?
get_status_bar_info()    # Check agent status
```

**Can't find element?**
```bash
util_find_elements_by_text("button text")
run_javascript("document.querySelector('.my-class')")
```

**Server issues?**
```bash
util_is_DNNE_running()
util_is_agent_server_running()
util_restart_dnne()  # Usually sufficient
# util_restart_dnne(restart_agent_server=True)  # Only if agent server also has issues
```

## Tips

1. **Always initialize browser first** - No auto-init on tool calls
2. **Screenshots are your friend** - Visual debugging beats guessing
3. **Dismiss dialogs immediately** - They block all interaction
4. **Select client before deploy+run** - Can always export locally, but need remote for run
5. **Use util functions for server control** - Cleaner than bash commands
6. **Check health after restart** - Ensure services are ready
7. **Runner args dialog accepts Enter** - No need to click Accept

## Environment Note

MCP server runs in Windows but tools execute from WSL2:
- DNNE UI: `http://172.22.160.1:8188`
- Use `util_restart_dnne()` for server control
- Screenshots save to `mcp_dnne_ui/screenshots/`
