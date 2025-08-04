# DNNE UI MCP Server

A Model Context Protocol (MCP) server that provides high-level automation for the DNNE UI using Python and Playwright.

## Overview

This MCP server replaces low-level Puppeteer commands with task-oriented functions, making it easier for AI assistants to interact with the DNNE UI. Instead of managing selectors and timing issues, you can use simple commands like `load_workflow("MNIST_Test.json")` or `export_workflow(run_after=True)`.

## Features

- **Workflow Management**: Load, save, create, and clear workflows
- **Export Operations**: Export workflows with configurable options
- **Client Management**: Monitor and select connected clients
- **Log Analysis**: Extract training metrics and error messages
- **UI Navigation**: Navigate menus and sidebars programmatically
- **Canvas Operations**: Control zoom, visibility, and capture screenshots

## Installation

### Prerequisites

- Python 3.8 or higher
- DNNE UI server running (typically on Windows)
- Chrome/Chromium browser

### Setup

1. Clone or navigate to the repository:
```bash
cd mcp-dnne-ui
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
playwright install chromium
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your DNNE server URL
```

## Usage

### Starting the MCP Server

```bash
python src/dnne_ui_mcp_server.py
```

### Configuring with Claude Desktop

Add to your Claude Desktop MCP configuration:

```json
{
  "servers": {
    "dnne-ui": {
      "command": "python",
      "args": ["/path/to/mcp-dnne-ui/src/dnne_ui_mcp_server.py"],
      "type": "stdio",
      "env": {
        "DNNE_URL": "http://172.22.160.1:8188"
      }
    }
  }
}
```

### Available Functions

#### Workflow Management
- `load_workflow(name)` - Load a workflow from the sidebar
- `save_workflow(name)` - Save the current workflow
- `new_blank_workflow()` - Create a new empty workflow
- `clear_workflow()` - Clear the current workflow
- `get_current_workflow_name()` - Get the active workflow name

#### Export Operations
- `export_workflow(run_after)` - Export the current workflow
- `set_export_target(target)` - Set export destination
- `get_export_status()` - Check export progress

#### Client Management
- `get_connected_clients()` - List all connected clients
- `select_client(name)` - Select a specific client
- `get_agent_status()` - Get agent connection status

#### Log Operations
- `get_client_logs(client_name)` - Get logs for a client
- `show_all_logs()` - Show logs from all clients
- `clear_logs()` - Clear the log window
- `get_training_metrics()` - Extract training metrics
- `get_export_errors()` - Find export-related errors

#### UI Navigation
- `open_sidebar_tab(tab)` - Open workflows or nodes sidebar
- `open_menu(path)` - Navigate menu items
- `dismiss_dialog()` - Close any open dialog
- `get_error_message()` - Get current error message

#### Canvas Operations
- `zoom_to_fit()` - Fit workflow to viewport
- `toggle_link_visibility()` - Show/hide connections
- `get_node_count()` - Count nodes in workflow
- `take_screenshot(name)` - Capture UI screenshot

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black src/
ruff check src/
```

## Troubleshooting

### Browser won't connect to DNNE server
- Ensure DNNE server is running on Windows
- Check firewall settings
- Verify the URL in `.env` file
- Test with: `curl http://172.22.160.1:8188`

### Status bar not visible in screenshots
- The browser must be launched with `--start-maximized`
- Don't use zoom as it breaks the maximized state

### Workflow won't load
- Check that the workflow exists in the sidebar
- Ensure no dialogs are blocking the UI
- Verify the workflow name matches exactly

## Documentation

- [MCP Specification](../docs-dnne/development/dnne_ui_MCP.md)
- [Implementation Plan](../docs-dnne/development/dnne_ui_MCP_implementation_plan.md)
- [Task Tracking](../docs-dnne/development/dnne_ui_MCP_TASKS.md)

## License

Part of the DNNE project.