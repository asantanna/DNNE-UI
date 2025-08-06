"""Register all additional tools with the MCP server"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dnne_ui_mcp_server import DNNE_UI_MCPServer

logger = logging.getLogger(__name__)

def register_all_tools(server: "DNNE_UI_MCPServer"):
    """
    Register all MCP tools with the server
    
    Args:
        server: The DNNE_UI_MCPServer instance
    
    Returns:
        List of all registered tool names
    """
    
    # Import tool classes
    try:
        from .lifecycle_tools import LifecycleTools
        from .workflow_tools import WorkflowTools
        from .export_tools import ExportTools
        from .utility_tools import UtilityTools
        from .client_tools import ClientTools
        from .log_tools import LogTools
        from .ui_tools import UITools
        from .canvas_tools import CanvasTools
    except ImportError:
        from tools.lifecycle_tools import LifecycleTools
        from tools.workflow_tools import WorkflowTools
        from tools.export_tools import ExportTools
        from tools.utility_tools import UtilityTools
        from tools.client_tools import ClientTools
        from tools.log_tools import LogTools
        from tools.ui_tools import UITools
        from tools.canvas_tools import CanvasTools
    
    # Create tool instances with server reference for dynamic browser access
    lifecycle_tools = LifecycleTools(server)
    workflow_tools = WorkflowTools(server)
    export_tools = ExportTools(server)
    utility_tools = UtilityTools(server)
    client_tools = ClientTools(server)
    log_tools = LogTools(server)
    ui_tools = UITools(server)
    canvas_tools = CanvasTools(server)
    
    # Track the number of tools registered and their names
    tool_count = 0
    registered_tools = []
    
    def register_tool(func, name, description):
        """Helper to register a tool and increment counter"""
        nonlocal tool_count
        nonlocal registered_tools
        server.server.add_tool(func, name=name, description=description)
        tool_count += 1
        registered_tools.append(name)
    
    # Register browser lifecycle tools
    register_tool(
        lifecycle_tools.initialize_browser,
        name="initialize_browser",
        description="Initialize the browser and navigate to DNNE UI"
    )
    
    register_tool(
        lifecycle_tools.shut_down_browser_automation,
        name="shut_down_browser_automation",
        description="Shut down browser automation and free all resources"
    )
    
    register_tool(
        lifecycle_tools.is_browser_running,
        name="is_browser_running",
        description="Check if browser window is available"
    )
    
    register_tool(
        lifecycle_tools.restart_browser,
        name="restart_browser",
        description="Restart the browser for recovery"
    )
    
    # Register workflow management tools
    register_tool(
        workflow_tools.load_workflow,
        name="load_workflow",
        description="Load a workflow from the workflows sidebar"
    )
    
    register_tool(
        workflow_tools.get_current_workflow_name,
        name="get_current_workflow_name",
        description="Get the name of the currently loaded workflow"
    )
    
    register_tool(
        workflow_tools.save_workflow,
        name="save_workflow",
        description="Save the current workflow (optionally with a new name)"
    )
    
    register_tool(
        workflow_tools.new_blank_workflow,
        name="new_blank_workflow",
        description="Create a new blank workflow"
    )
    
    register_tool(
        workflow_tools.clear_workflow,
        name="clear_workflow",
        description="Clear the current workflow"
    )
    
    register_tool(
        workflow_tools.get_workflow_list,
        name="get_workflow_list",
        description="Get list of available workflows"
    )
    
    register_tool(
        workflow_tools.export_workflow,
        name="export_workflow",
        description="Export the current workflow"
    )
    
    # Register export and screenshot tools
    register_tool(
        export_tools.take_screenshot,
        name="take_screenshot",
        description="Take a screenshot of the DNNE UI"
    )
    
    # Register utility tools
    register_tool(
        utility_tools.util_is_ui_healthy,
        name="is_ui_healthy",
        description="Check if the DNNE UI is healthy and responsive"
    )
    
    register_tool(
        utility_tools.util_is_agent_server_running,
        name="util_is_agent_server_running",
        description="Utility: Check agent server health directly (bypasses UI)"
    )
    
    register_tool(
        utility_tools.util_get_dnne_server_status,
        name="util_get_dnne_server_status",
        description="Utility: Get DNNE server status directly (bypasses UI)"
    )
    
    register_tool(
        utility_tools.util_find_elements_by_text,
        name="util_find_elements_by_text",
        description="Utility: Find DOM elements by text content for debugging"
    )
    
    # Register client management tools
    register_tool(
        client_tools.get_connected_clients,
        name="get_connected_clients",
        description="Get list of all connected clients"
    )
    
    register_tool(
        client_tools.select_client,
        name="select_client",
        description="Select a client from taskbar or log window (e.g., 'Local', 'Tardigrade')"
    )
    
    register_tool(
        client_tools.get_agent_status,
        name="get_agent_status",
        description="Get the agent connection status"
    )
    
    register_tool(
        client_tools.show_all_logs,
        name="show_all_logs",
        description="Show logs from all clients"
    )
    
    register_tool(
        client_tools.clear_logs,
        name="clear_logs",
        description="Clear the log window"
    )
    
    # Register log analysis tools
    register_tool(
        log_tools.get_client_logs,
        name="get_client_logs",
        description="Get logs for a specific client or current selection"
    )
    
    register_tool(
        log_tools.get_training_metrics,
        name="get_training_metrics",
        description="Extract training metrics from logs"
    )
    
    register_tool(
        log_tools.get_export_errors,
        name="get_export_errors",
        description="Find export-related errors in logs"
    )
    
    register_tool(
        log_tools.get_recent_errors,
        name="get_recent_errors",
        description="Get the most recent error messages"
    )
    
    register_tool(
        log_tools.wait_for_log_pattern,
        name="wait_for_log_pattern",
        description="Wait for a specific pattern to appear in logs"
    )
    
    # Register UI navigation tools
    register_tool(
        ui_tools.open_sidebar_tab,
        name="open_sidebar_tab",
        description="Open a specific sidebar tab (workflows or nodes)"
    )
    
    register_tool(
        ui_tools.dismiss_dialog,
        name="dismiss_dialog",
        description="Dismiss any open dialog or error message"
    )
    
    register_tool(
        ui_tools.get_error_message,
        name="get_error_message",
        description="Get the current error dialog message if any"
    )
    
    register_tool(
        ui_tools.wait_for_ui_ready,
        name="wait_for_ui_ready",
        description="Wait for the UI to be fully loaded"
    )
    
    register_tool(
        ui_tools.click_menu_header,
        name="click_menu_header",
        description="Click a menu header to open/close menu (e.g., 'Workflow', 'Edit')"
    )
    
    register_tool(
        ui_tools.click_menu_item,
        name="click_menu_item",
        description="Click a menu item by path (e.g., 'Workflow/Save As', 'Edit/Undo')"
    )
    
    register_tool(
        ui_tools.click_droplist,
        name="click_droplist",
        description="Click a dropdown to open it (e.g., 'taskbar/client', 'log_window/filter')"
    )
    
    register_tool(
        ui_tools.click_droplist_item,
        name="click_droplist_item",
        description="Click a dropdown list item (e.g., path='taskbar/client', item='Local')"
    )
    
    register_tool(
        ui_tools.run_javascript,
        name="run_javascript",
        description="Execute JavaScript code in the current browser context"
    )
    
    # Register canvas operation tools
    register_tool(
        canvas_tools.zoom_to_fit,
        name="zoom_to_fit",
        description="Fit the workflow to the viewport"
    )
    
    register_tool(
        canvas_tools.get_node_count,
        name="get_node_count",
        description="Get the number of nodes in the current workflow"
    )
    
    register_tool(
        canvas_tools.zoom_in,
        name="zoom_in",
        description="Zoom in on the canvas"
    )
    
    register_tool(
        canvas_tools.zoom_out,
        name="zoom_out",
        description="Zoom out on the canvas"
    )
    
    register_tool(
        canvas_tools.get_canvas_state,
        name="get_canvas_state",
        description="Get comprehensive canvas state information"
    )
    
    logger.info(f"Registered {tool_count} tools")
    return registered_tools