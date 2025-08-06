"""Register all additional tools with the MCP server"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dnne_ui_mcp_server import DNNE_UI_MCPServer

logger = logging.getLogger(__name__)

def register_all_additional_tools(server: "DNNE_UI_MCPServer"):
    """
    Register all additional tools from various tool modules
    
    Args:
        server: The DNNE_UI_MCPServer instance
    """
    
    # Import tool classes
    try:
        from .client_tools import ClientTools
        from .log_tools import LogTools
        from .ui_tools import UITools
        from .canvas_tools import CanvasTools
    except ImportError:
        from tools.client_tools import ClientTools
        from tools.log_tools import LogTools
        from tools.ui_tools import UITools
        from tools.canvas_tools import CanvasTools
    
    # Create tool instances with server reference for dynamic browser access
    client_tools = ClientTools(server, server.state)
    log_tools = LogTools(server, server.state)
    ui_tools = UITools(server, server.state)
    canvas_tools = CanvasTools(server, server.state)
    
    # Track the number of tools registered
    tool_count = 0
    
    def register_tool(func, name, description):
        """Helper to register a tool and increment counter"""
        nonlocal tool_count
        server.server.add_tool(func, name=name, description=description)
        tool_count += 1
    
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
    server.server.add_tool(
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
    server.server.add_tool(
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
        canvas_tools.get_link_visibility,
        name="get_link_visibility",
        description="Get current connection line visibility state"
    )
    
    register_tool(
        canvas_tools.set_link_visibility,
        name="set_link_visibility",
        description="Set connection line visibility (true to show, false to hide)"
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
    
    logger.info(f"Registered {tool_count} additional tools")