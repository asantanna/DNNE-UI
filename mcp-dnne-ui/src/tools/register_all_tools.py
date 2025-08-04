"""Register all additional tools with the MCP server"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dnne_ui_mcp_server import DNNEUIMCPServer

logger = logging.getLogger(__name__)

def register_all_additional_tools(server: "DNNEUIMCPServer"):
    """
    Register all additional tools from various tool modules
    
    Args:
        server: The DNNEUIMCPServer instance
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
    
    # Register client management tools
    server.server.add_tool(
        client_tools.get_connected_clients,
        name="get_connected_clients",
        description="Get list of all connected clients"
    )
    
    server.server.add_tool(
        client_tools.select_client,
        name="select_client",
        description="Select a specific client from the dropdown"
    )
    
    server.server.add_tool(
        client_tools.get_agent_status,
        name="get_agent_status",
        description="Get the agent connection status"
    )
    
    server.server.add_tool(
        client_tools.show_all_logs,
        name="show_all_logs",
        description="Show logs from all clients"
    )
    
    server.server.add_tool(
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
    
    server.server.add_tool(
        log_tools.get_training_metrics,
        name="get_training_metrics",
        description="Extract training metrics from logs"
    )
    
    server.server.add_tool(
        log_tools.get_export_errors,
        name="get_export_errors",
        description="Find export-related errors in logs"
    )
    
    server.server.add_tool(
        log_tools.get_recent_errors,
        name="get_recent_errors",
        description="Get the most recent error messages"
    )
    
    server.server.add_tool(
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
    
    server.server.add_tool(
        ui_tools.open_menu,
        name="open_menu",
        description="Open a menu item by path (e.g., 'Workflow/Save As')"
    )
    
    server.server.add_tool(
        ui_tools.dismiss_dialog,
        name="dismiss_dialog",
        description="Dismiss any open dialog or error message"
    )
    
    server.server.add_tool(
        ui_tools.get_error_message,
        name="get_error_message",
        description="Get the current error dialog message if any"
    )
    
    server.server.add_tool(
        ui_tools.wait_for_ui_ready,
        name="wait_for_ui_ready",
        description="Wait for the UI to be fully loaded"
    )
    
    # Register canvas operation tools
    server.server.add_tool(
        canvas_tools.zoom_to_fit,
        name="zoom_to_fit",
        description="Fit the workflow to the viewport"
    )
    
    server.server.add_tool(
        canvas_tools.toggle_link_visibility,
        name="toggle_link_visibility",
        description="Toggle connection line visibility"
    )
    
    server.server.add_tool(
        canvas_tools.get_node_count,
        name="get_node_count",
        description="Get the number of nodes in the current workflow"
    )
    
    server.server.add_tool(
        canvas_tools.zoom_in,
        name="zoom_in",
        description="Zoom in on the canvas"
    )
    
    server.server.add_tool(
        canvas_tools.zoom_out,
        name="zoom_out",
        description="Zoom out on the canvas"
    )
    
    server.server.add_tool(
        canvas_tools.get_canvas_state,
        name="get_canvas_state",
        description="Get comprehensive canvas state information"
    )
    
    logger.info(f"Registered {21} additional tools")