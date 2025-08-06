"""Canvas operation tools for DNNE UI MCP Server"""

import asyncio
import logging
from typing import Dict, Any
try:
    from ..utils.helpers import format_mcp_response
    from ..utils.js_defs import *
    from ..utils.timing_constants import ANIMATION_DELAY
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.helpers import format_mcp_response
    from utils.js_defs import *
    from utils.timing_constants import ANIMATION_DELAY

logger = logging.getLogger(__name__)

class CanvasTools:
    """Tools for canvas operations in DNNE UI"""
    
    def __init__(self, server, state: Dict[str, Any]):
        """
        Initialize canvas tools
        
        Args:
            server: DNNE_UI_MCPServer instance for dynamic browser access
            state: Shared state dictionary
        """
        self.server = server
        self.state = state
    
    @property
    def browser(self):
        """Get browser controller dynamically from server"""
        return self.server.browser_controller
    
    async def zoom_to_fit(self) -> Dict[str, Any]:
        """
        Fit the workflow to the viewport
        
        Returns:
            MCP response with success status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info("Fitting workflow to viewport")
            
            # Click the Fit View button
            success = await self.browser.click(FIT_VIEW)
            
            if success:
                await asyncio.sleep(ANIMATION_DELAY)  # Wait for animation
                
                # Get current zoom level if possible
                zoom_level = await self.browser.evaluate("""
                    () => {
                        if (window.app && window.app.canvas && window.app.canvas.ds) {
                            return window.app.canvas.ds.scale;
                        }
                        return null;
                    }
                """)
                
                return format_mcp_response(
                    True,
                    data={"zoom_level": zoom_level},
                    message="Workflow fitted to viewport"
                )
            else:
                return format_mcp_response(
                    False,
                    error="Failed to click Fit View button"
                )
                
        except Exception as e:
            logger.error(f"Failed to zoom to fit: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def toggle_link_visibility(self) -> Dict[str, Any]:
        """
        Toggle connection line visibility
        
        Returns:
            MCP response with visibility status
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info("Toggling link visibility")
            
            # Get current visibility state before toggle
            visible_before = await self.browser.evaluate("""
                () => {
                    // Check if links are visible
                    const links = document.querySelectorAll('.link, .connection, [class*="link"]');
                    if (links.length > 0) {
                        const firstLink = links[0];
                        const style = window.getComputedStyle(firstLink);
                        return style.display !== 'none' && style.visibility !== 'hidden';
                    }
                    return true; // Default to visible if no links found
                }
            """)
            
            # Click the toggle button
            success = await self.browser.click(TOGGLE_LINKS)
            
            if success:
                await asyncio.sleep(ANIMATION_DELAY)
                
                # Get new visibility state
                visible_after = await self.browser.evaluate("""
                    () => {
                        const links = document.querySelectorAll('.link, .connection, [class*="link"]');
                        if (links.length > 0) {
                            const firstLink = links[0];
                            const style = window.getComputedStyle(firstLink);
                            return style.display !== 'none' && style.visibility !== 'hidden';
                        }
                        return true; // Default to visible if no links found
                    }
                """)
                
                self.state["links_visible"] = visible_after
                
                return format_mcp_response(
                    True,
                    data={"visible": visible_after},
                    message=f"Links are now {'visible' if visible_after else 'hidden'}"
                )
            else:
                return format_mcp_response(
                    False,
                    error="Failed to toggle link visibility"
                )
                
        except Exception as e:
            logger.error(f"Failed to toggle link visibility: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def get_node_count(self) -> Dict[str, Any]:
        """
        Get the number of nodes in the current workflow
        
        Returns:
            MCP response with node count
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info("Counting workflow nodes")
            
            # Try to get node count from the app object
            node_count = await self.browser.evaluate("""
                () => {
                    // Try to access the graph directly
                    if (window.app && window.app.canvas && window.app.canvas.graph) {
                        return window.app.canvas.graph.nodes.length;
                    }
                    
                    // Fallback: count DOM elements
                    const nodes = document.querySelectorAll('.node, [class*="node"]:not([class*="node-"])');
                    return nodes.length;
                }
            """)
            
            if node_count is not None:
                return format_mcp_response(
                    True,
                    data={"count": node_count},
                    message=f"Workflow has {node_count} nodes"
                )
            else:
                return format_mcp_response(
                    True,
                    data={"count": 0},
                    message="No nodes found or unable to count"
                )
                
        except Exception as e:
            logger.error(f"Failed to get node count: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def zoom_in(self) -> Dict[str, Any]:
        """
        Zoom in on the canvas
        
        Returns:
            MCP response with new zoom level
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info("Zooming in")
            
            # Get current zoom level
            zoom_before = await self.browser.evaluate("""
                () => {
                    if (window.app && window.app.canvas && window.app.canvas.ds) {
                        return window.app.canvas.ds.scale;
                    }
                    return 1.0;
                }
            """)
            
            # Click zoom in button
            success = await self.browser.click(ZOOM_IN)
            
            if success:
                await asyncio.sleep(ANIMATION_DELAY)
                
                # Get new zoom level
                zoom_after = await self.browser.evaluate("""
                    () => {
                        if (window.app && window.app.canvas && window.app.canvas.ds) {
                            return window.app.canvas.ds.scale;
                        }
                        return null;
                    }
                """)
                
                return format_mcp_response(
                    True,
                    data={
                        "zoom_before": zoom_before,
                        "zoom_after": zoom_after
                    },
                    message=f"Zoomed in from {zoom_before:.2f} to {zoom_after:.2f}" if zoom_after else "Zoomed in"
                )
            else:
                return format_mcp_response(
                    False,
                    error="Failed to zoom in"
                )
                
        except Exception as e:
            logger.error(f"Failed to zoom in: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def zoom_out(self) -> Dict[str, Any]:
        """
        Zoom out on the canvas
        
        Returns:
            MCP response with new zoom level
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info("Zooming out")
            
            # Get current zoom level
            zoom_before = await self.browser.evaluate("""
                () => {
                    if (window.app && window.app.canvas && window.app.canvas.ds) {
                        return window.app.canvas.ds.scale;
                    }
                    return 1.0;
                }
            """)
            
            # Click zoom out button
            success = await self.browser.click(ZOOM_OUT)
            
            if success:
                await asyncio.sleep(ANIMATION_DELAY)
                
                # Get new zoom level
                zoom_after = await self.browser.evaluate("""
                    () => {
                        if (window.app && window.app.canvas && window.app.canvas.ds) {
                            return window.app.canvas.ds.scale;
                        }
                        return null;
                    }
                """)
                
                return format_mcp_response(
                    True,
                    data={
                        "zoom_before": zoom_before,
                        "zoom_after": zoom_after
                    },
                    message=f"Zoomed out from {zoom_before:.2f} to {zoom_after:.2f}" if zoom_after else "Zoomed out"
                )
            else:
                return format_mcp_response(
                    False,
                    error="Failed to zoom out"
                )
                
        except Exception as e:
            logger.error(f"Failed to zoom out: {e}")
            return format_mcp_response(False, error=str(e))
    
    async def get_canvas_state(self) -> Dict[str, Any]:
        """
        Get comprehensive canvas state information
        
        Returns:
            MCP response with canvas state
        """
        try:
            if not self.browser:
                return format_mcp_response(False, error="Browser not initialized")
            
            logger.info("Getting canvas state")
            
            # Get comprehensive canvas information
            canvas_state = await self.browser.evaluate("""
                () => {
                    const state = {
                        has_canvas: false,
                        zoom: 1.0,
                        offset: {x: 0, y: 0},
                        node_count: 0,
                        link_count: 0,
                        selected_nodes: 0,
                        canvas_size: {width: 0, height: 0}
                    };
                    
                    if (window.app && window.app.canvas) {
                        state.has_canvas = true;
                        
                        if (window.app.canvas.ds) {
                            state.zoom = window.app.canvas.ds.scale || 1.0;
                            state.offset = window.app.canvas.ds.offset || {x: 0, y: 0};
                        }
                        
                        if (window.app.canvas.graph) {
                            state.node_count = window.app.canvas.graph.nodes?.length || 0;
                            state.link_count = window.app.canvas.graph.links?.length || 0;
                        }
                        
                        if (window.app.canvas.selected_nodes) {
                            state.selected_nodes = window.app.canvas.selected_nodes.length;
                        }
                        
                        state.canvas_size = {
                            width: window.app.canvas.canvas?.width || 0,
                            height: window.app.canvas.canvas?.height || 0
                        };
                    }
                    
                    return state;
                }
            """)
            
            return format_mcp_response(
                True,
                data=canvas_state,
                message="Retrieved canvas state"
            )
            
        except Exception as e:
            logger.error(f"Failed to get canvas state: {e}")
            return format_mcp_response(False, error=str(e))