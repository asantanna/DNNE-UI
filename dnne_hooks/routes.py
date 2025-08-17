"""
DNNE Routes

This module contains all DNNE-specific routes that extend ComfyUI functionality.
These routes are added to the server during initialization.
"""

import logging
import os
import json
import asyncio
from aiohttp import web
from typing import Dict, Any

logger = logging.getLogger(__name__)


def dnne_add_routes(server_instance, routes):
    """
    Add DNNE-specific routes to the server.
    
    Args:
        server_instance: The PromptServer instance
        routes: The aiohttp routes collection to add routes to
    """
    
    @routes.get("/api/agent/clients")
    async def get_agent_clients(request):
        """Return list of connected agent clients."""
        clients = [
            {"id": "local", "type": "local", "display": "Local"}
        ]
        
        # Add connected remote clients
        for client_id, info in server_instance.agent_clients.items():
            clients.append({
                "id": client_id,
                "type": "remote",
                "display": info.get("hostname", "Unknown"),
                "hostname": info.get("hostname"),
                "platform": info.get("platform"),
                "connected_at": info.get("connected_at")
            })
        
        return web.json_response({
            "clients": clients,
            "connection_status": server_instance.agent_connection_status
        })
    
    # REMOVED: /dnne/env_config endpoint - replaced by WebSocket-based widget callbacks
    # The IsaacGymEnvsNode now uses the generic DNNE_COMBO widget with callbacks
    
    # Add more DNNE-specific routes here as they are identified
    logger.info("DNNE routes added to server")