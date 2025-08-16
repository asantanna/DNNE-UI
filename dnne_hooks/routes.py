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
    
    @routes.get("/dnne/env_config/{task_name}")
    async def get_env_config(request):
        """Get environment-specific configuration and metadata for connected nodes"""
        task_name = request.match_info.get("task_name", None)
        requesting_node_type = request.rel_url.query.get("node_type", None)
        trigger_widget = request.rel_url.query.get("trigger_widget", None)
        widget_values_str = request.rel_url.query.get("widget_values", None)
        
        logging.info(f" get_env_config called with task_name: {task_name}, requesting_node: {requesting_node_type}, trigger: {trigger_widget}")
        
        if not task_name or task_name == "none":
            logging.warning(f" Invalid task name: {task_name}")
            return web.json_response({"error": "Invalid task name"}, status=400)
        
        try:
            # Import the config loader and IsaacGymEnvsNode
            from custom_nodes.utils.isaac_gym_config_loader import IsaacGymEnvConfigLoader
            from custom_nodes.isaac_gym_envs_visnode import IsaacGymEnvsNode
            
            # Get singleton instance
            loader = IsaacGymEnvConfigLoader()
            
            # Load and get env config
            config = loader.get_task_config(task_name)
            
            if config is None:
                logging.warning(f" Config not found for task: {task_name}")
                return web.json_response({"error": f"Config not found for task: {task_name}"}, status=404)
            
            # Extract and send the configuration
            config_dict = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
            
            # Build response with config and metadata
            response = {
                "task_name": task_name,
                "config": config_dict,
                "requesting_node": requesting_node_type,
                "is_dnne_environment": loader.is_dnne_environment(task_name)
            }
            
            # Add DNNE-specific metadata if this is a DNNE environment
            if response["is_dnne_environment"]:
                response["subtasks"] = loader.get_task_subtasks(task_name)
                
                # Try to get dt value
                try:
                    response["dt"] = loader.get_task_dt(task_name)
                except NotImplementedError:
                    # dt not found, that's ok for some tasks
                    pass
            
            # If this is triggered by a widget change, add widget updates and schema display
            if trigger_widget:
                # Parse widget values if provided
                widget_values = {}
                if widget_values_str:
                    try:
                        widget_values = json.loads(widget_values_str)
                    except json.JSONDecodeError:
                        logging.warning(f"Failed to parse widget_values: {widget_values_str}")
                
                # Get widget updates for dynamic widgets
                if trigger_widget == 'task':
                    # Task changed - update dynamic widgets
                    response["widget_updates"] = IsaacGymEnvsNode.update_widgets_for_task(task_name)
                
                # Always update schema display based on current selections
                selections = {}
                for key, value in widget_values.items():
                    # Extract selections from dynamic widgets
                    if key.startswith('dynamic_') and value != 'none':
                        # Need to map widget to its label - get from widget updates
                        widget_updates = response.get("widget_updates", {})
                        if key in widget_updates and 'label' in widget_updates[key]:
                            selections[widget_updates[key]['label']] = value
                
                # Format schema display
                response["schema_display"] = IsaacGymEnvsNode.format_schema_display(task_name, selections)
            
            logging.info(f" Sending config for task {task_name}: {list(config_dict.keys())}")
            return web.json_response(response)
            
        except ImportError as e:
            logging.error(f" Failed to import IsaacGymEnvConfigLoader: {e}")
            return web.json_response({"error": "Isaac Gym configuration system not available"}, status=503)
        except FileNotFoundError as e:
            logging.error(f" Config file not found: {e}")
            return web.json_response({"error": f"Config file not found for task: {task_name}"}, status=404)
        except Exception as e:
            logging.error(f" Error loading env config for {task_name}: {e}")
            import traceback
            traceback.print_exc()
            return web.json_response({"error": f"Failed to load config: {str(e)}"}, status=500)
    
    # Add more DNNE-specific routes here as they are identified
    logger.info("DNNE routes added to server")