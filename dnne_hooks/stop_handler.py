"""
DNNE Stop Handler

This module handles the stop functionality for DNNE workflows running on agent clients.
It intercepts the interrupt_processing call from ComfyUI and sends stop signals to agents.
"""

import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)


def dnne_stop_handler(prompt_server=None):
    """
    Handle stop request for DNNE workflows.
    
    Called when the STOP button is pressed in the UI. This function
    sends stop signals to any running workflows on agent clients.
    
    Args:
        prompt_server: The PromptServer instance (if available)
    """
    logger.info("DNNE stop handler triggered")
    
    if prompt_server is None:
        logger.error("No prompt_server instance provided to stop handler - cannot stop workflows")
        return
    
    # Check if agent WebSocket connection exists
    if not hasattr(prompt_server, 'agent_ws') or not prompt_server.agent_ws:
        logger.error("No agent WebSocket connection available - cannot stop workflows")
        return
    
    # Get the active workflows from the server
    if not hasattr(prompt_server, 'active_workflows'):
        logger.error("No active_workflows tracking in prompt_server - cannot determine which workflows to stop")
        return
    
    # Find active workflows
    active_workflows = []
    for workflow_id, workflow_info in prompt_server.active_workflows.items():
        # Active workflows are those that have an open file handle
        if workflow_info.get('file_handle'):
            active_workflows.append({
                'workflow_id': workflow_id,
                'client_id': workflow_info.get('client_id', 'unknown'),
                'workflow_name': workflow_info.get('name', 'Unknown')
            })
    
    if not active_workflows:
        logger.info("No active workflows to stop")
        return
    
    # Stop each active workflow
    for workflow in active_workflows:
        logger.info(f"Stopping workflow: {workflow['workflow_name']} (ID: {workflow['workflow_id']}) on client {workflow['client_id']}")
        
        try:
            # Create stop message for agent server
            stop_message = {
                "type": "stop_workflow",
                "workflow_id": workflow['workflow_id'],
                "client_id": workflow['client_id']
            }
            
            # Send the message through agent WebSocket
            asyncio.create_task(prompt_server.agent_ws.send_json(stop_message))
            logger.info(f"Stop signal sent to agent server for workflow {workflow['workflow_id']}")
            
            # Notify UI of status change
            asyncio.create_task(prompt_server.send_sync("workflow_status", {
                "workflow_id": workflow['workflow_id'],
                "workflow_name": workflow['workflow_name'],
                "client_id": workflow['client_id'],
                "status": "stopping"
            }))
        except Exception as e:
            logger.error(f"Failed to send stop signal for workflow {workflow['workflow_id']}: {e}")
            # Continue trying to stop other workflows instead of failing completely