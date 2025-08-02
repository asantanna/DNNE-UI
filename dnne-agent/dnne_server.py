#!/usr/bin/env python3
"""
DNNE Server - Persistent service for workflow management and telemetry.

Runs as a background service on Windows, managing:
- Client connections (WSL/Linux)
- Workflow deployment and execution
- Telemetry aggregation
- Log collection
"""

import asyncio
import websockets
import json
import time
import logging
import sys
import os
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, Set, Optional, Any
from pathlib import Path
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dnne_server.log')
    ]
)
logger = logging.getLogger('dnne_server')


class WorkflowInfo:
    """Information about a deployed workflow"""
    def __init__(self, workflow_id: str, client_id: str, files: Dict[str, str]):
        self.workflow_id = workflow_id
        self.client_id = client_id
        self.files = files
        self.deployed_at = datetime.now()
        self.status = "deployed"
        self.process_info = None
        self.start_time = None
        self.end_time = None


class DNNEServer:
    """
    DNNE Server - manages clients, workflows, and telemetry.
    
    Ports:
    - 8766: Client connections (dnne_client)
    - 8767: UI connections (DNNE-UI)
    """
    
    def __init__(self):
        # Client management
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.client_info: Dict[str, Dict[str, Any]] = {}
        
        # UI connections
        self.ui_connections: Set[websockets.WebSocketServerProtocol] = set()
        
        # Workflow management
        self.workflows: Dict[str, WorkflowInfo] = {}
        self.active_workflow: Optional[str] = None  # Single workflow for now
        
        # Telemetry
        self.telemetry_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.last_telemetry_broadcast = time.time()
        
        # Logs
        self.workflow_logs: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Server info
        self.start_time = time.time()
        
    async def start(self):
        """Start WebSocket servers"""
        # Client server (port 8766)
        self.client_server = await websockets.serve(
            self.handle_client,
            "0.0.0.0",
            8766,
            ping_interval=30,
            ping_timeout=10
        )
        logger.info("Client server started on port 8766")
        
        # UI server (port 8767)
        self.ui_server = await websockets.serve(
            self.handle_ui,
            "0.0.0.0", 
            8767,
            ping_interval=30,
            ping_timeout=10
        )
        logger.info("UI server started on port 8767")
        
        # Start background tasks
        asyncio.create_task(self.telemetry_broadcaster())
        
    async def handle_client(self, websocket):
        """Handle client connections (dnne_client)"""
        client_id = str(uuid.uuid4())[:8]
        self.clients[client_id] = websocket
        
        logger.info(f"Client {client_id} connected from {websocket.remote_address}")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                await self.handle_client_message(client_id, data)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            # Cleanup
            del self.clients[client_id]
            if client_id in self.client_info:
                del self.client_info[client_id]
            
            # Notify UIs
            await self.broadcast_to_ui({
                "type": "client_disconnected",
                "client_id": client_id
            })
    
    async def handle_client_message(self, client_id: str, data: Dict[str, Any]):
        """Process messages from dnne_client"""
        msg_type = data.get("type")
        
        if msg_type == "register":
            # Client registration
            self.client_info[client_id] = {
                "hostname": data.get("hostname", "unknown"),
                "capabilities": data.get("capabilities", {}),
                "connected_at": time.time()
            }
            
            # Send acknowledgment
            await self.clients[client_id].send(json.dumps({
                "type": "registered",
                "client_id": client_id
            }))
            
            # Notify UIs
            await self.broadcast_to_ui({
                "type": "client_connected",
                "client_id": client_id,
                "info": self.client_info[client_id]
            })
            
            logger.info(f"Client {client_id} registered: {self.client_info[client_id]}")
            
        elif msg_type == "telemetry":
            # Telemetry data
            metrics = data.get("metrics", [])
            for metric in metrics:
                node_id = metric.get("node_id")
                if node_id:
                    self.telemetry_buffer[node_id].append(metric)
                    
        elif msg_type == "workflow_status":
            # Workflow status update
            workflow_id = data.get("workflow_id")
            status = data.get("status")
            
            if workflow_id in self.workflows:
                self.workflows[workflow_id].status = status
                if status == "running":
                    self.workflows[workflow_id].start_time = time.time()
                elif status in ["completed", "failed", "stopped"]:
                    self.workflows[workflow_id].end_time = time.time()
                    
                # Notify UIs
                await self.broadcast_to_ui({
                    "type": "workflow_status",
                    "workflow_id": workflow_id,
                    "status": status,
                    "details": data.get("details")
                })
                
        elif msg_type == "log":
            # Log message from workflow
            workflow_id = data.get("workflow_id")
            if workflow_id:
                self.workflow_logs[workflow_id].append({
                    "timestamp": time.time(),
                    "level": data.get("level", "info"),
                    "message": data.get("message", "")
                })
                
                # Forward to UIs
                await self.broadcast_to_ui({
                    "type": "workflow_log",
                    "workflow_id": workflow_id,
                    "log": data
                })
    
    async def handle_ui(self, websocket):
        """Handle UI connections (DNNE-UI)"""
        self.ui_connections.add(websocket)
        connection_time = time.time()
        logger.info(f"UI connected from {websocket.remote_address} ({len(self.ui_connections)} connection(s) open)")
        
        connection_was_brief = False
        try:
            # Send current state to new UI
            await websocket.send(json.dumps({
                "type": "server_state",
                "clients": {
                    client_id: {
                        **info,
                        "connected": client_id in self.clients
                    }
                    for client_id, info in self.client_info.items()
                },
                "workflows": {
                    wf_id: {
                        "workflow_id": wf.workflow_id,
                        "client_id": wf.client_id,
                        "status": wf.status,
                        "deployed_at": wf.deployed_at.isoformat()
                    }
                    for wf_id, wf in self.workflows.items()
                },
                "server_uptime": time.time() - self.start_time
            }))
            
            # Handle UI messages
            async for message in websocket:
                data = json.loads(message)
                await self.handle_ui_message(websocket, data)
                
        except websockets.exceptions.ConnectionClosed:
            # Check if this was a brief connection (likely a health check)
            if time.time() - connection_time < 0.5:
                connection_was_brief = True
            else:
                logger.info("UI disconnected")
        except Exception as e:
            logger.error(f"Error handling UI connection: {e}")
        finally:
            self.ui_connections.remove(websocket)
            # Log disconnection only for non-brief connections
            if not connection_was_brief and time.time() - connection_time >= 0.5:
                logger.info("UI connection closed")
    
    async def handle_ui_message(self, websocket, data: Dict[str, Any]):
        """Process messages from UI"""
        msg_type = data.get("type")
        
        if msg_type == "deploy_workflow":
            # Deploy workflow to client
            if not self.clients:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "No clients connected"
                }))
                return
                
            # For now, use first connected client
            client_id = list(self.clients.keys())[0]
            workflow_id = f"wf_{uuid.uuid4().hex[:8]}"
            
            # Store workflow info
            self.workflows[workflow_id] = WorkflowInfo(
                workflow_id=workflow_id,
                client_id=client_id,
                files=data.get("files", {})
            )
            
            # Send to client
            await self.clients[client_id].send(json.dumps({
                "type": "deploy",
                "workflow_id": workflow_id,
                "files": data.get("files", {})
            }))
            
            # Set as active workflow
            self.active_workflow = workflow_id
            
            # Acknowledge to UI
            await websocket.send(json.dumps({
                "type": "workflow_deployed",
                "workflow_id": workflow_id,
                "client_id": client_id
            }))
            
            logger.info(f"Deployed workflow {workflow_id} to client {client_id}")
            
        elif msg_type == "start_workflow":
            # Start workflow execution
            workflow_id = data.get("workflow_id", self.active_workflow)
            if workflow_id and workflow_id in self.workflows:
                workflow = self.workflows[workflow_id]
                if workflow.client_id in self.clients:
                    await self.clients[workflow.client_id].send(json.dumps({
                        "type": "start",
                        "workflow_id": workflow_id,
                        "args": data.get("args", [])
                    }))
                    
        elif msg_type == "stop_workflow":
            # Stop workflow execution
            workflow_id = data.get("workflow_id", self.active_workflow)
            if workflow_id and workflow_id in self.workflows:
                workflow = self.workflows[workflow_id]
                if workflow.client_id in self.clients:
                    await self.clients[workflow.client_id].send(json.dumps({
                        "type": "stop",
                        "workflow_id": workflow_id
                    }))
                    
        elif msg_type == "get_logs":
            # Get workflow logs
            workflow_id = data.get("workflow_id")
            if workflow_id in self.workflow_logs:
                await websocket.send(json.dumps({
                    "type": "logs",
                    "workflow_id": workflow_id,
                    "logs": list(self.workflow_logs[workflow_id])[-100:]  # Last 100 lines
                }))
    
    async def broadcast_to_ui(self, data: Dict[str, Any]):
        """Broadcast message to all connected UIs"""
        if self.ui_connections:
            message = json.dumps(data)
            await asyncio.gather(
                *[ui.send(message) for ui in self.ui_connections],
                return_exceptions=True
            )
    
    async def telemetry_broadcaster(self):
        """Periodically broadcast telemetry to UIs"""
        while True:
            await asyncio.sleep(0.1)  # 100ms intervals
            
            if self.ui_connections and self.telemetry_buffer:
                # Prepare telemetry batch
                batch = {}
                for node_id, metrics in self.telemetry_buffer.items():
                    if metrics:
                        # Get latest metrics
                        batch[node_id] = list(metrics)[-10:]  # Last 10 metrics
                        
                # Broadcast to UIs
                await self.broadcast_to_ui({
                    "type": "telemetry_batch",
                    "timestamp": time.time(),
                    "metrics": batch
                })
                
                # Clear sent metrics
                for node_id in batch:
                    self.telemetry_buffer[node_id].clear()
    
    async def run_forever(self):
        """Run the server forever"""
        await self.start()
        logger.info("DNNE Server started successfully")
        logger.info("  Client port: 8766")
        logger.info("  UI port: 8767")
        
        # Keep running
        await asyncio.Future()  # Run forever


async def main():
    """Main entry point"""
    server = DNNEServer()
    
    try:
        await server.run_forever()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    # Run the server
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped")