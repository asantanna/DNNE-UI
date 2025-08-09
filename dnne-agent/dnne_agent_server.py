#!/usr/bin/env python3
"""
DNNE Agent Server - Persistent service for workflow management and telemetry.

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
import argparse
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, Set, Optional, Any
from pathlib import Path
import uuid
from aiohttp import web

# Add parent directory to path to import dnne_config
sys.path.insert(0, str(Path(__file__).parent.parent))
from dnne_config import DNNEConfig

# Configure logging
# Set up logging to dnne_logs directory
import os as _os
_log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dnne_logs')
_os.makedirs(_log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(_log_dir, 'dnne_agent_server.log'))
    ]
)
logger = logging.getLogger('dnne_server')

# Suppress websockets library logging
logging.getLogger('websockets.server').setLevel(logging.WARNING)
logging.getLogger('websockets.protocol').setLevel(logging.WARNING)


class WorkflowInfo:
    """Information about a deployed workflow"""
    def __init__(self, workflow_id: str, client_id: str, files: Dict[str, str], workflow_name: str):
        # FAIL FAST: workflow_name is required
        if not workflow_name:
            raise ValueError(f"workflow_name is required for workflow {workflow_id}")
        self.workflow_id = workflow_id
        self.workflow_name = workflow_name
        self.client_id = client_id
        self.files = files
        self.deployed_at = datetime.now()
        self.status = "deployed"
        self.process_info = None
        self.start_time = None
        self.end_time = None


class DNNEAgentServer:
    """
    DNNE Agent Server - manages clients, workflows, and telemetry.
    
    Configurable ports loaded from dnne_config.json
    """
    
    def __init__(self, enable_test_port=False):
        # Load configuration
        self.config = DNNEConfig()
        self.enable_test_port = enable_test_port
        
        # Client management
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.client_info: Dict[str, Dict[str, Any]] = {}
        
        # UI connections
        self.ui_connections: Set[websockets.WebSocketServerProtocol] = set()
        
        # Test connections (only used when test port is enabled)
        self.test_connections: Set[websockets.WebSocketServerProtocol] = set()
        
        # Workflow management
        self.workflows: Dict[str, WorkflowInfo] = {}
        
        # Telemetry
        buffer_size = self.config.get('dnne.agent_server.telemetry_buffer_size', 1000)
        self.telemetry_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.last_telemetry_broadcast = time.time()
        
        # Logs
        log_buffer_size = self.config.get('dnne.agent_server.log_buffer_size', 10000)
        self.workflow_logs: Dict[str, deque] = defaultdict(lambda: deque(maxlen=log_buffer_size))
        
        # Server info
        self.start_time = time.time()
        
        # HTTP server for health checks
        self.http_app = web.Application()
        self.http_app.router.add_get('/health', self.handle_health)
        self.http_server = None
        
    async def start(self):
        """Start WebSocket servers"""
        # Get ports from config
        client_port = self.config.get('dnne.agent_server.client_port', 8766)
        ui_port = self.config.get('dnne.agent_server.ui_port', 8767)
        ping_interval = self.config.get('dnne.agent_server.ping_interval', 30)
        ping_timeout = self.config.get('dnne.agent_server.ping_timeout', 10)
        
        # Client server
        self.client_server = await websockets.serve(
            self.handle_client,
            "0.0.0.0",
            client_port,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout
        )
        logger.info(f"Client server started on port {client_port}")
        
        # UI server
        self.ui_server = await websockets.serve(
            self.handle_ui,
            "0.0.0.0", 
            ui_port,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout
        )
        logger.info(f"UI server started on port {ui_port}")
        
        # Test control server (only if enabled)
        if self.enable_test_port:
            test_port = self.config.get('dnne.agent_server.test_port', 8768)
            self.test_server = await websockets.serve(
                self.handle_test_control,
                "0.0.0.0",
                test_port
            )
            logger.warning(f"WARNING: Test control port enabled on {test_port} - DO NOT USE IN PRODUCTION")
        else:
            self.test_server = None
        
        # Start background tasks
        asyncio.create_task(self.telemetry_broadcaster())
        
        # Start HTTP health check server on a separate port
        health_port = self.config.get('dnne.agent_server.health_port', 8769)
        runner = web.AppRunner(self.http_app)
        await runner.setup()
        self.http_server = web.TCPSite(runner, '0.0.0.0', health_port)
        await self.http_server.start()
        logger.info(f"HTTP health check available at http://localhost:{health_port}/health")
    
    async def handle_health(self, request):
        """HTTP health check endpoint"""
        return web.json_response({
            'status': 'healthy',
            'uptime': time.time() - self.start_time,
            'connections': {
                'ui': len(self.ui_connections),
                'clients': len(self.clients),
                'test': len(self.test_connections) if self.enable_test_port else 0
            },
            'workflows': len(self.workflows),
            'server_time': datetime.now().isoformat()
        })
        
    async def handle_client(self, websocket):
        """Handle client connections (dnne_client)"""
        client_id = str(uuid.uuid4())[:8]
        self.clients[client_id] = websocket
        
        logger.info(f"Client {client_id} connected from {websocket.remote_address}")
        logger.debug(f"Open connections: UI: {len(self.ui_connections)}, agent: {len(self.clients)}, test: {len(self.test_connections)}")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                await self.handle_client_message(client_id, data)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
            logger.debug(f"Open connections: UI: {len(self.ui_connections)}, agent: {len(self.clients) - 1}, test: {len(self.test_connections)}")
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
            workflow_name = data.get("workflow_name")
            
            # Log the received status message
            logger.debug(f"[WORKFLOW_STATUS] Received from client {client_id}: workflow={workflow_id}, status={status}, name={workflow_name}")
            
            if workflow_id in self.workflows:
                workflow_info = self.workflows[workflow_id]
                workflow_info.status = status
                # Update workflow name if provided (in case it wasn't known during deployment)
                if workflow_name:
                    workflow_info.workflow_name = workflow_name
                if status == "running":
                    workflow_info.start_time = time.time()
                elif status in ["completed", "failed", "terminated"]:
                    workflow_info.end_time = time.time()
                    
                # FAIL-FAST: Critical fields must exist
                assert workflow_info.client_id is not None, f"workflow {workflow_id} missing client_id"
                assert workflow_info.workflow_name is not None, f"workflow {workflow_id} missing workflow_name"
                
                # Notify UIs and test connections with workflow name
                message = {
                    "type": "workflow_status",
                    "workflow_id": workflow_id,
                    "workflow_name": workflow_info.workflow_name,
                    "client_id": workflow_info.client_id,
                    "status": status,
                    "details": data.get("details")
                }
                logger.debug(f"[WORKFLOW_STATUS] Broadcasting to UIs: workflow={workflow_id}, status={status}, client_id={workflow_info.client_id}")
                await self.broadcast_to_ui(message)
            else:
                # FAIL FAST: This should never happen - workflows must be tracked before status updates
                logger.error(f"ERROR: Received status '{status}' for UNKNOWN workflow {workflow_id} from client {client_id}")
                logger.error(f"       This indicates a serious bug in the workflow tracking system!")
                logger.error(f"       Known workflows: {list(self.workflows.keys())}")
                
                # This is a critical error - we should not continue
                raise ValueError(f"Received status for unknown workflow {workflow_id}. This should never happen!")
                
                # Also broadcast to test connections
                if self.test_connections:
                    msg_json = json.dumps(message)
                    await asyncio.gather(
                        *[conn.send(msg_json) for conn in self.test_connections],
                        return_exceptions=True
                    )
                
        elif msg_type == "log":
            # Log message from workflow
            workflow_id = data.get("workflow_id")
            if workflow_id:
                logger.debug(f"[LOG] Received from client {client_id}: workflow={workflow_id}, message={data.get('message', '')[:100]}")
                
                self.workflow_logs[workflow_id].append({
                    "timestamp": time.time(),
                    "level": data.get("level", "info"),
                    "message": data.get("message", "")
                })
                
                # Forward to UIs and test connections
                message = {
                    "type": "workflow_log",
                    "workflow_id": workflow_id,
                    "log": data
                }
                logger.debug(f"[LOG] Broadcasting to {len(self.ui_connections)} UI(s): workflow={workflow_id}")
                await self.broadcast_to_ui(message)
                
                # Also forward to test connections
                if self.test_connections:
                    msg_json = json.dumps(message)
                    await asyncio.gather(
                        *[conn.send(msg_json) for conn in self.test_connections],
                        return_exceptions=True
                    )
    
    async def handle_ui(self, websocket):
        """Handle UI connections (DNNE-UI)"""
        self.ui_connections.add(websocket)
        connection_time = time.time()
        logger.info(f"UI connected from {websocket.remote_address}")
        logger.debug(f"Open connections: UI: {len(self.ui_connections)}, agent: {len(self.clients)}, test: {len(self.test_connections)}")
        
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
                logger.info(f"UI disconnected")
                logger.debug(f"Open connections: UI: {len(self.ui_connections) - 1}, agent: {len(self.clients)}, test: {len(self.test_connections)}")
        except Exception as e:
            logger.error(f"Error handling UI connection: {e}")
        finally:
            self.ui_connections.remove(websocket)
            # Log disconnection only for non-brief connections
            if not connection_was_brief and time.time() - connection_time >= 0.5:
                logger.info(f"UI connection closed")
                logger.debug(f"Open connections: UI: {len(self.ui_connections)}, agent: {len(self.clients)}, test: {len(self.test_connections)}")
    
    async def handle_ui_message(self, websocket, data: Dict[str, Any]):
        """Process messages from UI"""
        msg_type = data.get("type")
        
        if msg_type == "deploy_workflow":
            # Deploy workflow to client
            client_id = data.get("client_id")
            
            # If no client_id specified, check if any clients connected
            if not client_id:
                if not self.clients:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "No clients connected"
                    }))
                    return
                    
                # Use first connected client if not specified
                client_id = list(self.clients.keys())[0]
            
            # Verify client exists
            if client_id not in self.clients:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Client {client_id} not connected"
                }))
                return
            
            # Use workflow_id from message (generated by DNNE server based on content hash)
            workflow_id = data.get("workflow_id")
            workflow_name = data.get("workflow_name")
            
            # FAIL FAST: Both workflow_id and workflow_name are required
            if not workflow_id:
                error_msg = "workflow_id is required for deployment"
                logger.error(f"[DEPLOY ERROR] {error_msg}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": error_msg
                }))
                return
            
            if not workflow_name:
                error_msg = f"workflow_name is required for deployment of {workflow_id}"
                logger.error(f"[DEPLOY ERROR] {error_msg}")
                await websocket.send(json.dumps({
                    "type": "error", 
                    "message": error_msg
                }))
                return
            
            
            # Store workflow info with name
            self.workflows[workflow_id] = WorkflowInfo(
                workflow_id=workflow_id,
                client_id=client_id,
                files=data.get("files", {}),
                workflow_name=workflow_name
            )
            
            # Extract run_after_deploy flag
            run_after_deploy = data.get("run_after_deploy", False)
            
            # Send to client with run_after_deploy flag
            await self.clients[client_id].send(json.dumps({
                "type": "deploy",
                "workflow_id": workflow_id,
                "files": data.get("files", {}),
                "run_after_deploy": run_after_deploy
            }))
            
            # Acknowledge to UI with workflow name
            await websocket.send(json.dumps({
                "type": "workflow_deployed",
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "client_id": client_id
            }))
            
            logger.info(f"Deployed workflow {workflow_id} to client {client_id}, run_after_deploy={run_after_deploy}")
            
        elif msg_type == "start_workflow":
            # Start workflow execution
            workflow_id = data.get("workflow_id")
            
            if not workflow_id:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "workflow_id required"
                }))
                return
                
            if workflow_id not in self.workflows:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Workflow {workflow_id} not found"
                }))
                return
                
            workflow = self.workflows[workflow_id]
            if workflow.client_id not in self.clients:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Client {workflow.client_id} not connected"
                }))
                return
                
            await self.clients[workflow.client_id].send(json.dumps({
                "type": "start",
                "workflow_id": workflow_id,
                "args": data.get("args", [])
            }))
                    
        elif msg_type == "stop_workflow":
            # Stop workflow execution
            workflow_id = data.get("workflow_id")
            
            if not workflow_id:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "workflow_id required"
                }))
                return
                
            if workflow_id not in self.workflows:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Workflow {workflow_id} not found"
                }))
                return
                
            workflow = self.workflows[workflow_id]
            if workflow.client_id not in self.clients:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Client {workflow.client_id} not connected"
                }))
                return
                
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
    
    async def handle_test_control(self, websocket):
        """Handle test control connections (test harness only)"""
        self.test_connections.add(websocket)
        logger.warning(f"Test control connection from {websocket.remote_address}")
        logger.debug(f"Open connections: UI: {len(self.ui_connections)}, agent: {len(self.clients)}, test: {len(self.test_connections)}")
        
        try:
            # Send current state to test client
            await websocket.send(json.dumps({
                "type": "state",
                "clients": {
                    client_id: {
                        **self.client_info.get(client_id, {}),
                        "connected": client_id in self.clients
                    }
                    for client_id in self.client_info
                },
                "workflows": {
                    wf_id: {
                        "client_id": wf.client_id,
                        "status": wf.status,
                        "start_time": wf.start_time,
                        "end_time": wf.end_time
                    }
                    for wf_id, wf in self.workflows.items()
                }
            }))
            
            async for message in websocket:
                data = json.loads(message)
                # Forward test commands to UI handler
                await self.handle_ui_message(websocket, data)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Test control connection closed")
            logger.debug(f"Open connections: UI: {len(self.ui_connections)}, agent: {len(self.clients)}, test: {len(self.test_connections) - 1})")
        except Exception as e:
            logger.error(f"Error handling test control connection: {e}")
        finally:
            self.test_connections.remove(websocket)
    
    async def broadcast_to_ui(self, data: Dict[str, Any]):
        """Broadcast message to all connected UIs"""
        if self.ui_connections:
            logger.debug(f"[BROADCAST] Sending to {len(self.ui_connections)} UI(s): type={data.get('type')}, data={data}")
            message = json.dumps(data)
            results = await asyncio.gather(
                *[ui.send(message) for ui in self.ui_connections],
                return_exceptions=True
            )
            # Log any errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"[BROADCAST] Failed to send to UI {i}: {result}")
        else:
            logger.warning(f"[BROADCAST] No UI connections to send message: type={data.get('type')}")
    
    async def telemetry_broadcaster(self):
        """Periodically broadcast telemetry to UIs"""
        while True:
            batch_interval = self.config.get('exported.agent_client.telemetry_batch_interval', 0.1)
            await asyncio.sleep(batch_interval)
            
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
        logger.info("DNNE Agent Server started successfully")
        logger.info(f"  Client port: {self.config.get('dnne.agent_server.client_port', 8766)}")
        logger.info(f"  UI port: {self.config.get('dnne.agent_server.ui_port', 8767)}")
        logger.info(f"  Health port: {self.config.get('dnne.agent_server.health_port', 8769)}")
        if self.enable_test_port:
            logger.info(f"  Test port: {self.config.get('dnne.agent_server.test_port', 8768)} (TEST MODE ONLY)")
        
        # Keep running
        await asyncio.Future()  # Run forever


async def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DNNE Agent Server")
    parser.add_argument("--enable-test-port", action="store_true",
                       help="Enable test control port (DO NOT USE IN PRODUCTION)")
    parser.add_argument("--verbose", default='INFO', const='DEBUG', nargs="?", 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       help='Set the logging level (default: INFO, --verbose alone: DEBUG)')
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    log_level = getattr(logging, args.verbose)
    logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level)
    if args.verbose == 'DEBUG':
        logger.info("DEBUG logging enabled")
    else:
        logger.info(f"Logging level set to {args.verbose}")
    
    # Create server with test port if requested
    server = DNNEAgentServer(enable_test_port=args.enable_test_port)
    
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