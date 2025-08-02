#!/usr/bin/env python3
"""
DNNE Client - Executes workflows and forwards telemetry.

Runs on Linux/WSL, connecting to dnne_server to:
- Receive workflow deployments
- Execute workflows locally
- Forward telemetry from nodes
- Manage process lifecycle
"""

import asyncio
import websockets
import json
import socket
import time
import logging
import sys
import os
import subprocess
import signal
from pathlib import Path
from typing import Dict, Optional, Any
from collections import deque
import tempfile
import shutil
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dnne_client')


class WorkflowProcess:
    """Manages a running workflow process"""
    def __init__(self, workflow_id: str, process: subprocess.Popen, workspace: Path):
        self.workflow_id = workflow_id
        self.process = process
        self.workspace = workspace
        self.start_time = time.time()
        self.log_reader_task = None


class DNNEClient:
    """
    DNNE Client - executes workflows and forwards telemetry.
    
    Connects to dnne_server and provides:
    - Workflow deployment and execution
    - UDP telemetry listener (port 9999)
    - WebSocket control server for nodes (port 9998)
    - Process management
    """
    
    def __init__(self, server_url: str = "ws://localhost:8766"):
        self.server_url = server_url
        self.websocket = None
        self.client_id = None
        
        # Check conda environment
        self._check_conda_environment()
        
        # Workflow management
        self.workflows: Dict[str, WorkflowProcess] = {}
        self.workspace_base = Path(tempfile.gettempdir()) / "dnne_workspaces"
        self.workspace_base.mkdir(exist_ok=True)
        
        # Telemetry
        self.telemetry_buffer = deque(maxlen=1000)
        self.telemetry_socket = None
        
        # Control server for nodes
        self.node_connections = set()
    
    def _check_conda_environment(self):
        """Check that DNNE_PY38 conda environment is activated"""
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
        
        if conda_env != 'DNNE_PY38':
            logger.error("❌ DNNE_PY38 conda environment is not activated!")
            logger.error(f"   Current environment: {conda_env or 'None'}")
            logger.error("   Please activate the environment:")
            logger.error("   $ conda activate DNNE_PY38")
            logger.error("   or")
            logger.error("   $ source ~/miniconda3/bin/activate DNNE_PY38")
            sys.exit(1)
        
        logger.info("✅ DNNE_PY38 conda environment is active")
        
    async def connect_to_server(self):
        """Connect to dnne_server"""
        logger.info(f"Connecting to {self.server_url}")
        
        try:
            self.websocket = await websockets.connect(
                self.server_url,
                ping_interval=30,
                ping_timeout=10
            )
            
            # Register with server
            await self.websocket.send(json.dumps({
                "type": "register",
                "hostname": socket.gethostname(),
                "capabilities": {
                    "platform": sys.platform,
                    "python_version": sys.version,
                    "cpu_count": os.cpu_count(),
                    "memory_gb": psutil.virtual_memory().total / (1024**3),
                    "gpu": self._check_gpu()
                }
            }))
            
            # Wait for registration confirmation
            response = await self.websocket.recv()
            data = json.loads(response)
            if data.get("type") == "registered":
                self.client_id = data.get("client_id")
                logger.info(f"Registered with server as client {self.client_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False
            
        return False
    
    def _check_gpu(self) -> Dict[str, Any]:
        """Check GPU availability"""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "available": True,
                    "count": torch.cuda.device_count(),
                    "device_name": torch.cuda.get_device_name(0)
                }
        except ImportError:
            pass
            
        return {"available": False}
    
    async def start_telemetry_listener(self):
        """Start UDP listener for telemetry"""
        try:
            # Create UDP socket
            self.telemetry_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.telemetry_socket.bind(('localhost', 9999))
            self.telemetry_socket.setblocking(False)
            
            logger.info("Telemetry listener started on localhost:9999")
            
            while True:
                try:
                    # Non-blocking receive
                    data, addr = self.telemetry_socket.recvfrom(1024)
                    
                    # Parse telemetry packet
                    packet = self._parse_telemetry(data.decode())
                    if packet:
                        self.telemetry_buffer.append(packet)
                        
                except BlockingIOError:
                    # No data available
                    await asyncio.sleep(0.001)
                except Exception as e:
                    logger.error(f"Telemetry receive error: {e}")
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Failed to start telemetry listener: {e}")
    
    def _parse_telemetry(self, data: str) -> Optional[Dict[str, Any]]:
        """Parse telemetry packet"""
        try:
            # Try JSON format first
            if data.startswith('{'):
                return json.loads(data)
            
            # Parse pipe-delimited format: metric|node_id|value|timestamp
            parts = data.split('|')
            if len(parts) >= 4:
                return {
                    "metric": parts[0],
                    "node_id": parts[1],
                    "value": float(parts[2]),
                    "timestamp": float(parts[3])
                }
                
        except Exception as e:
            logger.debug(f"Failed to parse telemetry: {e}")
            
        return None
    
    async def telemetry_forwarder(self):
        """Forward telemetry to server"""
        while True:
            await asyncio.sleep(0.1)  # Batch every 100ms
            
            if self.telemetry_buffer and self.websocket:
                # Send batch to server
                batch = list(self.telemetry_buffer)
                self.telemetry_buffer.clear()
                
                try:
                    await self.websocket.send(json.dumps({
                        "type": "telemetry",
                        "metrics": batch
                    }))
                except Exception as e:
                    logger.error(f"Failed to forward telemetry: {e}")
    
    async def handle_server_message(self, data: Dict[str, Any]):
        """Handle messages from dnne_server"""
        msg_type = data.get("type")
        
        if msg_type == "deploy":
            # Deploy workflow
            workflow_id = data.get("workflow_id")
            files = data.get("files", {})
            
            logger.info(f"Deploying workflow {workflow_id}")
            
            try:
                # Create workspace
                workspace = self.workspace_base / workflow_id
                workspace.mkdir(exist_ok=True)
                
                # Write files
                for file_path, content in files.items():
                    full_path = workspace / file_path
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(content)
                    
                logger.info(f"Deployed {len(files)} files to {workspace}")
                
                # Notify server
                await self.websocket.send(json.dumps({
                    "type": "workflow_status",
                    "workflow_id": workflow_id,
                    "status": "deployed"
                }))
                
            except Exception as e:
                logger.error(f"Deployment failed: {e}")
                await self.websocket.send(json.dumps({
                    "type": "workflow_status",
                    "workflow_id": workflow_id,
                    "status": "deploy_failed",
                    "details": str(e)
                }))
                
        elif msg_type == "start":
            # Start workflow
            workflow_id = data.get("workflow_id")
            args = data.get("args", [])
            
            await self.start_workflow(workflow_id, args)
            
        elif msg_type == "stop":
            # Stop workflow
            workflow_id = data.get("workflow_id")
            await self.stop_workflow(workflow_id)
    
    async def start_workflow(self, workflow_id: str, args: list):
        """Start workflow execution"""
        if workflow_id in self.workflows:
            logger.warning(f"Workflow {workflow_id} already running")
            return
            
        workspace = self.workspace_base / workflow_id
        runner_path = workspace / "runner.py"
        
        if not runner_path.exists():
            logger.error(f"Runner not found: {runner_path}")
            await self.websocket.send(json.dumps({
                "type": "workflow_status",
                "workflow_id": workflow_id,
                "status": "start_failed",
                "details": "runner.py not found"
            }))
            return
            
        try:
            # Start process (conda environment already verified to be active)
            cmd = [sys.executable, "runner.py"] + args
            
            process = subprocess.Popen(
                cmd,
                cwd=workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Store process info
            self.workflows[workflow_id] = WorkflowProcess(workflow_id, process, workspace)
            
            # Start log reader
            self.workflows[workflow_id].log_reader_task = asyncio.create_task(
                self.read_process_logs(workflow_id)
            )
            
            logger.info(f"Started workflow {workflow_id} (PID: {process.pid})")
            
            # Notify server
            await self.websocket.send(json.dumps({
                "type": "workflow_status",
                "workflow_id": workflow_id,
                "status": "running",
                "details": {"pid": process.pid}
            }))
            
        except Exception as e:
            logger.error(f"Failed to start workflow: {e}")
            await self.websocket.send(json.dumps({
                "type": "workflow_status",
                "workflow_id": workflow_id,
                "status": "start_failed",
                "details": str(e)
            }))
    
    async def stop_workflow(self, workflow_id: str):
        """Stop workflow execution"""
        if workflow_id not in self.workflows:
            logger.warning(f"Workflow {workflow_id} not running")
            return
            
        workflow = self.workflows[workflow_id]
        
        try:
            # Send SIGTERM
            workflow.process.terminate()
            
            # Wait up to 10 seconds
            try:
                workflow.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill
                workflow.process.kill()
                workflow.process.wait()
                
            # Cancel log reader
            if workflow.log_reader_task:
                workflow.log_reader_task.cancel()
                
            # Cleanup
            del self.workflows[workflow_id]
            
            logger.info(f"Stopped workflow {workflow_id}")
            
            # Notify server
            await self.websocket.send(json.dumps({
                "type": "workflow_status",
                "workflow_id": workflow_id,
                "status": "stopped"
            }))
            
        except Exception as e:
            logger.error(f"Failed to stop workflow: {e}")
    
    async def read_process_logs(self, workflow_id: str):
        """Read process output and send to server"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return
            
        try:
            while True:
                line = workflow.process.stdout.readline()
                if not line:
                    # Process ended
                    break
                    
                # Send log to server
                if self.websocket:
                    await self.websocket.send(json.dumps({
                        "type": "log",
                        "workflow_id": workflow_id,
                        "level": "info",
                        "message": line.strip()
                    }))
                    
                await asyncio.sleep(0)  # Yield control
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Log reader error: {e}")
        finally:
            # Check exit code
            exit_code = workflow.process.poll()
            if exit_code is not None:
                status = "completed" if exit_code == 0 else "failed"
                
                if self.websocket:
                    await self.websocket.send(json.dumps({
                        "type": "workflow_status",
                        "workflow_id": workflow_id,
                        "status": status,
                        "details": {"exit_code": exit_code}
                    }))
                    
                # Cleanup
                if workflow_id in self.workflows:
                    del self.workflows[workflow_id]
    
    async def run(self):
        """Main client loop"""
        # Connect to server
        if not await self.connect_to_server():
            logger.error("Failed to connect to server")
            return
            
        # Start background tasks
        telemetry_task = asyncio.create_task(self.start_telemetry_listener())
        forwarder_task = asyncio.create_task(self.telemetry_forwarder())
        
        try:
            # Handle server messages
            async for message in self.websocket:
                data = json.loads(message)
                await self.handle_server_message(data)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection to server lost")
        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            # Cleanup
            telemetry_task.cancel()
            forwarder_task.cancel()
            
            # Stop all workflows
            for workflow_id in list(self.workflows.keys()):
                await self.stop_workflow(workflow_id)
                
            if self.telemetry_socket:
                self.telemetry_socket.close()


async def main():
    """Main entry point"""
    # Get server URL from environment or use default
    server_url = os.environ.get('DNNE_SERVER_URL', 'ws://localhost:8766')
    
    client = DNNEClient(server_url)
    
    try:
        await client.run()
    except KeyboardInterrupt:
        logger.info("Client shutdown requested")
    except Exception as e:
        logger.error(f"Client error: {e}")
        raise


if __name__ == "__main__":
    # Run the client
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Client stopped")