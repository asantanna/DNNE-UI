#!/usr/bin/env python3
"""
DNNE Agent Client - Executes workflows and forwards telemetry.

Runs on Linux/WSL, connecting to dnne_agent_server to:
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
from typing import Dict, Optional, Any, Tuple
from collections import deque
import tempfile
import shutil
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dnne_agent_client')


class WorkflowProcess:
    """Manages a running workflow process"""
    def __init__(self, workflow_id: str, process: asyncio.subprocess.Process, workspace: Path):
        self.workflow_id = workflow_id
        self.process = process
        self.workspace = workspace
        self.start_time = time.time()
        self.log_reader_task = None


class TelemetryProtocol(asyncio.DatagramProtocol):
    """Async UDP protocol for receiving telemetry"""
    
    def __init__(self, client):
        self.client = client
        self.transport = None
    
    def connection_made(self, transport):
        self.transport = transport
        logger.info(f"Telemetry UDP listener ready on {transport.get_extra_info('sockname')}")
    
    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        """Handle incoming telemetry packet"""
        try:
            # Parse telemetry packet
            packet = self._parse_telemetry(data.decode())
            if packet:
                self.client.telemetry_buffer.append(packet)
        except Exception as e:
            logger.debug(f"Failed to process telemetry from {addr}: {e}")
    
    def _parse_telemetry(self, data: str) -> Optional[Dict[str, Any]]:
        """Parse telemetry packet (JSON only)"""
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            logger.debug(f"Invalid JSON telemetry: {data[:50]}...")
            return None
    
    def error_received(self, exc):
        logger.error(f"Telemetry UDP error: {exc}")
    
    def connection_lost(self, exc):
        if exc:
            logger.error(f"Telemetry UDP connection lost: {exc}")


class DNNEAgentClient:
    """
    DNNE Agent Client - executes workflows and forwards telemetry.
    
    Connects to dnne_agent_server and provides:
    - Workflow deployment and execution
    - UDP telemetry listener (configurable port)
    - Process management
    """
    
    def __init__(self, config_path: Optional[str] = None, server_url: Optional[str] = None):
        # Load configuration
        self._load_config(config_path)
        # Set server URL (parameter overrides config)
        if server_url:
            self.server_url = server_url
        else:
            server_port = self.get('dnne.agent_server.client_port', 8766)
            self.server_url = f"ws://localhost:{server_port}"
        
        self.websocket = None
        self.client_id = None
        
        # Check conda environment
        self._check_conda_environment()
        
        # Workflow management
        self.workflows: Dict[str, WorkflowProcess] = {}
        work_area = self.get('agent_client.work_area_base', '/tmp/dnne_work_areas')
        self.workspace_base = Path(os.path.expanduser(work_area))
        self.workspace_base.mkdir(exist_ok=True)
        
        # Telemetry
        buffer_size = self.get('agent_client.telemetry_buffer_size', 1000)
        self.telemetry_buffer = deque(maxlen=buffer_size)
        self.telemetry_transport = None
        self.telemetry_protocol = None
        self.telemetry_port = self.get('agent_client.telemetry_port', 9999)
        
        # Control server for nodes
        self.node_connections = set()
    
    def _load_config(self, config_path: Optional[str]):
        """Load configuration from exported_config.json"""
        if config_path is None:
            # Look for exported_config.json in standard locations
            possible_paths = [
                Path("exported_config.json"),
                Path(__file__).parent / "exported_config.json",
                Path.home() / ".dnne" / "exported_config.json"
            ]
            
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break
            else:
                # Use empty config if not found
                logger.warning("No exported_config.json found, using defaults")
                self.config = {}
                return
        
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def _check_conda_environment(self):
        """Check that required conda environment is activated"""
        required_env = self.get('conda.conda_env', 'DNNE_PY38')
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
        
        if conda_env != required_env:
            logger.error(f"❌ {required_env} conda environment is not activated!")
            logger.error(f"   Current environment: {conda_env or 'None'}")
            logger.error("   Please activate the environment:")
            
            conda_path = self.get('conda.conda_path', '~/miniconda')
            conda_path = os.path.expanduser(conda_path)
            logger.error(f"   $ conda activate {required_env}")
            logger.error("   or")
            logger.error(f"   $ source {conda_path}/bin/activate {required_env}")
            sys.exit(1)
        
        logger.info(f"✅ {required_env} conda environment is active")
        
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
        """Start UDP listener for telemetry using asyncio"""
        try:
            loop = asyncio.get_event_loop()
            
            # Create datagram endpoint
            transport, protocol = await loop.create_datagram_endpoint(
                lambda: TelemetryProtocol(self),
                local_addr=('localhost', self.telemetry_port)
            )
            
            self.telemetry_transport = transport
            self.telemetry_protocol = protocol
            
            logger.info(f"Telemetry listener started on localhost:{self.telemetry_port}")
            
            # Keep the listener running
            await asyncio.Future()  # Run forever
            
        except Exception as e:
            logger.error(f"Failed to start telemetry listener: {e}")
    
    async def telemetry_forwarder(self):
        """Forward telemetry to server"""
        batch_interval = self.get('agent_client.telemetry_batch_interval', 0.1)
        
        while True:
            await asyncio.sleep(batch_interval)
            
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
                    # Convert Windows paths to Posix paths
                    normalized_path = file_path.replace('\\', '/')
                    full_path = workspace / normalized_path
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
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
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
            
            # Wait for configured timeout
            stop_timeout = self.get('agent_client.workflow_stop_timeout', 10)
            try:
                await asyncio.wait_for(workflow.process.wait(), timeout=stop_timeout)
            except asyncio.TimeoutError:
                # Force kill
                workflow.process.kill()
                await workflow.process.wait()
                
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
                # Read line asynchronously
                line_bytes = await workflow.process.stdout.readline()
                if not line_bytes:
                    # Process ended
                    break
                
                line = line_bytes.decode('utf-8', errors='replace').strip()
                
                # Send log to server
                if self.websocket and line:
                    await self.websocket.send(json.dumps({
                        "type": "log",
                        "workflow_id": workflow_id,
                        "level": "info",
                        "message": line
                    }))
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Log reader error: {e}")
        finally:
            # Wait for process to complete and get exit code
            exit_code = await workflow.process.wait()
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
                
            if self.telemetry_transport:
                self.telemetry_transport.close()


async def main():
    """Main entry point"""
    # Get config path from environment or use default
    config_path = os.environ.get('DNNE_CONFIG_PATH')
    
    # Get server URL from environment or use config
    server_url = os.environ.get('DNNE_SERVER_URL')
    
    client = DNNEAgentClient(config_path=config_path, server_url=server_url)
    
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