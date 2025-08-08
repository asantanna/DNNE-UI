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
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from collections import deque
import tempfile
import shutil
import psutil

# Configure logging - will be set up properly in main()
logger = logging.getLogger('dnne_agent_client')


class WorkflowProcess:
    """Manages a running workflow process"""
    def __init__(self, workflow_id: str, process: asyncio.subprocess.Process, workspace: Path, workflow_name: str = None):
        self.workflow_id = workflow_id
        self.workflow_name = workflow_name or workflow_id  # Fallback to ID if name not available
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
    
    def __init__(self, config_path: Optional[str] = None, server_ip: Optional[str] = None):
        # Load configuration
        self._load_config(config_path)
        
        # Resolve server URL
        self.server_url = self._resolve_server_url(server_ip)
        logger.info(f"Server URL resolved to: {self.server_url}")
        
        self.websocket = None
        self.client_id = None
        
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
        """Load configuration from dnne_config.json"""
        if config_path is None:
            # Look for dnne_config.json in script directory
            config_path = Path(__file__).parent / "dnne_config.json"
            if not config_path.exists():
                logger.warning("No dnne_config.json found, using empty config")
                self.config = {}
                return
        else:
            config_path = Path(config_path)
            if not config_path.exists():
                logger.error(f"Config file not found: {config_path}")
                sys.exit(1)
        
        # Load config file
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            sys.exit(1)
    
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
    
    def _is_wsl(self) -> bool:
        """Check if running in WSL"""
        try:
            with open('/proc/version', 'r') as f:
                version = f.read().lower()
                return 'microsoft' in version or 'wsl' in version
        except:
            return False
    
    def _get_wsl_host_ip(self) -> str:
        """Get Windows host IP address when running in WSL"""
        try:
            result = subprocess.run(
                ['ip', 'route', 'show', 'default'],
                capture_output=True,
                text=True,
                check=True
            )
            # Parse the output to get the gateway IP
            # Format: "default via 172.22.160.1 dev eth0"
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.startswith('default via'):
                    parts = line.split()
                    if len(parts) >= 3:
                        return parts[2]
            raise Exception("Could not parse gateway IP from ip route output")
        except Exception as e:
            logger.error(f"Failed to get WSL host IP: {e}")
            sys.exit(1)
    
    def _resolve_server_url(self, server_ip: Optional[str]) -> str:
        """Resolve the server URL from config or command-line argument"""
        default_port = 8766
        
        if server_ip:
            # Server IP was provided via command line
            if server_ip.lower() == 'auto':
                # Auto mode - must be in WSL
                if not self._is_wsl():
                    logger.error("Error: --server_ip is 'auto' but not running in WSL")
                    sys.exit(1)
                host = self._get_wsl_host_ip()
                port = self.get('dnne.agent_server.client_port', default_port)
                logger.info(f"Auto-detected WSL host IP: {host}")
            else:
                # Parse IP:port or just IP
                if ':' in server_ip:
                    host, port_str = server_ip.rsplit(':', 1)
                    try:
                        port = int(port_str)
                    except ValueError:
                        logger.error(f"Invalid port in server_ip: {port_str}")
                        sys.exit(1)
                else:
                    host = server_ip
                    port = self.get('dnne.agent_server.client_port', default_port)
        else:
            # No server IP provided - try config first
            host = self.get('dnne.agent_server.host')
            port = self.get('dnne.agent_server.client_port', default_port)
            
            if not host:
                # Not in config - try auto if in WSL
                if self._is_wsl():
                    logger.info("No server IP in config, trying auto-detection for WSL...")
                    host = self._get_wsl_host_ip()
                    logger.info(f"Auto-detected WSL host IP: {host}")
                else:
                    logger.error("Error: Server IP not specified and not found in dnne_config.json")
                    sys.exit(1)
        
        return f"ws://{host}:{port}"
        
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
        # FAIL-FAST: If torch is expected, it must be available
        import torch
        if torch.cuda.is_available():
            return {
                "available": True,
                "count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0)
            }
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
                
                # FAIL-FAST: If websocket is broken, we should know
                await self.websocket.send(json.dumps({
                    "type": "telemetry",
                    "metrics": batch
                }))
    
    async def handle_server_message(self, data: Dict[str, Any]):
        """Handle messages from dnne_server"""
        msg_type = data.get("type")
        
        if msg_type == "deploy":
            # Deploy workflow
            workflow_id = data.get("workflow_id")
            files = data.get("files", {})
            run_after_deploy = data.get("run_after_deploy", False)
            
            logger.info(f"Deploying workflow {workflow_id}, run_after_deploy={run_after_deploy}")
            
            try:
                # Clean deployment directory if it exists
                workspace = self.workspace_base / workflow_id
                if workspace.exists():
                    import shutil
                    logger.info(f"Cleaning existing deployment at {workspace}")
                    shutil.rmtree(workspace)
                
                # Create fresh workspace
                workspace.mkdir(parents=True, exist_ok=True)
                
                # Write files
                for file_path, content in files.items():
                    # Convert Windows paths to Posix paths
                    normalized_path = file_path.replace('\\', '/')
                    full_path = workspace / normalized_path
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(content)
                    
                logger.info(f"Deployed {len(files)} files to {workspace}")
                
                # Load metadata to get workflow name
                workflow_name = workflow_id  # Default to ID
                metadata_path = workspace / "metadata.json"
                if metadata_path.exists():
                    # FAIL-FAST: Metadata must be valid JSON if it exists
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        workflow_name = metadata.get("workflow_name", workflow_id)
                        logger.info(f"Loaded workflow metadata: name={workflow_name}")
                
                # Log deployment status
                logger.info(f"Workflow {workflow_id} ({workflow_name}) deployed successfully")
                
                # Notify server with workflow name
                await self.websocket.send(json.dumps({
                    "type": "workflow_status",
                    "workflow_id": workflow_id,
                    "workflow_name": workflow_name,
                    "status": "deployed"
                }))
                
                # Auto-start if requested
                if run_after_deploy:
                    logger.info(f"Auto-starting workflow {workflow_id}")
                    await self.start_workflow(workflow_id, [])
                
            except Exception as e:
                logger.error(f"Deployment failed: {e}")
                await self.websocket.send(json.dumps({
                    "type": "workflow_status",
                    "workflow_id": workflow_id,
                    "workflow_name": workflow_name,  # Include workflow_name in failure message
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
            
        # Check conda environment before starting workflow (if specified in config)
        required_env = self.get('conda.conda_env', '')
        
        if required_env:  # Only check if a conda environment is specified
            conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
            
            if conda_env != required_env:
                error_msg = f"Cannot start workflow: {required_env} conda environment is not activated (current: {conda_env or 'None'})"
                logger.error(error_msg)
                
                # Send failure status to server
                await self.websocket.send(json.dumps({
                    "type": "workflow_status",
                    "workflow_id": workflow_id,
                    "status": "start_failed",
                    "details": error_msg
                }))
                return
            else:
                logger.info(f"✅ {required_env} conda environment is active")
            
        workspace = self.workspace_base / workflow_id
        runner_path = workspace / "runner.py"
        
        # Load workflow name from metadata if available
        workflow_name = workflow_id  # Default to ID
        metadata_path = workspace / "metadata.json"
        if metadata_path.exists():
            # FAIL-FAST: Metadata must be valid JSON if it exists
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                workflow_name = metadata.get("workflow_name", workflow_id)
        
        if not runner_path.exists():
            logger.error(f"Runner not found: {runner_path}")
            await self.websocket.send(json.dumps({
                "type": "workflow_status",
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
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
            
            # Store process info with workflow name
            self.workflows[workflow_id] = WorkflowProcess(workflow_id, process, workspace, workflow_name)
            
            # Start log reader
            self.workflows[workflow_id].log_reader_task = asyncio.create_task(
                self.read_process_logs(workflow_id)
            )
            
            logger.info(f"Started workflow {workflow_name} ({workflow_id}) (PID: {process.pid})")
            
            # Notify server with workflow name
            await self.websocket.send(json.dumps({
                "type": "workflow_status",
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "status": "running",
                "details": {"pid": process.pid}
            }))
            
        except Exception as e:
            logger.error(f"Failed to start workflow: {e}")
            await self.websocket.send(json.dumps({
                "type": "workflow_status",
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
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
            
            logger.info(f"Stopped workflow {workflow_id} ({workflow.workflow_name})")
            
            # Notify server
            await self.websocket.send(json.dumps({
                "type": "workflow_status",
                "workflow_id": workflow_id,
                "workflow_name": workflow.workflow_name,
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
            
            # Log the workflow completion status
            logger.info(f"Workflow {workflow_id} ({workflow.workflow_name}) {status} with exit code {exit_code}")
            
            if self.websocket:
                await self.websocket.send(json.dumps({
                    "type": "workflow_status",
                    "workflow_id": workflow_id,
                    "workflow_name": workflow.workflow_name,
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


def setup_logging(log_dir: Optional[str] = None):
    """
    Set up logging to both console and file
    
    Args:
        log_dir: Directory for log files (default: ./logs)
    """
    # Create logs directory
    if log_dir is None:
        log_dir = Path(__file__).parent / "logs"
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"dnne_agent_client_{timestamp}.log"
    
    # Set up formatters
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            # Console handler
            logging.StreamHandler(sys.stdout),
            # File handler
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    # Log the startup message
    logger.info(f"=== DNNE Agent Client Started ===")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    
    return log_file


async def main():
    """Main entry point"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='DNNE Agent Client - Executes workflows and forwards telemetry')
    parser.add_argument(
        '--server_ip',
        help='Server IP address (can be IP, IP:port, or "auto" for WSL auto-detection)',
        default=None
    )
    parser.add_argument(
        '--config',
        help='Path to dnne_config.json file',
        default=None
    )
    parser.add_argument(
        '--log-dir',
        help='Directory for log files (default: ./logs)',
        default=None
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_file = setup_logging(args.log_dir)
    
    # Create and run client
    client = DNNEAgentClient(config_path=args.config, server_ip=args.server_ip)
    
    try:
        await client.run()
    except KeyboardInterrupt:
        logger.info("Client shutdown requested")
    except Exception as e:
        logger.error(f"Client error: {e}")
        raise
    finally:
        logger.info("=== DNNE Agent Client Stopped ===")


if __name__ == "__main__":
    # Run the client
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Logger isn't available here since setup_logging() is called inside main()
        # This is fine - the KeyboardInterrupt inside main() will log properly
        pass
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)