#!/usr/bin/env python3
"""
Test script for dnne_agent_server - Windows side (passive listener).
Connects to dnne_agent_server as a UI client and displays activity.
"""

import asyncio
import websockets
import json
import sys
import time
import subprocess
import argparse
import signal
from pathlib import Path

# Add dnne-agent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "dnne-agent"))
from datetime import datetime

# Global flag for shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print("\nShutdown requested...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)


class TestDNNEAgentServer:
    """Test UI client for dnne_agent_server"""
    
    def __init__(self, server_url="ws://localhost:8767", verbose=False, quiet=False):
        self.server_url = server_url
        self.verbose = verbose
        self.quiet = quiet
        self.websocket = None
        self.start_time = time.time()
        self.telemetry_count = 0
        self.log_count = 0
        
    def log(self, message, level="info"):
        """Log message based on verbosity settings"""
        if self.quiet and level != "error":
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "error": "❌",
            "warning": "⚠️ ",
            "info": "ℹ️ ",
            "success": "✅",
            "telemetry": "📊",
            "log": "📝",
            "verbose": "🔍"
        }.get(level, "  ")
        
        if level == "verbose" and not self.verbose:
            return
            
        print(f"[{timestamp}] {prefix} {message}")
    
    async def ensure_dnne_agent_server_running(self):
        """Check if dnne_agent_server is running and start it if not"""
        self.log("Checking if dnne_agent_server is running...")
        
        # Check if dnne_agent_server is already running
        if await self._check_server_running():
            self.log("dnne_agent_server is already running", "success")
            return True
            
        # Not running, start it
        self.log("dnne_agent_server not found, starting it...")
        
        # Find dnne_agent_server.py
        server_path = Path(__file__).parent.parent.parent.parent / "dnne-agent" / "dnne_agent_server.py"
        if not server_path.exists():
            self.log(f"dnne_agent_server.py not found at {server_path}", "error")
            return False
            
        try:
            # Start dnne_agent_server
            if sys.platform == "win32":
                # Windows: Create new console window
                subprocess.Popen(
                    [sys.executable, str(server_path), "--enable-test-port"],
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                # Linux/Mac: Run in background
                subprocess.Popen(
                    [sys.executable, str(server_path), "--enable-test-port"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
            # Wait for server to start
            self.log("Waiting for dnne_agent_server to start...")
            await asyncio.sleep(3)
            
            # Check if it started successfully
            if await self._check_server_running():
                self.log("dnne_agent_server started successfully", "success")
                return True
            else:
                self.log("dnne_agent_server failed to start", "error")
                return False
                
        except Exception as e:
            self.log(f"Failed to start dnne_agent_server: {e}", "error")
            return False
    
    async def _check_server_running(self):
        """Check if dnne_agent_server is running by trying a WebSocket connection"""
        try:
            ws = await asyncio.wait_for(
                websockets.connect("ws://localhost:8767"),
                timeout=5.0
            )
            await ws.close()
            return True
        except asyncio.TimeoutError:
            self.log("Server check timed out", "verbose")
            return False
        except Exception as e:
            # Only log unexpected errors in verbose mode
            if "refused" not in str(e).lower() and "connection" not in str(e).lower():
                self.log(f"Server check failed: {type(e).__name__}: {e}", "verbose")
            return False
    
    async def connect(self):
        """Connect to dnne_agent_server"""
        try:
            self.log(f"Connecting to {self.server_url}...")
            self.websocket = await websockets.connect(self.server_url)
            self.log("Connected to dnne_agent_server", "success")
            return True
        except Exception as e:
            self.log(f"Failed to connect: {e}", "error")
            return False
    
    async def handle_message(self, data):
        """Handle messages from dnne_agent_server"""
        msg_type = data.get("type")
        
        if msg_type == "server_state":
            # Initial state - only show in verbose mode
            if self.verbose:
                clients = data.get("clients", {})
                workflows = data.get("workflows", {})
                uptime = data.get("server_uptime", 0)
                self.log(f"Initial server state: {len(clients)} clients, {len(workflows)} workflows, uptime: {uptime:.1f}s", "verbose")
                for client_id, info in clients.items():
                    self.log(f"  Client {client_id}: {info.get('hostname')} - Connected: {info.get('connected')}", "verbose")
                    
        elif msg_type == "client_connected":
            client_id = data.get("client_id")
            info = data.get("info", {})
            self.log(f"Client connected: {client_id} ({info.get('hostname')})", "success")
            
        elif msg_type == "client_disconnected":
            client_id = data.get("client_id")
            self.log(f"Client disconnected: {client_id}", "warning")
            
        elif msg_type == "workflow_deployed":
            workflow_id = data.get("workflow_id")
            client_id = data.get("client_id")
            self.log(f"Workflow deployed: {workflow_id} to {client_id}", "success")
            
        elif msg_type == "workflow_status":
            workflow_id = data.get("workflow_id")
            status = data.get("status")
            details = data.get("details", {})
            
            if status == "running":
                self.log(f"Workflow started: {workflow_id} (PID: {details.get('pid')})", "success")
            elif status == "completed":
                self.log(f"Workflow completed: {workflow_id}", "success")
            elif status == "failed":
                self.log(f"Workflow failed: {workflow_id} (exit code: {details.get('exit_code')})", "error")
            elif status == "stopped":
                self.log(f"Workflow stopped: {workflow_id}", "warning")
            else:
                self.log(f"Workflow {workflow_id}: {status}", "info")
                
        elif msg_type == "workflow_log":
            if not self.quiet:
                workflow_id = data.get("workflow_id")
                log = data.get("log", {})
                message = log.get("message", "").strip()
                if message:
                    self.log(f"[{workflow_id}] {message}", "log")
                    self.log_count += 1
                    
        elif msg_type == "telemetry_batch":
            metrics = data.get("metrics", {})
            if metrics:
                self.telemetry_count += len(metrics)
                
                if not self.quiet:
                    self.log(f"Telemetry batch: {len(metrics)} metrics from {len(set(metrics.keys()))} nodes", "telemetry")
                    
                    if self.verbose:
                        for node_id, node_metrics in metrics.items():
                            for metric in node_metrics[:3]:  # Show first 3
                                metric_type = metric.get("metric", metric.get("type"))
                                value = metric.get("value")
                                self.log(f"  {node_id}: {metric_type} = {value}", "verbose")
                                
        elif msg_type == "error":
            message = data.get("message")
            self.log(f"Server error: {message}", "error")
    
    async def run(self, timeout=None):
        """Main test loop"""
        if not self.websocket:
            return
            
        self.log("Test server ready and listening...")
        self.log("Waiting for test commands from dnne_client...")
        
        if timeout:
            self.log(f"Will exit after {timeout} seconds")
            
        try:
            start_time = time.time()
            
            while not shutdown_requested:
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    self.log("Timeout reached, exiting")
                    break
                    
                try:
                    # Receive with timeout
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=1.0
                    )
                    
                    data = json.loads(message)
                    await self.handle_message(data)
                    
                except asyncio.TimeoutError:
                    # No message, check if we should continue
                    continue
                except websockets.exceptions.ConnectionClosed:
                    self.log("Connection to server lost", "error")
                    break
                    
        except KeyboardInterrupt:
            self.log("Interrupted by user")
        finally:
            # Print summary
            runtime = time.time() - self.start_time
            self.log(f"\nTest Summary:", "info")
            self.log(f"  Runtime: {runtime:.1f}s", "info")
            self.log(f"  Telemetry received: {self.telemetry_count} metrics", "info")
            self.log(f"  Logs received: {self.log_count} lines", "info")
            
            if self.websocket:
                await self.websocket.close()


async def main():
    parser = argparse.ArgumentParser(description="Test UI client for dnne_agent_server")
    parser.add_argument("--server", default="localhost:8767",
                       help="dnne_agent_server UI endpoint (default: localhost:8767)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Minimal output")
    parser.add_argument("--timeout", type=int,
                       help="Exit after timeout seconds")
    parser.add_argument("--no-autostart", action="store_true",
                       help="Don't auto-start dnne_agent_server")
    
    args = parser.parse_args()
    
    # Build WebSocket URL
    if not args.server.startswith("ws://"):
        server_url = f"ws://{args.server}"
    else:
        server_url = args.server
        
    # Create test client
    test_client = TestDNNEAgentServer(server_url, args.verbose, args.quiet)
    
    # Ensure dnne_agent_server is running
    if not args.no_autostart:
        if not await test_client.ensure_dnne_agent_server_running():
            return 1
            
    # Connect to server
    if not await test_client.connect():
        return 1
        
    # Run test
    await test_client.run(args.timeout)
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)