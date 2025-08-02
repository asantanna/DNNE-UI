#!/usr/bin/env python3
"""
Test script for dnne_agent_client - Linux side (active controller).
Runs various tests against the dnne agent system.
"""

import asyncio
import websockets
import socket
import json
import sys
import time
import subprocess
import argparse
import signal
import os
import psutil
from pathlib import Path
from typing import Optional, List

# Add dnne-agent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "dnne-agent"))

def get_windows_host_ip():
    """Get Windows host IP address when running in WSL"""
    try:
        # Try to get the default gateway which is usually the Windows host in WSL
        result = subprocess.run(['ip', 'route'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'default' in line:
                    # Extract IP from line like: "default via 172.22.160.1 dev eth0"
                    parts = line.split()
                    if len(parts) >= 3 and parts[0] == 'default' and parts[1] == 'via':
                        return parts[2]
    except Exception:
        pass
    
    # Fall back to hardcoded IP if detection fails
    return "172.22.160.1"

# Exit codes
EXIT_SUCCESS = 0
EXIT_CONNECTION_FAILED = 1
EXIT_AGENT_ERROR = 2
EXIT_DEPLOYMENT_FAILED = 3
EXIT_WORKFLOW_ERROR = 4
EXIT_TIMEOUT = 5
EXIT_TEST_FAILED = 6


class TestDNNEAgentClient:
    """Test controller for dnne_client"""
    
    def __init__(self, server_host="localhost", server_port=8766):
        self.server_host = server_host
        self.server_port = server_port
        self.server_url = f"ws://{server_host}:{server_port}"
        self.ui_port = 8767  # UI port is always one higher than client port
        self.client_process = None
        self.test_dir = Path(__file__).parent / "tests"
        
    def log(self, message, prefix=""):
        """Simple logging"""
        timestamp = time.strftime("%H:%M:%S")
        if prefix:
            print(f"[{timestamp}] {prefix} {message}")
        else:
            print(f"[{timestamp}] {message}")
    
    def find_dnne_client_process(self):
        """Find running dnne_client process"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and 'dnne_client.py' in ' '.join(cmdline):
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def ensure_dnne_client_running(self):
        """Start dnne_client if not running"""
        # Check if already running
        proc = self.find_dnne_client_process()
        if proc:
            self.log(f"dnne_client already running (PID: {proc.pid})", "✓")
            return True
            
        # Start dnne_client
        self.log("Starting dnne_client...")
        
        client_path = Path(__file__).parent / "dnne_client.py"
        if not client_path.exists():
            self.log(f"dnne_client.py not found at {client_path}", "✗")
            return False
            
        try:
            # Set server URL via environment
            env = os.environ.copy()
            env['DNNE_SERVER_URL'] = self.server_url
            
            self.client_process = subprocess.Popen(
                [sys.executable, str(client_path)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            
            # Wait for startup
            time.sleep(3)
            
            # Verify it's running
            if self.client_process.poll() is None:
                self.log(f"dnne_client started (PID: {self.client_process.pid})", "✓")
                return True
            else:
                self.log("dnne_client failed to start", "✗")
                return False
                
        except Exception as e:
            self.log(f"Failed to start dnne_client: {e}", "✗")
            return False
    
    def stop_dnne_client(self):
        """Stop dnne_client gracefully"""
        proc = self.find_dnne_client_process()
        if not proc:
            self.log("dnne_client not running", "✓")
            return True
            
        try:
            self.log(f"Stopping dnne_client (PID: {proc.pid})...")
            proc.terminate()
            
            # Wait up to 5 seconds
            try:
                proc.wait(timeout=5)
                self.log("dnne_client stopped", "✓")
                return True
            except psutil.TimeoutExpired:
                # Force kill
                proc.kill()
                proc.wait()
                self.log("dnne_client killed", "⚠")
                return True
                
        except Exception as e:
            self.log(f"Failed to stop dnne_client: {e}", "✗")
            return False
    
    async def test_connectivity(self):
        """Test basic connectivity to dnne_server via dnne_client"""
        self.log("Testing connectivity...")
        
        # dnne_client should be connected to dnne_server
        # We'll verify by checking if dnne_server has clients
        try:
            # Connect directly to dnne_server UI port to check
            ui_url = f"ws://{self.server_host}:{self.ui_port}"
            ws = await websockets.connect(ui_url)
            
            # Wait for server state
            message = await ws.recv()
            data = json.loads(message)
            
            if data.get("type") == "server_state":
                clients = data.get("clients", {})
                if clients:
                    self.log(f"Connectivity OK: {len(clients)} client(s) connected", "✓")
                    await ws.close()
                    return True
                else:
                    self.log("No clients connected to dnne_server", "✗")
                    await ws.close()
                    return False
                    
        except Exception as e:
            self.log(f"Connectivity test failed: {e}", "✗")
            return False
    
    def send_telemetry(self, metric_type, value, node_id="test_node", 
                      host="localhost", port=9999):
        """Send a single telemetry packet"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            packet = f"{metric_type}|{node_id}|{value}|{time.time()}"
            sock.sendto(packet.encode(), (host, port))
            sock.close()
            self.log(f"Sent {metric_type}={value}", "→")
            return True
        except Exception as e:
            self.log(f"Failed to send telemetry: {e}", "✗")
            return False
    
    def send_telemetry_burst(self, count, host="localhost", port=9999):
        """Send burst of telemetry packets"""
        self.log(f"Sending burst of {count} packets...")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        
        success = 0
        start_time = time.time()
        
        for i in range(count):
            try:
                node_id = f"burst_node_{i % 10}"
                value = 100 + (i % 50)
                packet = f"throughput|{node_id}|{value}|{time.time()}"
                sock.sendto(packet.encode(), (host, port))
                success += 1
                
                # Brief pause to avoid overwhelming
                if i % 100 == 99:
                    time.sleep(0.01)
                    
            except Exception as e:
                self.log(f"Burst error at packet {i}: {e}", "✗")
                
        duration = time.time() - start_time
        rate = success / duration if duration > 0 else 0
        
        sock.close()
        self.log(f"Sent {success}/{count} packets in {duration:.2f}s ({rate:.0f} pkt/s)", "✓")
        return success == count
    
    async def deploy_and_run_test(self, test_name, wait_complete=True):
        """Deploy and run a test workflow"""
        test_file = self.test_dir / f"{test_name}.py"
        if not test_file.exists():
            self.log(f"Test file not found: {test_file}", "✗")
            return False
            
        self.log(f"Deploying test: {test_name}")
        
        try:
            # Connect to dnne_server UI port
            ui_url = f"ws://{self.server_host}:{self.ui_port}"
            ws = await websockets.connect(ui_url)
            
            # Wait for initial state
            await ws.recv()
            
            # Read test file
            content = test_file.read_text()
            
            # Deploy workflow
            await ws.send(json.dumps({
                "type": "deploy_workflow",
                "files": {
                    "runner.py": content
                }
            }))
            
            # Wait for deployment confirmation
            while True:
                message = await ws.recv()
                data = json.loads(message)
                
                if data.get("type") == "workflow_deployed":
                    workflow_id = data.get("workflow_id")
                    self.log(f"Deployed: {workflow_id}", "✓")
                    
                    # Start workflow
                    await ws.send(json.dumps({
                        "type": "start_workflow",
                        "workflow_id": workflow_id
                    }))
                    
                    # Wait for completion if requested
                    if wait_complete:
                        return await self.wait_for_completion(ws, workflow_id)
                    else:
                        await ws.close()
                        return True
                        
                elif data.get("type") == "error":
                    self.log(f"Deployment error: {data.get('message')}", "✗")
                    await ws.close()
                    return False
                    
        except Exception as e:
            self.log(f"Deploy/run failed: {e}", "✗")
            return False
    
    async def wait_for_completion(self, ws, workflow_id, timeout=60):
        """Wait for workflow to complete"""
        self.log(f"Waiting for workflow completion (timeout: {timeout}s)...")
        
        start_time = time.time()
        
        try:
            while (time.time() - start_time) < timeout:
                message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                data = json.loads(message)
                
                if data.get("type") == "workflow_status":
                    if data.get("workflow_id") == workflow_id:
                        status = data.get("status")
                        
                        if status == "completed":
                            self.log("Workflow completed successfully", "✓")
                            await ws.close()
                            return True
                        elif status == "failed":
                            self.log(f"Workflow failed: {data.get('details')}", "✗")
                            await ws.close()
                            return False
                        elif status == "stopped":
                            self.log("Workflow stopped", "⚠")
                            await ws.close()
                            return True
                            
        except asyncio.TimeoutError:
            pass
            
        self.log("Workflow completion timeout", "✗")
        await ws.close()
        return False
    
    async def stop_workflow(self):
        """Stop running workflow"""
        self.log("Stopping workflow...")
        
        try:
            ui_url = f"ws://{self.server_host}:{self.ui_port}"
            ws = await websockets.connect(ui_url)
            await ws.recv()  # Initial state
            
            # Send stop command (stops active workflow)
            await ws.send(json.dumps({
                "type": "stop_workflow"
            }))
            
            # Wait for confirmation
            timeout = 10
            start_time = time.time()
            
            while (time.time() - start_time) < timeout:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(message)
                    
                    if data.get("type") == "workflow_status" and data.get("status") == "stopped":
                        self.log("Workflow stopped", "✓")
                        await ws.close()
                        return True
                        
                except asyncio.TimeoutError:
                    continue
                    
            self.log("Stop workflow timeout", "✗")
            await ws.close()
            return False
            
        except Exception as e:
            self.log(f"Failed to stop workflow: {e}", "✗")
            return False


async def main():
    parser = argparse.ArgumentParser(description="Test controller for dnne_client")
    
    # Connection options
    default_host = get_windows_host_ip()
    parser.add_argument("--server", default=f"{default_host}:8766",
                       help=f"dnne_server endpoint (default: {default_host}:8766 for WSL)")
    parser.add_argument("--server-host", default=default_host,
                       help=f"dnne_server host (default: {default_host} for WSL)")
    parser.add_argument("--client", default="localhost:9999",
                       help="dnne_client UDP endpoint (default: localhost:9999)")
    parser.add_argument("--no-autostart", action="store_true",
                       help="Don't auto-start dnne_client")
    
    # Basic operations
    parser.add_argument("--test-connectivity", action="store_true",
                       help="Test basic connectivity")
    parser.add_argument("--stop-agent", action="store_true",
                       help="Stop dnne_client")
    parser.add_argument("--ensure-agent", action="store_true",
                       help="Ensure dnne_client is running")
    
    # Telemetry tests
    parser.add_argument("--send-metric", metavar="TYPE:VALUE",
                       help="Send single metric (e.g., throughput:100)")
    parser.add_argument("--send-burst", type=int, metavar="COUNT",
                       help="Send burst of COUNT packets")
    parser.add_argument("--send-continuous", action="store_true",
                       help="Send continuous metrics")
    
    # Workflow tests
    parser.add_argument("--run", metavar="TEST_NAME",
                       help="Run predefined test")
    parser.add_argument("--deploy", metavar="PATH",
                       help="Deploy custom workflow")
    parser.add_argument("--start", action="store_true",
                       help="Start deployed workflow")
    parser.add_argument("--stop-workflow", action="store_true",
                       help="Stop running workflow")
    parser.add_argument("--wait-complete", action="store_true",
                       help="Wait for workflow completion")
    
    # Automated test suite options
    parser.add_argument("--run-tests", action="store_true",
                       help="Run automated test suite")
    parser.add_argument("--test-basic", action="store_true",
                       help="Run basic connectivity test")
    parser.add_argument("--test-deploy", action="store_true",
                       help="Run deployment test")
    parser.add_argument("--test-workflow", action="store_true",
                       help="Run workflow execution test")
    parser.add_argument("--test-telemetry", action="store_true",
                       help="Run telemetry test")
    
    # Other options
    parser.add_argument("--duration", type=int, default=10,
                       help="Test duration in seconds (default: 10)")
    parser.add_argument("--timeout", type=int, default=60,
                       help="Overall timeout in seconds (default: 60)")
    
    args = parser.parse_args()
    
    # Parse server endpoint
    if args.server_host:
        host = args.server_host
        port = 8766
    elif ':' in args.server:
        host, port = args.server.split(':')
        port = int(port)
    else:
        host = args.server
        port = 8766
        
    # Parse client endpoint
    if ':' in args.client:
        client_host, client_port = args.client.split(':')
        client_port = int(client_port)
    else:
        client_host = args.client
        client_port = 9999
        
    # Create test controller
    test = TestDNNEAgentClient(host, port)
    
    # Ensure agent unless --no-autostart or --stop-agent
    if not args.no_autostart and not args.stop_agent and not args.ensure_agent:
        if not test.ensure_dnne_client_running():
            return EXIT_AGENT_ERROR
            
    # Execute requested operations
    exit_code = EXIT_SUCCESS
    
    try:
        # Basic operations
        if args.ensure_agent:
            if not test.ensure_dnne_client_running():
                return EXIT_AGENT_ERROR
                
        if args.test_connectivity:
            if not await test.test_connectivity():
                return EXIT_CONNECTION_FAILED
                
        if args.stop_agent:
            if not test.stop_dnne_client():
                return EXIT_AGENT_ERROR
            return EXIT_SUCCESS
            
        # Telemetry operations
        if args.send_metric:
            try:
                metric_type, value = args.send_metric.split(':')
                value = float(value)
                if not test.send_telemetry(metric_type, value, host=client_host, port=client_port):
                    exit_code = EXIT_AGENT_ERROR
            except ValueError:
                test.log("Invalid metric format. Use TYPE:VALUE", "✗")
                return EXIT_AGENT_ERROR
                
        if args.send_burst:
            if not test.send_telemetry_burst(args.send_burst, host=client_host, port=client_port):
                exit_code = EXIT_AGENT_ERROR
                
        if args.send_continuous:
            test.log(f"Sending continuous metrics for {args.duration}s...")
            start_time = time.time()
            
            while (time.time() - start_time) < args.duration:
                test.send_telemetry("throughput", 100 + (time.time() % 50),
                                  host=client_host, port=client_port)
                time.sleep(0.1)
                
        # Automated test suite
        if args.run_tests:
            test.log("Running automated test suite...", "🧪")
            test_passed = True
            
            # Run all requested tests
            if args.test_basic:
                test.log("Running basic connectivity test...", "🔍")
                if not args.test_connectivity:
                    args.test_connectivity = True
                    
            if args.test_deploy:
                test.log("Running deployment test...", "📦")
                if not await test.deploy_and_run_test("hello_world", wait_complete=True):
                    test_passed = False
                    
            if args.test_workflow:
                test.log("Running workflow execution test...", "⚙️")
                if not await test.deploy_and_run_test("long_running", wait_complete=False):
                    test_passed = False
                await asyncio.sleep(2)  # Let it run
                if not await test.stop_workflow():
                    test_passed = False
                    
            if args.test_telemetry:
                test.log("Running telemetry test...", "📊")
                if not test.send_telemetry_burst(10, host=client_host, port=client_port):
                    test_passed = False
                    
            if not test_passed:
                return EXIT_TEST_FAILED
                
        # Workflow operations
        if args.run:
            if not await test.deploy_and_run_test(args.run, args.wait_complete):
                return EXIT_WORKFLOW_ERROR
                
        if args.stop_workflow:
            if not await test.stop_workflow():
                return EXIT_WORKFLOW_ERROR
                
    except KeyboardInterrupt:
        test.log("Interrupted by user", "⚠")
        exit_code = EXIT_TIMEOUT
    except Exception as e:
        test.log(f"Unexpected error: {e}", "✗")
        exit_code = EXIT_AGENT_ERROR
        
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)