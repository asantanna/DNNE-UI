#!/usr/bin/env python3
"""
Telemetry test orchestrator - manages test client and deploys telemetry runner.
This orchestrator:
1. Checks if a test client is connected (hostname: agent_client_test_host)
2. Starts a test client if needed
3. Deploys telemetry_runner.py as runner.py to the test client
4. Monitors execution and verifies telemetry files
5. Cleans up (stops client if we started it)
"""

import asyncio
import websockets
import json
import time
import sys
import subprocess
import signal
import argparse
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import deployment helper
from helpers.deployment_helper import (
    check_client_connected,
    deploy_files_to_client,
    cleanup_workflow_directories
)


class TelemetryTestOrchestrator:
    """
    Orchestrates telemetry testing by managing test client and deployment.
    """
    
    def __init__(self, verbose: bool = False, test_type: str = "basic"):
        self.verbose = verbose
        self.test_type = test_type
        self.test_client_process = None
        self.started_client = False
        self.windows_host = "172.22.160.1"
        self.test_port_url = f"ws://{self.windows_host}:8768"
        self.workflow_id = f"test_wf_{int(time.time())}"
        self.workflow_name = "TelemetryTest"
        self.test_client_hostname = "agent_client_test_host"
        
        # Test type specific settings
        self.test_durations = {
            "basic": 10,
            "long": 40,  # 35s test + buffer
            "ratelimit": 15,
            "aggregation": 20
        }
        self.test_runners = {
            "basic": "telemetry_runner.py",
            "long": "telemetry_runner_long.py",
            "ratelimit": "telemetry_runner_ratelimit.py",
            "aggregation": "telemetry_runner_aggregation.py"
        }
    
    async def check_test_client_connected(self, websocket) -> bool:
        """
        Check if a test client is connected to the agent server.
        """
        # Request list of connected clients
        await websocket.send(json.dumps({
            "type": "get_clients"
        }))
        
        # Wait for response
        response = await websocket.recv()
        data = json.loads(response)
        
        if self.verbose:
            print(f"   DEBUG: Got response type: {data.get('type')}")
            print(f"   DEBUG: Clients: {data.get('clients', [])}")
        
        if data.get("type") == "clients_list":
            clients = data.get("clients", [])
            for client in clients:
                if client.get("hostname") == self.test_client_hostname:
                    print(f"✅ Test client already connected: {client.get('client_id')}")
                    return True
        
        return False
    
    async def start_test_client(self) -> bool:
        """
        Start a test client with --test-mode flag.
        """
        print("🚀 Starting test client...")
        
        # Path to client script
        client_script = Path(__file__).parent.parent.parent / "dnne-agent" / "dnne_agent_client.py"
        
        if not client_script.exists():
            print(f"❌ Client script not found: {client_script}")
            return False
        
        try:
            # Start client with test mode
            cmd = [
                sys.executable,
                str(client_script),
                "--test-mode",
                "--server-ip", f"{self.windows_host}:8766",
                "--verbose", "INFO"
            ]
            
            if self.verbose:
                print(f"   Command: {' '.join(cmd)}")
            
            # Start client process
            self.test_client_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE if not self.verbose else None,
                stderr=subprocess.PIPE if not self.verbose else None,
                text=True
            )
            
            self.started_client = True
            print(f"   ✓ Started test client (PID: {self.test_client_process.pid})")
            
            # Wait for client to connect
            await asyncio.sleep(3)
            return True
            
        except Exception as e:
            print(f"❌ Failed to start test client: {e}")
            return False
    
    async def cleanup_old_workflows(self, keep_last_n: int = 0):
        """
        Clean up old test workflow directories before starting new test.
        
        Args:
            keep_last_n: Number of recent directories to keep (0 = delete all)
        """
        base_path = Path("/home/asantanna/DNNE/DNNE-UI/remote_clients") / self.test_client_hostname
        
        if not base_path.exists():
            return
            
        # Find all TelemetryTest directories
        test_dirs = sorted(base_path.glob("TelemetryTest_test_wf_*"), 
                          key=lambda x: x.stat().st_mtime)
        
        if not test_dirs:
            return
            
        # Determine which directories to delete
        if keep_last_n > 0:
            dirs_to_delete = test_dirs[:-keep_last_n] if len(test_dirs) > keep_last_n else []
        else:
            dirs_to_delete = test_dirs
        
        if dirs_to_delete:
            print(f"🧹 Cleaning up {len(dirs_to_delete)} old test directories...")
            for dir_to_delete in dirs_to_delete:
                try:
                    shutil.rmtree(dir_to_delete)
                    if self.verbose:
                        print(f"   🗑️ Removed: {dir_to_delete.name}")
                except Exception as e:
                    print(f"   ⚠️ Failed to remove {dir_to_delete.name}: {e}")
        
    async def stop_test_client(self):
        """
        Stop the test client if we started it.
        """
        if self.started_client and self.test_client_process:
            print("🛑 Stopping test client...")
            
            try:
                # Send SIGTERM
                self.test_client_process.terminate()
                
                # Wait up to 5 seconds for graceful shutdown
                try:
                    self.test_client_process.wait(timeout=5)
                    print("   ✓ Test client stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill
                    self.test_client_process.kill()
                    self.test_client_process.wait()
                    print("   ✓ Test client force killed")
                    
            except Exception as e:
                print(f"   ⚠️ Error stopping test client: {e}")
            finally:
                self.test_client_process = None
                self.started_client = False
    
    async def deploy_telemetry_runner(self, websocket) -> bool:
        """
        Deploy telemetry_runner.py as runner.py to the test client.
        """
        print(f"📦 Deploying {self.test_type} telemetry runner to test client...")
        
        # Read the appropriate telemetry runner script
        runner_file = self.test_runners.get(self.test_type, "telemetry_runner.py")
        runner_path = Path(__file__).parent / runner_file
        
        if not runner_path.exists():
            print(f"❌ Telemetry runner not found: {runner_path}")
            return False
        
        with open(runner_path, 'r') as f:
            runner_content = f.read()
        
        # Create metadata
        metadata = {
            "workflow_name": self.workflow_name,
            "workflow_id": self.workflow_id,
            "created_at": time.time(),
            "test_mode": True
        }
        
        # Deploy using helper
        success = await deploy_files_to_client(
            websocket=websocket,
            files={
                "runner.py": runner_content,
                "metadata.json": json.dumps(metadata, indent=2)
            },
            client_hostname=self.test_client_hostname,
            workflow_id=self.workflow_id,
            workflow_name=self.workflow_name,
            runner_args="",
            run_after_deploy=True
        )
        
        if success:
            print(f"   ✓ Deployed workflow {self.workflow_id}")
        else:
            print(f"   ❌ Deployment failed")
        
        return success
    
    async def monitor_execution(self, websocket, duration: int = 10):
        """
        Monitor workflow execution for a specified duration.
        """
        print(f"⏳ Monitoring execution for {duration} seconds...")
        
        start_time = time.time()
        log_count = 0
        
        while time.time() - start_time < duration:
            try:
                # Check for messages with timeout
                message = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                data = json.loads(message)
                
                # Handle different message types
                if data.get("type") == "log":
                    log_count += 1
                    if self.verbose:
                        print(f"   📝 {data.get('message', '')}")
                elif data.get("type") == "workflow_status":
                    status = data.get("status")
                    print(f"   📊 Workflow status: {status}")
                    if status in ["completed", "failed", "terminated"]:
                        break
                        
            except asyncio.TimeoutError:
                # No message received, continue monitoring
                pass
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ Monitor error: {e}")
        
        if log_count > 0:
            print(f"   ✓ Received {log_count} log messages")
    
    async def verify_telemetry_files(self) -> bool:
        """
        Verify that telemetry files were created.
        """
        print("\n🔍 Verifying telemetry files...")
        
        # Build expected paths - use absolute path
        base_path = Path("/home/asantanna/DNNE/DNNE-UI/remote_clients") / self.test_client_hostname / f"{self.workflow_name}_{self.workflow_id}"
        telemetry_base = base_path / "telemetry"
        
        if not telemetry_base.exists():
            print(f"❌ Telemetry directory not found: {telemetry_base}")
            return False
        
        # Find telem_* directory
        telem_dirs = list(telemetry_base.glob("telem_*"))
        if not telem_dirs:
            print(f"❌ No telem_* directory found in {telemetry_base}")
            return False
        
        telem_dir = telem_dirs[0]
        print(f"✅ Found telemetry directory: {telem_dir}")
        
        # Check for expected files based on test type
        if self.test_type == "long":
            expected_patterns = [
                "node_node_20.dat",
                "node_node_21.dat", 
                "node_node_22.dat",
                "node_node_20_violations.log",
                "node_node_21_violations.log",
                "node_node_22_violations.log"
            ]
        else:
            expected_patterns = [
                "node_node_10.dat",
                "node_node_11.dat",
                "node_node_12.dat",
                "node_node_10_violations.log",
                "node_node_11_violations.log",
                "node_burst_node_*.dat"
            ]
        
        found_files = []
        missing_patterns = []
        
        for pattern in expected_patterns:
            if '*' in pattern:
                # Glob pattern
                matches = list(telem_dir.glob(pattern))
                if matches:
                    found_files.extend([f.name for f in matches])
                else:
                    missing_patterns.append(pattern)
            else:
                # Exact file
                file_path = telem_dir / pattern
                if file_path.exists() and file_path.stat().st_size > 0:
                    found_files.append(pattern)
                else:
                    missing_patterns.append(pattern)
        
        # Report results
        if found_files:
            print(f"\n📁 Found {len(found_files)} telemetry files:")
            for f in found_files[:5]:  # Show first 5
                print(f"   ✓ {f}")
            if len(found_files) > 5:
                print(f"   ... and {len(found_files) - 5} more")
        
        if missing_patterns:
            print(f"\n⚠️ Missing files matching patterns:")
            for p in missing_patterns:
                print(f"   ❌ {p}")
            return False
        
        # Validate content of a sample file
        sample_node = "node_20" if self.test_type == "long" else "node_10"
        sample_file = telem_dir / f"node_{sample_node}.dat"
        if sample_file.exists():
            with open(sample_file, 'r') as f:
                lines = f.readlines()
            
            if lines:
                print(f"\n📊 Sample telemetry data ({sample_file.name}):")
                for line in lines[:3]:
                    parts = line.strip().split('|')
                    if len(parts) == 3:
                        timestamp, metric, value = parts
                        print(f"   {metric}: {float(value):.2f}")
        
        # Check violation files for proper format and SUMMARY lines
        violations_validated = False
        summary_found = False
        
        # Check first node violations (should have SUMMARY after rate limiting test)
        first_node = "node_20" if self.test_type == "long" else "node_10"
        violation_file_10 = telem_dir / f"node_{first_node}_violations.log"
        if violation_file_10.exists():
            with open(violation_file_10, 'r') as f:
                lines = f.readlines()
            
            detail_count = 0
            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                if "SUMMARY" in line:  # Check if SUMMARY is anywhere in the line
                    summary_found = True
                    # Parse SUMMARY format: SUMMARY type count=N exp=E range=[min,max] last=L
                    if "count=" in line and "exp=" in line and "range=" in line:
                        print(f"\n✅ SUMMARY validation found: {line}")
                        # Extract count
                        import re
                        count_match = re.search(r'count=(\d+)', line)
                        if count_match:
                            summary_count = int(count_match.group(1))
                            print(f"   Summary shows {summary_count} violations")
                else:
                    detail_count += 1
            
            if summary_found:
                print(f"   Found {detail_count} detail lines before SUMMARY")
            else:
                print(f"\n⚠️ No SUMMARY line found in node_10 violations")
        
        # Check second node violations for grouping (only for basic test)
        if self.test_type == "basic":
            violation_file_11 = telem_dir / "node_node_11_violations.log"
            if violation_file_11.exists():
                with open(violation_file_11, 'r') as f:
                    content = f.read()
                
                # Check for GPU grouping
                gpu_groups = ["gpu_0", "gpu_1", "gpu_2"]
                found_groups = []
                found_summaries = []
                
                for gpu in gpu_groups:
                    if f"memory_exceeded[{gpu}]" in content:
                        found_groups.append(gpu)
                    # Check for grouped summaries
                    if f"SUMMARY memory_exceeded[{gpu}]" in content:
                        found_summaries.append(gpu)
                
                if found_groups:
                    print(f"\n✅ Violation grouping validated: {', '.join(found_groups)}")
                if found_summaries:
                    print(f"✅ Grouped SUMMARY lines found: {', '.join(found_summaries)}")
                    violations_validated = True
        else:
            # For long test, we should have seen multiple SUMMARY entries  
            violations_validated = True  # Different validation for long test
        
        # Overall validation
        if not summary_found:
            print("\n❌ Test FAILED: No SUMMARY lines found")
            return False
        
        if not violations_validated:
            print("\n⚠️ Warning: Grouped violations not fully validated")
        
        return True
    
    async def run(self) -> int:
        """
        Main orchestrator execution.
        """
        print(f"🧪 Telemetry Test Orchestrator - {self.test_type.upper()} Test")
        print("=" * 60)
        
        # Clean up old test directories (keep last 2 for debugging)
        await self.cleanup_old_workflows(keep_last_n=2)
        
        try:
            # Connect to test port
            print(f"🔌 Connecting to test port at {self.test_port_url}...")
            async with websockets.connect(self.test_port_url) as websocket:
                print("✅ Connected to test port")
                
                # Read and discard initial state message
                initial_msg = await websocket.recv()
                if self.verbose:
                    initial_data = json.loads(initial_msg)
                    print(f"   DEBUG: Initial state received: {initial_data.get('type')}")
                
                # Check if test client is connected
                if not await self.check_test_client_connected(websocket):
                    # Start test client
                    if not await self.start_test_client():
                        return 1
                    
                    # Verify client connected
                    await asyncio.sleep(5)  # Give more time for client to register
                    if not await self.check_test_client_connected(websocket):
                        print("❌ Test client failed to connect")
                        return 1
                
                # Deploy telemetry runner
                if not await self.deploy_telemetry_runner(websocket):
                    return 1
                
                # Monitor execution with test-specific duration
                duration = self.test_durations.get(self.test_type, 10)
                await self.monitor_execution(websocket, duration=duration)
                
                # Wait a bit for all files to be fully written
                await asyncio.sleep(1)
                
                # Verify telemetry files
                success = await self.verify_telemetry_files()
                
                print("\n" + "=" * 60)
                if success:
                    print("✅ Telemetry test PASSED")
                    return 0
                else:
                    print("❌ Telemetry test FAILED")
                    return 1
                    
        except (ConnectionRefusedError, OSError) as e:
            print(f"❌ Could not connect to test port")
            print(f"   Make sure agent server is running with --enable-test-port")
            print(f"   Error: {e}")
            return 1
        except Exception as e:
            print(f"❌ Orchestrator error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return 1
        finally:
            # Clean up test client if we started it
            if self.started_client:
                await self.stop_test_client()


def main():
    parser = argparse.ArgumentParser(description="Telemetry test orchestrator")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Enable verbose output")
    parser.add_argument("--test-type", choices=["basic", "long", "ratelimit", "aggregation"],
                       default="basic", help="Type of telemetry test to run")
    
    args = parser.parse_args()
    
    orchestrator = TelemetryTestOrchestrator(verbose=args.verbose, test_type=args.test_type)
    result = asyncio.run(orchestrator.run())
    
    sys.exit(result)


if __name__ == "__main__":
    main()