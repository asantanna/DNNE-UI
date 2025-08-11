#!/usr/bin/env python3
"""
Deployment Helper for DNNE Test Suite

Provides reusable functions for deploying workflows and files to test clients,
hiding the complexity of export, file management, and WebSocket communication.
"""

import json
import time
import asyncio
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import websockets


class TestClientManager:
    """Manages test client lifecycle for testing"""
    
    def __init__(self, windows_host: str = "172.22.160.1", verbose: bool = False):
        self.windows_host = windows_host
        self.verbose = verbose
        self.test_client_process = None
        self.started_client = False
    
    async def start_test_client(self) -> bool:
        """
        Start a test client with --test-mode flag.
        
        Returns:
            True if client started successfully, False otherwise
        """
        print("🚀 Starting test client...")
        
        # Path to client script
        client_script = Path(__file__).parent.parent.parent.parent / "dnne-agent" / "dnne_agent_client.py"
        
        if not client_script.exists():
            print(f"❌ Client script not found: {client_script}")
            return False
        
        try:
            # Start client with test mode using DNNE_PY38 environment
            # Build command with conda activation
            conda_activate = "source /home/asantanna/miniconda/bin/activate DNNE_PY38"
            python_cmd = f"python {str(client_script)} --test-mode --server-ip {self.windows_host}:8766 --verbose {'DEBUG' if self.verbose else 'INFO'}"
            
            # Combine into a single bash command
            full_cmd = f"{conda_activate} && {python_cmd}"
            
            if self.verbose:
                print(f"   Command: {full_cmd}")
            
            # Start client process using bash to handle conda activation
            self.test_client_process = subprocess.Popen(
                ["bash", "-c", full_cmd],
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
    
    async def stop_test_client(self):
        """Stop the test client if we started it."""
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
    
    async def ensure_test_client_connected(self, websocket, test_hostname: str = "agent_client_test_host") -> bool:
        """
        Ensure a test client is connected, starting one if necessary.
        
        Args:
            websocket: Active WebSocket connection to test port
            test_hostname: Expected hostname of test client (default: agent_client_test_host)
            
        Returns:
            True if client is connected (either already or after starting), False otherwise
        """
        # Check if already connected
        if await DeploymentHelper.check_client_connected(websocket, test_hostname):
            print(f"✅ Test client already connected")
            return True
        
        # Start test client
        if not await self.start_test_client():
            return False
        
        # Wait for registration and verify
        await asyncio.sleep(5)
        if await DeploymentHelper.check_client_connected(websocket, test_hostname):
            print(f"✅ Test client connected after startup")
            return True
        
        print("❌ Test client failed to connect")
        return False
    
    async def __aenter__(self):
        """Support async context manager for automatic cleanup."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up test client on exit."""
        await self.stop_test_client()


class DeploymentHelper:
    """Helper class for deploying workflows to test clients"""
    
    @staticmethod
    async def check_client_connected(websocket, hostname: str) -> bool:
        """
        Check if a test client with the specified hostname is connected.
        
        Args:
            websocket: Active WebSocket connection to test port
            hostname: Hostname of the client to check
            
        Returns:
            True if client is connected, False otherwise
        """
        await websocket.send(json.dumps({
            "type": "get_clients"
        }))
        
        response = await websocket.recv()
        data = json.loads(response)
        
        if data.get("type") == "clients_list":
            clients = data.get("clients", [])
            for client in clients:
                if client.get("hostname") == hostname and client.get("connected"):
                    return True
        return False
    
    @staticmethod
    async def deploy_files_to_client(
        websocket,
        files: Dict[str, str],
        client_hostname: str,
        workflow_id: str,
        workflow_name: str,
        runner_args: str = "",
        run_after_deploy: bool = True,
        wait_for_confirmation: bool = True
    ) -> bool:
        """
        Deploy arbitrary files to a test client.
        
        Args:
            websocket: Active WebSocket connection to test port
            files: Dictionary of relative_path -> file_content
            client_hostname: Hostname of the target client
            workflow_id: Unique workflow identifier
            workflow_name: Human-readable workflow name
            runner_args: Arguments to pass to runner.py
            run_after_deploy: Whether to start execution after deployment
            wait_for_confirmation: Whether to wait for deployment confirmation
            
        Returns:
            True if deployment successful, False otherwise
        """
        # Send deployment message
        await websocket.send(json.dumps({
            "type": "deploy_to_client",
            "client_hostname": client_hostname,
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "files": files,
            "run_after_deploy": run_after_deploy,
            "runner_args": runner_args
        }))
        
        if not wait_for_confirmation:
            return True
        
        # Wait for deployment confirmation
        try:
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                data = json.loads(response)
                
                if data.get("type") == "deploy_success":
                    return True
                elif data.get("type") == "deploy_failed":
                    print(f"Deployment failed: {data.get('error')}")
                    return False
                    
        except asyncio.TimeoutError:
            print("Deployment confirmation timed out")
            return False
    
    @staticmethod
    def export_workflow_locally(workflow_name: str) -> Optional[Dict[str, str]]:
        """
        Export a workflow using programmatic_export.py and read the files.
        
        Args:
            workflow_name: Name of the workflow to export (without .json)
            
        Returns:
            Dictionary of relative_path -> file_content, or None if export failed
        """
        # Use programmatic_export.py to export the workflow
        export_script = Path("/home/asantanna/DNNE/DNNE-UI/dnne-test-suite/utilities/programmatic_export.py")
        target_dir = f"{workflow_name.lower().replace(' ', '_')}_temp_export"
        
        try:
            # Run the export with --add-metadata flag
            result = subprocess.run(
                [sys.executable, str(export_script), workflow_name, "--target-dir", target_dir, "--add-metadata"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd="/home/asantanna/DNNE/DNNE-UI"
            )
            
            if result.returncode != 0:
                print(f"Export failed: {result.stderr}")
                return None
            
            # Read exported files
            export_path = Path("/home/asantanna/DNNE/DNNE-UI/export_system/exports") / target_dir
            if not export_path.exists():
                print(f"Export directory not found: {export_path}")
                return None
            
            files = {}
            for file_path in export_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(export_path)
                    with open(file_path, 'r') as f:
                        files[str(relative_path)] = f.read()
            
            # Clean up export directory
            # TEMPORARILY COMMENTED FOR DEBUGGING
            # shutil.rmtree(export_path)
            
            return files
            
        except Exception as e:
            print(f"Export error: {e}")
            return None
    
    @staticmethod
    async def deploy_workflow_to_client(
        websocket,
        workflow_name: str,
        client_hostname: str,
        runner_args: str = "",
        workflow_id: Optional[str] = None,
        display_name: Optional[str] = None,
        run_after_deploy: bool = True,
        monitor_execution: bool = False,
        copy_dir: Optional[Tuple[str, str]] = None
    ) -> Optional[float]:
        """
        Export and deploy a workflow to a test client.
        
        Args:
            websocket: Active WebSocket connection to test port
            workflow_name: Name of the workflow to export and deploy
            client_hostname: Hostname of the target client
            runner_args: Arguments to pass to runner.py
            workflow_id: Optional workflow ID (auto-generated if not provided)
            display_name: Optional display name (uses workflow_name if not provided)
            run_after_deploy: Whether to start execution after deployment
            monitor_execution: Whether to monitor execution and return timing
            copy_dir: Optional tuple of (src_dir, rel_targ_dir) to copy data files
                     src_dir: Path to source directory with data
                     rel_targ_dir: Relative path within workflow directory
            
        Returns:
            If monitor_execution is True, returns execution time in seconds.
            If monitor_execution is False, returns 0.0 for success.
            Returns None if deployment or execution failed.
        """
        # Export the workflow
        files = DeploymentHelper.export_workflow_locally(workflow_name)
        if not files:
            print(f"Failed to export workflow: {workflow_name}")
            return None
        
        # Generate workflow ID if not provided
        if not workflow_id:
            workflow_id = f"{workflow_name}_{int(time.time())}"
        
        # Use display name if provided
        if not display_name:
            display_name = workflow_name
        
        # If copy_dir is provided, we'll deploy without running, copy files, then start manually
        deploy_and_run = run_after_deploy and copy_dir is None
        
        # Deploy the files
        success = await DeploymentHelper.deploy_files_to_client(
            websocket=websocket,
            files=files,
            client_hostname=client_hostname,
            workflow_id=workflow_id,
            workflow_name=display_name,
            runner_args=runner_args,
            run_after_deploy=deploy_and_run,
            wait_for_confirmation=True
        )
        
        if not success:
            return None
        
        # If copy_dir is provided, copy the data files
        if copy_dir:
            # No need to wait - deploy_success now means files are on disk
            src_dir, rel_targ_dir = copy_dir
            success = await DeploymentHelper.copy_data_to_workflow(
                workflow_id=workflow_id,
                src_dir=src_dir,
                rel_targ_dir=rel_targ_dir
            )
            if not success:
                print(f"Failed to copy data files to workflow")
                return None
            
            # Now start the workflow manually
            if run_after_deploy:
                success = await DeploymentHelper.start_workflow_manually(
                    websocket=websocket,
                    workflow_id=workflow_id,
                    runner_args=runner_args
                )
                if not success:
                    print(f"Failed to start workflow manually")
                    return None
        
        # If not monitoring execution, return success
        if not monitor_execution:
            return 0.0
        
        # Monitor workflow execution
        return await DeploymentHelper.monitor_workflow_execution(websocket, workflow_id)
    
    @staticmethod
    async def copy_data_to_workflow(
        workflow_id: str,
        src_dir: str,
        rel_targ_dir: str
    ) -> bool:
        """
        Copy data files to a deployed workflow directory.
        
        Args:
            workflow_id: ID of the deployed workflow
            src_dir: Source directory containing files to copy
            rel_targ_dir: Relative target directory within workflow
            
        Returns:
            True if copy succeeded, False otherwise
        """
        try:
            # Get workflow directory
            workflow_dir = Path(f"/tmp/dnne_work_areas/{workflow_id}")
            if not workflow_dir.exists():
                print(f"Workflow directory not found: {workflow_dir}")
                return False
            
            # Create target directory
            target_dir = workflow_dir / rel_targ_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files from source to target
            src_path = Path(src_dir)
            if not src_path.exists():
                print(f"Source directory not found: {src_path}")
                return False
            
            # Copy all files in the source directory
            import shutil
            for item in src_path.iterdir():
                if item.is_file():
                    dest = target_dir / item.name
                    shutil.copy2(item, dest)
                    print(f"   Copied {item.name} to workflow")
                elif item.is_dir():
                    dest = target_dir / item.name
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                    print(f"   Copied directory {item.name} to workflow")
            
            print(f"✅ Data files copied to {target_dir}")
            return True
            
        except Exception as e:
            print(f"Error copying data files: {e}")
            return False
    
    @staticmethod
    async def start_workflow_manually(
        websocket,
        workflow_id: str,
        runner_args: str = ""
    ) -> bool:
        """
        Manually start a deployed workflow.
        
        Args:
            websocket: Active WebSocket connection
            workflow_id: ID of the workflow to start
            runner_args: Arguments to pass to runner.py
            
        Returns:
            True if start command sent successfully, False otherwise
        """
        try:
            # Parse runner_args into a list
            import shlex
            args_list = shlex.split(runner_args) if runner_args else []
            
            # Send start_workflow message with args as a list
            await websocket.send(json.dumps({
                "type": "start_workflow",
                "workflow_id": workflow_id,
                "args": args_list  # Server expects "args" not "runner_args"
            }))
            
            # Wait for confirmation (with timeout)
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                
                if data.get("type") == "workflow_started":
                    print(f"✅ Workflow {workflow_id} started manually")
                    return True
                elif data.get("type") == "error":
                    print(f"Error starting workflow: {data.get('message')}")
                    return False
                    
            except asyncio.TimeoutError:
                # No confirmation, but may have started anyway
                print(f"⚠️  No confirmation received for workflow start (may have started anyway)")
                return True
                
            return True
            
        except Exception as e:
            print(f"Error starting workflow: {e}")
            return False
    
    @staticmethod
    async def monitor_workflow_execution(
        websocket,
        workflow_id: str,
        timeout: float = 600.0
    ) -> Optional[float]:
        """
        Monitor a workflow's execution and return the execution time.
        
        Args:
            websocket: Active WebSocket connection to test port
            workflow_id: ID of the workflow to monitor
            timeout: Maximum time to wait for completion (seconds)
            
        Returns:
            Execution time in seconds, or None if workflow failed/timed out
        """
        start_time = None
        end_time = None
        
        start_monitor = time.time()
        
        while time.time() - start_monitor < timeout:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                data = json.loads(response)
                
                # Check for workflow status messages
                if data.get("type") == "workflow_status" and data.get("workflow_id") == workflow_id:
                    status = data.get("status")
                    
                    if status == "running" and start_time is None:
                        start_time = time.time()
                    elif status in ["completed", "failed", "terminated"]:
                        end_time = time.time()
                        
                        if status == "completed" and start_time:
                            return end_time - start_time
                        else:
                            return None
                
                # Check for workflow logs that indicate completion
                elif data.get("type") == "workflow_log" and data.get("workflow_id") == workflow_id:
                    log_msg = data.get("log", {}).get("message", "")
                    completion_indicators = [
                        "Training complete",
                        "epochs completed",
                        "Workflow completed",
                        "Execution finished"
                    ]
                    
                    if any(indicator.lower() in log_msg.lower() for indicator in completion_indicators):
                        if start_time and not end_time:
                            end_time = time.time()
                            return end_time - start_time
                            
            except asyncio.TimeoutError:
                # No message received in 1 second, continue waiting
                pass
        
        print(f"Workflow execution monitoring timed out after {timeout} seconds")
        return None
    
    @staticmethod
    async def cleanup_workflow_directories(hostname: str, pattern: str = "*"):
        """
        Clean up workflow directories for a specific test client.
        
        Args:
            hostname: Hostname of the test client
            pattern: Glob pattern for directories to remove (default: all)
        """
        base_path = Path("/home/asantanna/DNNE/DNNE-UI/remote_clients") / hostname
        
        if base_path.exists():
            for test_dir in base_path.glob(pattern):
                if test_dir.is_dir():
                    try:
                        shutil.rmtree(test_dir)
                    except Exception as e:
                        print(f"Warning: Could not remove {test_dir}: {e}")


# Convenience functions for direct use
async def check_client_connected(websocket, hostname: str) -> bool:
    """Check if a test client is connected"""
    return await DeploymentHelper.check_client_connected(websocket, hostname)


async def deploy_files_to_client(
    websocket,
    files: Dict[str, str],
    client_hostname: str,
    workflow_id: str,
    workflow_name: str,
    runner_args: str = "",
    run_after_deploy: bool = True
) -> bool:
    """Deploy files to a test client"""
    return await DeploymentHelper.deploy_files_to_client(
        websocket, files, client_hostname, workflow_id, 
        workflow_name, runner_args, run_after_deploy
    )


async def deploy_workflow_to_client(
    websocket,
    workflow_name: str,
    client_hostname: str,
    runner_args: str = "",
    workflow_id: Optional[str] = None,
    monitor_execution: bool = False,
    copy_dir: Optional[Tuple[str, str]] = None
) -> Optional[float]:
    """Export and deploy a workflow to a test client"""
    return await DeploymentHelper.deploy_workflow_to_client(
        websocket, workflow_name, client_hostname, runner_args,
        workflow_id, run_after_deploy=True, monitor_execution=monitor_execution,
        copy_dir=copy_dir
    )


async def monitor_workflow_execution(websocket, workflow_id: str) -> Optional[float]:
    """Monitor workflow execution and return timing"""
    return await DeploymentHelper.monitor_workflow_execution(websocket, workflow_id)


async def cleanup_workflow_directories(hostname: str, pattern: str = "*"):
    """Clean up workflow directories"""
    await DeploymentHelper.cleanup_workflow_directories(hostname, pattern)