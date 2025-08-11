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
            # Run the export
            result = subprocess.run(
                [sys.executable, str(export_script), workflow_name, "--target-dir", target_dir],
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
            shutil.rmtree(export_path)
            
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
        monitor_execution: bool = False
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
        
        # Deploy the files
        success = await DeploymentHelper.deploy_files_to_client(
            websocket=websocket,
            files=files,
            client_hostname=client_hostname,
            workflow_id=workflow_id,
            workflow_name=display_name,
            runner_args=runner_args,
            run_after_deploy=run_after_deploy,
            wait_for_confirmation=True
        )
        
        if not success:
            return None
        
        # If not monitoring execution, return success
        if not monitor_execution:
            return 0.0
        
        # Monitor workflow execution
        return await DeploymentHelper.monitor_workflow_execution(websocket, workflow_id)
    
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
        
        try:
            start_monitor = time.time()
            
            while time.time() - start_monitor < timeout:
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
    monitor_execution: bool = False
) -> Optional[float]:
    """Export and deploy a workflow to a test client"""
    return await DeploymentHelper.deploy_workflow_to_client(
        websocket, workflow_name, client_hostname, runner_args,
        workflow_id, run_after_deploy=True, monitor_execution=monitor_execution
    )


async def monitor_workflow_execution(websocket, workflow_id: str) -> Optional[float]:
    """Monitor workflow execution and return timing"""
    return await DeploymentHelper.monitor_workflow_execution(websocket, workflow_id)


async def cleanup_workflow_directories(hostname: str, pattern: str = "*"):
    """Clean up workflow directories"""
    await DeploymentHelper.cleanup_workflow_directories(hostname, pattern)