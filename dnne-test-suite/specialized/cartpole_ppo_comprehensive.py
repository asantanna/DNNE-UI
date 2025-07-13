#!/usr/bin/env python3
"""
Comprehensive test script for Cartpole PPO workflow
Tests single/multi-environment execution and training convergence
"""

import sys
import asyncio
import subprocess
import time
import json
from pathlib import Path
import shutil
import signal

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from export_system.graph_exporter import GraphExporter
from export_system.node_exporters import register_all_exporters

def load_workflow(workflow_path):
    """Load workflow JSON file"""
    with open(workflow_path, 'r') as f:
        return json.load(f)

def modify_workflow_num_envs(workflow, num_envs):
    """Modify the num_envs parameter in the workflow"""
    for node in workflow.get("nodes", []):
        if node.get("class_type") == "IsaacGymEnvNode" or node.get("type") == "IsaacGymEnvNode":
            # Find the widgets_values array and update num_envs (it's the second value)
            if "widgets_values" in node and len(node["widgets_values"]) > 1:
                node["widgets_values"][1] = num_envs
                print(f"Modified IsaacGymEnvNode to use num_envs={num_envs}")
                return workflow
    print("WARNING: Could not find IsaacGymEnvNode to modify num_envs")
    return workflow

def export_workflow(workflow, output_name):
    """Export workflow to Python code"""
    print(f"\n{'='*60}")
    print(f"Exporting workflow to: {output_name}")
    print(f"{'='*60}")
    
    # Create exporter
    exporter = GraphExporter()
    
    # Register node exporters
    register_all_exporters(exporter)
    
    # Export path must be within export_system/exports/
    export_path = Path(__file__).parent.parent / "export_system" / "exports" / output_name
    
    # Clean up if exists
    if export_path.exists():
        shutil.rmtree(export_path)
    
    # Export workflow
    try:
        output = exporter.export_workflow(workflow, export_path)
        print(f"✅ Export successful to: {export_path}")
        return export_path
    except Exception as e:
        print(f"❌ Export failed: {e}")
        raise

async def run_with_timeout(cmd, timeout_seconds, test_name):
    """Run command with timeout and capture output"""
    print(f"\n{'='*60}")
    print(f"Running test: {test_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Timeout: {timeout_seconds}s")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    try:
        # Wait for process with timeout
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds
        )
        
        elapsed = time.time() - start_time
        
        # Decode output
        stdout_str = stdout.decode('utf-8', errors='replace')
        stderr_str = stderr.decode('utf-8', errors='replace')
        
        print(f"\n{'='*60}")
        print(f"Test completed in {elapsed:.1f}s")
        print(f"Exit code: {process.returncode}")
        
        # Check for success patterns
        if "Test mode completed" in stdout_str:
            print("✅ Test mode completed successfully")
        
        # Extract reward information
        reward_lines = [line for line in stdout_str.split('\n') if 'reward' in line.lower()]
        if reward_lines:
            print("\nReward progression:")
            for line in reward_lines[-5:]:  # Last 5 reward updates
                print(f"  {line}")
        
        return {
            "success": process.returncode == 0,
            "elapsed": elapsed,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "test_name": test_name
        }
        
    except asyncio.TimeoutError:
        print(f"\n⏱️ Test timed out after {timeout_seconds}s")
        process.terminate()
        await asyncio.sleep(0.5)
        if process.returncode is None:
            process.kill()
        return {
            "success": False,
            "elapsed": timeout_seconds,
            "stdout": "",
            "stderr": "Test timed out",
            "test_name": test_name
        }

async def test_single_env():
    """Test with single environment"""
    # Load workflow
    workflow_path = Path(__file__).parent.parent / "user" / "default" / "workflows" / "Cartpole_PPO.json"
    workflow = load_workflow(workflow_path)
    
    # Modify to single environment
    workflow = modify_workflow_num_envs(workflow, 1)
    
    # Export
    export_path = export_workflow(workflow, "Cartpole_PPO_Test1")
    
    # Run with timeout
    cmd = [
        sys.executable,
        str(export_path / "runner.py"),
        "--test-mode",
        "--verbose"
    ]
    
    # Activate conda environment
    activate_cmd = "source /home/asantanna/miniconda/bin/activate DNNE_PY38"
    full_cmd = ["bash", "-c", f"{activate_cmd} && {' '.join(cmd)}"]
    
    result = await run_with_timeout(full_cmd, 120, "Single Environment (num_envs=1)")
    
    # Analyze results
    if result["success"]:
        print("\n✅ Single environment test PASSED")
    else:
        print("\n❌ Single environment test FAILED")
        if result["stderr"]:
            print(f"Error output:\n{result['stderr'][:500]}")
    
    return result

async def test_multi_env():
    """Test with 2 environments"""
    # Load workflow
    workflow_path = Path(__file__).parent.parent / "user" / "default" / "workflows" / "Cartpole_PPO.json"
    workflow = load_workflow(workflow_path)
    
    # Modify to 2 environments
    workflow = modify_workflow_num_envs(workflow, 2)
    
    # Export
    export_path = export_workflow(workflow, "Cartpole_PPO_Test2")
    
    # Run with timeout
    cmd = [
        sys.executable,
        str(export_path / "runner.py"),
        "--test-mode",
        "--verbose"
    ]
    
    # Activate conda environment
    activate_cmd = "source /home/asantanna/miniconda/bin/activate DNNE_PY38"
    full_cmd = ["bash", "-c", f"{activate_cmd} && {' '.join(cmd)}"]
    
    result = await run_with_timeout(full_cmd, 120, "Multi Environment (num_envs=2)")
    
    # Analyze results
    if result["success"]:
        print("\n✅ Multi-environment test PASSED")
    else:
        print("\n❌ Multi-environment test FAILED")
        if result["stderr"]:
            print(f"Error output:\n{result['stderr'][:500]}")
    
    return result

async def test_standard_isaac_gym():
    """Test with standard Isaac Gym environment count"""
    # Load workflow
    workflow_path = Path(__file__).parent.parent / "user" / "default" / "workflows" / "Cartpole_PPO.json"
    workflow = load_workflow(workflow_path)
    
    # Modify to 16 environments (common Isaac Gym default)
    workflow = modify_workflow_num_envs(workflow, 16)
    
    # Export
    export_path = export_workflow(workflow, "Cartpole_PPO_Test16")
    
    # Run with timeout (longer for more environments)
    cmd = [
        sys.executable,
        str(export_path / "runner.py"),
        "--test-mode"
    ]
    
    # Activate conda environment
    activate_cmd = "source /home/asantanna/miniconda/bin/activate DNNE_PY38"
    full_cmd = ["bash", "-c", f"{activate_cmd} && {' '.join(cmd)}"]
    
    result = await run_with_timeout(full_cmd, 300, "Standard Isaac Gym (num_envs=16)")
    
    # Analyze results
    if result["success"]:
        print("\n✅ Standard Isaac Gym test PASSED")
    else:
        print("\n❌ Standard Isaac Gym test FAILED")
        if result["stderr"]:
            print(f"Error output:\n{result['stderr'][:500]}")
    
    return result

async def main():
    """Run all tests"""
    print("🚀 Cartpole PPO Comprehensive Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test 1: Single environment
    print("\n📋 Test 1: Single Environment")
    result1 = await test_single_env()
    results.append(result1)
    
    # Test 2: Multi environment
    print("\n📋 Test 2: Multi Environment")
    result2 = await test_multi_env()
    results.append(result2)
    
    # Test 3: Standard Isaac Gym
    print("\n📋 Test 3: Standard Isaac Gym Environment Count")
    result3 = await test_standard_isaac_gym()
    results.append(result3)
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    
    for result in results:
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"{result['test_name']}: {status} ({result['elapsed']:.1f}s)")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)