#!/usr/bin/env python3
"""
Test script to verify telemetry pipeline from exported code to DNNE.

This script:
1. Exports a workflow with a balancing node
2. Runs it with telemetry enabled
3. Verifies telemetry files are created
"""

import json
import time
import sys
import os
from pathlib import Path
import subprocess

def main():
    print("🧪 Testing Telemetry Pipeline")
    print("=" * 60)
    
    # Test workflow with balancing node
    workflow = {
        "prompt": {
            "10": {
                "class_type": "BalancingNode",
                "inputs": {
                    "enabled": True,
                    "min_hz": 10.0,
                    "max_hz": 100.0,
                    "target_hz": 30.0,
                    "max_latency_ms": 100.0,
                    "log_violations": True
                }
            }
        }
    }
    
    # Save test workflow
    workflow_path = Path("/tmp/test_telemetry_workflow.json")
    workflow_path.write_text(json.dumps(workflow, indent=2))
    print(f"✅ Created test workflow: {workflow_path}")
    
    # Export the workflow
    print("\n📦 Exporting workflow...")
    export_cmd = [
        "python", "export_system/graph_exporter.py",
        "--workflow", str(workflow_path),
        "--output", "/tmp/telemetry_test_export"
    ]
    
    try:
        result = subprocess.run(export_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print(f"❌ Export failed: {result.stderr}")
            return 1
        print("✅ Workflow exported successfully")
    except Exception as e:
        print(f"❌ Export error: {e}")
        return 1
    
    # Run the exported workflow with telemetry
    print("\n🚀 Running workflow with telemetry...")
    run_cmd = [
        "python", "/tmp/telemetry_test_export/runner.py",
        "--enable-telemetry", "10",  # Enable for node 10
        "--timeout", "5s"  # Run for 5 seconds
    ]
    
    print(f"Command: {' '.join(run_cmd)}")
    
    try:
        # Note: This assumes the agent is running and will receive telemetry
        result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=10)
        print("\nWorkflow output:")
        print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
        
        if result.returncode != 0 and "Stopped by user" not in result.stdout:
            print(f"⚠️ Workflow exited with code {result.returncode}")
            print(f"Error output: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("✅ Workflow ran for full duration")
    except Exception as e:
        print(f"❌ Run error: {e}")
        return 1
    
    # Check for telemetry files
    print("\n🔍 Checking for telemetry files...")
    
    # Look for telemetry in remote_clients directory
    remote_clients = Path("remote_clients")
    if remote_clients.exists():
        telemetry_found = False
        for client_dir in remote_clients.iterdir():
            if client_dir.is_dir():
                for workflow_dir in client_dir.iterdir():
                    if workflow_dir.is_dir():
                        telemetry_dir = workflow_dir / "telemetry"
                        if telemetry_dir.exists():
                            print(f"📊 Found telemetry directory: {telemetry_dir}")
                            for telem_run in telemetry_dir.iterdir():
                                if telem_run.is_dir() and telem_run.name.startswith("telem_"):
                                    print(f"  Run: {telem_run.name}")
                                    for file in telem_run.iterdir():
                                        size = file.stat().st_size
                                        print(f"    - {file.name}: {size} bytes")
                                        if file.name.endswith("_violations.log"):
                                            # Show first few violations
                                            lines = file.read_text().splitlines()[:5]
                                            for line in lines:
                                                print(f"      {line}")
                                    telemetry_found = True
        
        if telemetry_found:
            print("\n✅ Telemetry pipeline test PASSED!")
            return 0
        else:
            print("⚠️ No telemetry files found - may need agent running")
            return 1
    else:
        print("⚠️ No remote_clients directory - telemetry requires agent")
        return 1

if __name__ == "__main__":
    sys.exit(main())