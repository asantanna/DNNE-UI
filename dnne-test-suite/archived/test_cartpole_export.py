#!/usr/bin/env python3
"""
Test export of Cartpole PPO workflow with fallbacks removed
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export_system.graph_exporter import GraphExporter
from export_system.node_exporters import register_all_exporters

def main():
    print("Testing Cartpole PPO Workflow Export (No Fallbacks)...")
    print("=" * 60)
    
    # Create exporter
    exporter = GraphExporter()
    register_all_exporters(exporter)
    
    print(f" Loaded export system with {len(exporter.node_registry)} node types")
    
    # Load Cartpole PPO workflow
    workflow_path = Path("user/default/workflows/Cartpole_PPO.json")
    if not workflow_path.exists():
        print(f"✗ Workflow not found: {workflow_path}")
        return
    
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    print(f" Loaded workflow with {len(workflow.get('nodes', []))} nodes")
    
    # Find IsaacGymEnvNode and check its widget values
    isaac_gym_node = None
    for node in workflow.get('nodes', []):
        if node.get('type') == 'IsaacGymEnvNode':
            isaac_gym_node = node
            break
    
    if isaac_gym_node:
        widgets = isaac_gym_node.get('widgets_values', [])
        print(f" Found IsaacGymEnvNode with widgets: {widgets}")
        if len(widgets) > 1:
            print(f" num_envs value: {widgets[1]} (type: {type(widgets[1])})")
    
    # Export the workflow
    try:
        output_path = Path("export_system/exports/Cartpole_PPO_NoFallbacks")
        result = exporter.export_workflow(workflow, output_path)
        print("✓ Export successful!")
        print(f" Generated package at: {output_path}")
        
        # Check if the generated code contains the correct num_envs value
        runner_path = output_path / "runner.py"
        if runner_path.exists():
            with open(runner_path, 'r') as f:
                content = f.read()
            
            # Look for num_envs in the generated code
            if "num_envs = 16" in content:
                print("✓ num_envs=16 found in generated code!")
            elif "num_envs = 1" in content:
                print("✗ num_envs=1 found in generated code (bug still present)")
            else:
                print("? num_envs value not clearly identifiable in generated code")
                # Look for any num_envs references
                import re
                matches = re.findall(r'num_envs\s*=\s*(\w+)', content)
                if matches:
                    print(f" Found num_envs values: {matches}")
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()