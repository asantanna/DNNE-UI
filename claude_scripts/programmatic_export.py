#!/usr/bin/env python3
"""
Programmatic export of Cartpole PPO workflow
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export_system.graph_exporter import GraphExporter
from export_system.node_exporters import register_all_exporters

def export_cartpole_ppo():
    """Export the Cartpole PPO workflow programmatically"""
    print("🚀 Starting programmatic export of Cartpole PPO...")
    
    # Create exporter and register all node types
    exporter = GraphExporter()
    register_all_exporters(exporter)
    
    print(f"✓ Loaded export system with {len(exporter.node_registry)} node types")
    
    # Load the workflow file
    workflow_path = Path("user/default/workflows/Cartpole_PPO.json")
    
    if not workflow_path.exists():
        print(f"❌ Workflow file not found: {workflow_path}")
        return False
        
    print(f"📁 Loading workflow: {workflow_path}")
    
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    print(f"✓ Loaded workflow with {len(workflow.get('nodes', []))} nodes")
    
    # Export the workflow
    try:
        output_path = exporter.export_workflow(workflow, output_path=Path("export_system/exports/Cartpole_PPO"))
        print(f"✅ Export completed successfully!")
        print(f"📂 Output location: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = export_cartpole_ppo()
    sys.exit(0 if success else 1)