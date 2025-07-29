#!/usr/bin/env python3
"""
Programmatic export of DNNE workflows
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export_system.graph_exporter import GraphExporter
from export_system.node_exporters import register_all_exporters

def export_workflow(workflow_name):
    """Export the specified workflow programmatically"""
    print(f"🚀 Starting programmatic export of {workflow_name}...")
    
    # Create exporter and register all node types
    exporter = GraphExporter()
    register_all_exporters(exporter)
    
    print(f"✓ Loaded export system with {len(exporter.node_registry)} node types")
    
    # Load the workflow file
    workflow_path = Path(f"user/default/workflows/{workflow_name}.json")
    
    if not workflow_path.exists():
        print(f"❌ Workflow file not found: {workflow_path}")
        return False
        
    print(f"📁 Loading workflow: {workflow_path}")
    
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    print(f"✓ Loaded workflow with {len(workflow.get('nodes', []))} nodes")
    
    # Export the workflow
    try:
        # Clean workflow name for directory (replace spaces with underscores)
        clean_name = workflow_name.replace(" ", "_")
        output_path = exporter.export_workflow(workflow, output_path=Path(f"export_system/exports/{clean_name}"))
        print(f"✅ Export completed successfully!")
        print(f"📂 Output location: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Workflow name is required
    if len(sys.argv) < 2:
        print("❌ Error: Workflow name is required")
        print("Usage: python programmatic_export.py <workflow_name>")
        print("Example: python programmatic_export.py Cartpole_PPO")
        print("Example: python programmatic_export.py \"MNIST Test\"")
        sys.exit(1)
    
    workflow_name = sys.argv[1]
    success = export_workflow(workflow_name)
    sys.exit(0 if success else 1)