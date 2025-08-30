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

def export_workflow(workflow_name="Cartpole_PPO"):
    """Export the specified workflow programmatically
    
    Args:
        workflow_name: Can be:
            - Just the workflow name (e.g., "Shadow_Train")
            - A relative path (e.g., "user/default/workflows/Shadow_Train.json")
            - An absolute path (e.g., "/home/user/workflows/Shadow_Train.json")
    """
    print(f"🚀 Starting programmatic export of {workflow_name}...")
    
    # Create exporter and register all node types
    exporter = GraphExporter()
    register_all_exporters(exporter)
    
    print(f"✓ Loaded export system with {len(exporter.node_registry)} node types")
    
    # Determine the workflow path based on input format
    workflow_path = None
    original_name = workflow_name
    
    # Check if it's an absolute or relative path
    if os.path.sep in workflow_name or workflow_name.endswith('.json'):
        # It's a path
        workflow_path = Path(workflow_name)
        
        # If relative and doesn't exist, try from current directory
        if not workflow_path.is_absolute() and not workflow_path.exists():
            workflow_path = Path.cwd() / workflow_path
        
        # Extract the workflow name from the path
        if workflow_path.exists():
            workflow_name = workflow_path.stem  # Get filename without extension
    else:
        # It's just a workflow name - use default location
        workflow_path = Path(f"user/default/workflows/{workflow_name}.json")
    
    # Check if the file exists
    if not workflow_path.exists():
        print(f"❌ Workflow file not found: {workflow_path}")
        
        # Try some common variations
        alternatives = []
        if not str(workflow_path).endswith('.json'):
            alternatives.append(Path(str(workflow_path) + '.json'))
        
        # If it was just a name, try looking in current directory too
        if os.path.sep not in original_name:
            alternatives.append(Path(f"{workflow_name}.json"))
            alternatives.append(Path.cwd() / f"{workflow_name}.json")
        
        for alt in alternatives:
            if alt.exists():
                workflow_path = alt
                print(f"✓ Found workflow at: {workflow_path}")
                break
        else:
            print(f"Tried alternatives: {[str(a) for a in alternatives]}")
            return False
        
    print(f"📁 Loading workflow: {workflow_path}")
    
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    print(f"✓ Loaded workflow with {len(workflow.get('nodes', []))} nodes")
    
    # Export the workflow
    try:
        # Clean workflow name for directory (replace spaces with underscores)
        clean_name = workflow_name.replace(" ", "_")
        # Add workflow name to metadata for slot correction
        if 'metadata' not in workflow:
            workflow['metadata'] = {}
        workflow['metadata']['workflow_name'] = workflow_name
        # Skip slot correction if workflow has labels (they add dynamic connections)
        if 'extra' in workflow and 'labelDictionary' in workflow.get('extra', {}):
            workflow['metadata']['skip-slot-correction'] = True
            print(f"📝 Set workflow_name in metadata: {workflow['metadata']['workflow_name']} (skip-slot-correction enabled for labels)")
        else:
            print(f"📝 Set workflow_name in metadata: {workflow['metadata']['workflow_name']}")
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
    # Get workflow name from command line argument or default to Cartpole_PPO
    workflow_name = sys.argv[1] if len(sys.argv) > 1 else "Cartpole_PPO"
    success = export_workflow(workflow_name)
    sys.exit(0 if success else 1)