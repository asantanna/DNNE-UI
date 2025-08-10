#!/usr/bin/env python3
"""
Programmatic export utility for DNNE workflows
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path (go up two levels from utilities/programmatic_export.py)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from export_system.graph_exporter import GraphExporter
from export_system.node_exporters import register_all_exporters

def get_available_workflows():
    """Get list of available workflow files"""
    workflows_dir = Path("user/default/workflows")
    if not workflows_dir.exists():
        return []
    
    workflows = []
    for workflow_file in workflows_dir.glob("*.json"):
        # Remove .json extension
        workflow_name = workflow_file.stem
        workflows.append(workflow_name)
    
    return sorted(workflows)

def normalize_workflow_name(name):
    """Convert workflow name to filename format"""
    # Handle common cases like "MNIST_Test" -> "MNIST_Test.json"
    if not name.endswith('.json'):
        return f"{name}.json"
    return name

def export_workflow(workflow_name, target_dir=None, add_metadata=False):
    """Export a workflow programmatically"""
    print(f"🚀 Starting programmatic export of {workflow_name}...")
    
    # Create exporter and register all node types
    exporter = GraphExporter()
    register_all_exporters(exporter)
    
    print(f"✓ Loaded export system with {len(exporter.node_registry)} node types")
    
    # Load the workflow file
    workflow_filename = normalize_workflow_name(workflow_name)
    workflow_path = Path("user/default/workflows") / workflow_filename
    
    if not workflow_path.exists():
        print(f"❌ Workflow file not found: {workflow_path}")
        print(f"Available workflows: {', '.join(get_available_workflows())}")
        return False
        
    print(f"📁 Loading workflow: {workflow_path}")
    
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    # Add metadata if requested
    if add_metadata:
        if "metadata" not in workflow:
            workflow["metadata"] = {}
        workflow["metadata"]["dnne-test"] = True
    
    print(f"✓ Loaded workflow with {len(workflow.get('nodes', []))} nodes")
    
    # Determine output path
    if target_dir:
        output_path = Path("export_system/exports") / target_dir
        print(f"📂 Target directory: {target_dir}")
    else:
        # Use default behavior (timestamped directory)
        output_path = None
        print("📂 Using default timestamped directory")
    
    # Export the workflow
    try:
        result_path = exporter.export_workflow(workflow, output_path=output_path)
        print(f"✅ Export completed successfully!")
        print(f"📂 Output location: {result_path}")
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Export DNNE workflows to standalone Python scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "MNIST_Test"                    # Export with default timestamped directory
  %(prog)s "MNIST_Test" --target-dir "MNIST_Test"  # Export to specific directory
  %(prog)s --list                         # Show available workflows
        """
    )
    
    parser.add_argument(
        "workflow_name", 
        nargs='?',
        help="Name of the workflow to export (without .json extension)"
    )
    
    parser.add_argument(
        "--target-dir",
        help="Target directory name within export_system/exports/ (optional)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available workflows"
    )
    
    parser.add_argument(
        "--add-metadata",
        action="store_true",
        help="Add test metadata to the workflow (for test suite)"
    )
    
    args = parser.parse_args()
    
    # Handle --list option
    if args.list:
        workflows = get_available_workflows()
        print("Available workflows:")
        for workflow in workflows:
            print(f"  - {workflow}")
        return 0
    
    # Require workflow name if not listing
    if not args.workflow_name:
        parser.error("workflow_name is required unless using --list")
    
    # Perform export
    success = export_workflow(args.workflow_name, args.target_dir, args.add_metadata)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())