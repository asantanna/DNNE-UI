#!/usr/bin/env python3
"""
Test export of all workflows in the user/default/workflows directory
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from export_system.graph_exporter import GraphExporter

def test_all_workflows():
    """Test exporting all workflows"""
    workflows_dir = Path("user/default/workflows")
    results = []
    
    # Create exporter once
    exporter = GraphExporter()
    
    # Find all workflow files
    workflow_files = sorted(workflows_dir.glob("*.json"))
    
    print(f"Found {len(workflow_files)} workflows to test\n")
    
    for workflow_file in workflow_files:
        workflow_name = workflow_file.stem
        print(f"{'='*60}")
        print(f"Testing: {workflow_name}")
        print(f"{'='*60}")
        
        try:
            # Load workflow
            with open(workflow_file) as f:
                workflow = json.load(f)
            
            # Add workflow name to metadata if not present
            if 'metadata' not in workflow:
                workflow['metadata'] = {}
            if 'workflow_name' not in workflow['metadata']:
                workflow['metadata']['workflow_name'] = workflow_name
            
            # Try to export
            output_path = Path(f"export_system/exports/test_{workflow_name.lower().replace(' ', '_')}")
            output = exporter.export_workflow(workflow, output_path=output_path)
            
            print(f"✅ SUCCESS: Exported to {output}")
            results.append((workflow_name, "SUCCESS", None))
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            results.append((workflow_name, "FAILED", str(e)))
        
        print()
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    success_count = sum(1 for _, status, _ in results if status == "SUCCESS")
    total_count = len(results)
    
    for name, status, error in results:
        if status == "SUCCESS":
            print(f"✅ {name}")
        else:
            print(f"❌ {name}: {error}")
    
    print(f"\n{success_count}/{total_count} workflows exported successfully")
    
    return success_count == total_count

if __name__ == "__main__":
    success = test_all_workflows()
    sys.exit(0 if success else 1)