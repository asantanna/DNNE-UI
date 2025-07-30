#!/usr/bin/env python3
"""
Export all workflows in the user/default/workflows directory.

This script finds all workflow JSON files and exports each one using
the programmatic_export script.
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    """Export all workflows."""
    # Get the base directory
    base_dir = Path(__file__).parent.parent
    workflows_dir = base_dir / "user" / "default" / "workflows"
    
    if not workflows_dir.exists():
        print(f"Error: Workflows directory not found at {workflows_dir}")
        sys.exit(1)
    
    # Find all JSON workflow files
    workflow_files = list(workflows_dir.glob("*.json"))
    
    if not workflow_files:
        print("No workflow files found.")
        return
    
    print(f"Found {len(workflow_files)} workflows to export:")
    for workflow_file in workflow_files:
        print(f"  - {workflow_file.stem}")
    print()
    
    # Export each workflow
    failed_exports = []
    successful_exports = []
    
    for workflow_file in workflow_files:
        workflow_name = workflow_file.stem
        print(f"Exporting '{workflow_name}'...")
        
        try:
            # Call programmatic_export.py with the workflow name
            result = subprocess.run(
                [sys.executable, "claude_scripts/programmatic_export.py", workflow_name],
                capture_output=True,
                text=True,
                cwd=str(base_dir)
            )
            
            if result.returncode == 0:
                print(f"  ✓ Successfully exported '{workflow_name}'")
                successful_exports.append(workflow_name)
            else:
                print(f"  ✗ Failed to export '{workflow_name}'")
                print(f"    Error: {result.stderr.strip()}")
                failed_exports.append(workflow_name)
                
        except Exception as e:
            print(f"  ✗ Exception while exporting '{workflow_name}': {e}")
            failed_exports.append(workflow_name)
        
        print()
    
    # Summary
    print("=" * 60)
    print(f"Export Summary:")
    print(f"  Successful: {len(successful_exports)}")
    print(f"  Failed: {len(failed_exports)}")
    
    if failed_exports:
        print(f"\nFailed workflows:")
        for workflow in failed_exports:
            print(f"  - {workflow}")
        sys.exit(1)
    else:
        print("\nAll workflows exported successfully!")

if __name__ == "__main__":
    main()