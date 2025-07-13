#!/usr/bin/env python3
"""
Quick script to enable checkpoints in an exported workflow for testing
"""

import sys
from pathlib import Path

def enable_checkpoints(export_path: str):
    """Enable checkpoints in all nodes in the export"""
    export_dir = Path(export_path)
    
    if not export_dir.exists():
        print(f"Export directory not found: {export_dir}")
        return False
    
    nodes_dir = export_dir / "nodes"
    if not nodes_dir.exists():
        print(f"Nodes directory not found: {nodes_dir}")
        return False
    
    # Find all Python files in nodes directory
    node_files = list(nodes_dir.glob("*.py"))
    if not node_files:
        print("No node files found")
        return False
    
    modified_count = 0
    
    for node_file in node_files:
        if node_file.name == "__init__.py":
            continue
            
        print(f"Processing {node_file.name}...")
        
        # Read the file
        content = node_file.read_text()
        
        # Check if it has checkpoint configuration
        if "self.checkpoint_enabled = False" in content:
            print(f"  Enabling checkpoints in {node_file.name}")
            content = content.replace(
                "self.checkpoint_enabled = False",
                "self.checkpoint_enabled = True"
            )
            
            # Write back
            node_file.write_text(content)
            modified_count += 1
        else:
            print(f"  No checkpoint config found in {node_file.name}")
    
    print(f"\nModified {modified_count} files")
    return modified_count > 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python enable_checkpoints_in_export.py <export_path>")
        sys.exit(1)
    
    export_path = sys.argv[1]
    if enable_checkpoints(export_path):
        print("✅ Checkpoints enabled successfully")
    else:
        print("❌ Failed to enable checkpoints")
        sys.exit(1)