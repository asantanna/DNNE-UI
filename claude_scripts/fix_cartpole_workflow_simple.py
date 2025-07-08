#!/usr/bin/env python3
"""
Simple fix for Cartpole workflow - only fix top-level workflow ID
"""

import json
import uuid
from pathlib import Path

def generate_uuid():
    """Generate a new UUID string"""
    return str(uuid.uuid4())

def fix_workflow_id_only(workflow_path):
    """Fix only the top-level workflow ID to be a valid UUID, leave node IDs as integers"""
    
    # Load workflow
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    # Only fix the top-level workflow ID if it's not already a valid UUID
    current_id = workflow.get("id", "")
    
    try:
        # Test if current ID is already a valid UUID
        uuid.UUID(current_id)
        print(f"Workflow ID is already a valid UUID: {current_id}")
        return workflow
    except ValueError:
        # Current ID is not a valid UUID, fix it
        print(f"Fixing invalid workflow ID: {current_id}")
        workflow["id"] = generate_uuid()
        print(f"New workflow ID: {workflow['id']}")
    
    return workflow

def main():
    """Main function"""
    workflow_path = Path("user/default/workflows/Cartpole_RL_Single.json")
    
    print(f"Fixing workflow ID in: {workflow_path}")
    
    # Fix the workflow
    fixed_workflow = fix_workflow_id_only(workflow_path)
    
    # Create backup
    backup_path = workflow_path.with_suffix('.json.simple_backup')
    print(f"Creating backup: {backup_path}")
    
    with open(workflow_path, 'r') as f:
        original_content = f.read()
    
    with open(backup_path, 'w') as f:
        f.write(original_content)
    
    # Write fixed workflow
    print("Writing fixed workflow...")
    with open(workflow_path, 'w') as f:
        json.dump(fixed_workflow, f, indent=2)
    
    print("✓ Workflow ID fixed successfully!")
    print(f"✓ Backup saved to: {backup_path}")
    
    # Validate the result
    with open(workflow_path, 'r') as f:
        result = json.load(f)
    
    print(f"✓ New workflow ID: {result['id']}")
    print(f"✓ Number of nodes: {len(result.get('nodes', []))}")
    
    # Check first few node IDs to verify they're still integers
    nodes = result.get('nodes', [])
    if nodes:
        print("✓ Node IDs (first 3):")
        for i, node in enumerate(nodes[:3]):
            print(f"  - {node['type']}: {node['id']} (type: {type(node['id']).__name__})")

if __name__ == "__main__":
    main()