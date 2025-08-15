#!/usr/bin/env python3
"""
Simple test to check if workflow can be loaded without errors
"""

import json
from pathlib import Path

def test_workflow_loading():
    """Test that the workflow can be loaded"""
    
    workflow_path = Path("user/default/workflows/Cartpole_RL_Single.json")
    
    print(f"Loading workflow: {workflow_path}")
    
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    print(f"✓ Workflow loaded successfully")
    print(f"✓ Workflow ID: {workflow['id']}")
    print(f"✓ Number of nodes: {len(workflow.get('nodes', []))}")
    print(f"✓ Number of links: {len(workflow.get('links', []))}")
    
    # Check node IDs are valid UUIDs
    import uuid
    for node in workflow.get('nodes', []):
        node_id = node['id']
        try:
            uuid.UUID(node_id)
            print(f"✓ Node {node['type']}: Valid UUID {node_id}")
        except ValueError:
            print(f"✗ Node {node['type']}: Invalid UUID {node_id}")
            return False
    
    # Check workflow ID is valid UUID
    try:
        uuid.UUID(workflow['id'])
        print(f"✓ Workflow ID is valid UUID")
    except ValueError:
        print(f"✗ Workflow ID is invalid UUID")
        return False
    
    print("✓ All checks passed!")
    return True

if __name__ == "__main__":
    test_workflow_loading()