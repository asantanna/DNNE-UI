#!/usr/bin/env python3
"""
Clean context references from Cartpole_PPO.json workflow
"""

import json
from pathlib import Path

def clean_context_from_workflow():
    """Remove all context inputs and outputs from workflow nodes"""
    
    workflow_path = Path("user/default/workflows/Cartpole_PPO.json")
    
    # Load workflow
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    print("Cleaning context from Cartpole_PPO workflow...")
    print("=" * 50)
    
    changes_made = []
    
    # Process each node
    for node in workflow.get('nodes', []):
        node_id = node['id']
        node_type = node['type']
        
        # Remove context from inputs
        original_inputs = node.get('inputs', [])
        cleaned_inputs = [inp for inp in original_inputs if 'context' not in inp.get('name', '').lower()]
        
        if len(cleaned_inputs) != len(original_inputs):
            removed_count = len(original_inputs) - len(cleaned_inputs)
            changes_made.append(f"Node {node_id} ({node_type}): Removed {removed_count} context input(s)")
            node['inputs'] = cleaned_inputs
        
        # Remove context from outputs  
        original_outputs = node.get('outputs', [])
        cleaned_outputs = [out for out in original_outputs if 'context' not in out.get('name', '').lower()]
        
        if len(cleaned_outputs) != len(original_outputs):
            removed_count = len(original_outputs) - len(cleaned_outputs)
            changes_made.append(f"Node {node_id} ({node_type}): Removed {removed_count} context output(s)")
            node['outputs'] = cleaned_outputs
    
    if changes_made:
        print("Changes made:")
        for change in changes_made:
            print(f"  ✓ {change}")
        
        # Save cleaned workflow
        with open(workflow_path, 'w') as f:
            json.dump(workflow, f, indent=2)
        
        print(f"\n✓ Cleaned workflow saved to {workflow_path}")
    else:
        print("✓ No context references found - workflow already clean")
    
    # Validate the result
    print("\nValidation:")
    with open(workflow_path, 'r') as f:
        updated_workflow = json.load(f)
    
    context_found = False
    for node in updated_workflow.get('nodes', []):
        for inp in node.get('inputs', []):
            if 'context' in inp.get('name', '').lower():
                print(f"  ⚠️  Context input still found in node {node['id']}: {inp['name']}")
                context_found = True
        for out in node.get('outputs', []):
            if 'context' in out.get('name', '').lower():
                print(f"  ⚠️  Context output still found in node {node['id']}: {out['name']}")
                context_found = True
    
    if not context_found:
        print("  ✓ No context references remain in workflow")
    
    return len(changes_made) > 0

if __name__ == "__main__":
    clean_context_from_workflow()