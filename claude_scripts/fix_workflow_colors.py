#!/usr/bin/env python3
"""
Fix node colors in DNNE workflow files.
Updates the stored color and bgcolor values to match the current node definitions.
"""

import json
import glob
from pathlib import Path

# Define the correct colors for each node type based on node_colors.py
NODE_COLOR_MAP = {
    'BatchSampler': {'color': '#332922', 'bgcolor': '#593930'},  # data
    'CrossEntropyLoss': {'color': '#432', 'bgcolor': '#653'},     # training
    'EpochTracker': {'color': '#432', 'bgcolor': '#653'},         # training
    'GetBatch': {'color': '#332922', 'bgcolor': '#593930'},       # data
    'LinearLayer': {'color': '#232', 'bgcolor': '#353'},          # layer
    'MNISTDataset': {'color': '#332922', 'bgcolor': '#593930'},   # data
    'Network': {'color': '#223', 'bgcolor': '#335'},              # network
    'SGDOptimizer': {'color': '#432', 'bgcolor': '#653'},         # training
    'TrainingStep': {'color': '#432', 'bgcolor': '#653'},         # training
    'CIFAR10Dataset': {'color': '#332922', 'bgcolor': '#593930'}, # data
    'PPOAgent': {'color': '#322', 'bgcolor': '#533'},             # rl
    'PPOConfig': {'color': '#334455', 'bgcolor': '#556677'},      # utility
    'BalancingNode': {'color': '#334455', 'bgcolor': '#556677'},  # utility
    'BalancingConfig': {'color': '#334455', 'bgcolor': '#556677'}, # utility
    'IsaacGymSim': {'color': '#323', 'bgcolor': '#535'},          # simulation
    'IsaacGymEnvs': {'color': '#334455', 'bgcolor': '#556677'},   # utility
    'DataStreamer': {'color': '#332922', 'bgcolor': '#593930'},   # data
    'CustomComputation': {'color': '#334455', 'bgcolor': '#556677'}, # utility
    'ORNode': {'color': '#334455', 'bgcolor': '#556677'}          # utility
}

def fix_workflow_colors(workflow_path):
    """Fix colors in a single workflow file."""
    print(f"\nProcessing: {workflow_path}")
    
    # Load workflow
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    # Track changes
    changes_made = 0
    
    # Update node colors
    for node in workflow['nodes']:
        node_type = node.get('type')
        
        if node_type and node_type in NODE_COLOR_MAP:
            expected_colors = NODE_COLOR_MAP[node_type]
            
            # Check if update needed
            current_color = node.get('color')
            current_bgcolor = node.get('bgcolor')
            
            if current_color != expected_colors['color'] or current_bgcolor != expected_colors['bgcolor']:
                print(f"  Updating node {node['id']} ({node_type}):")
                print(f"    Old: color={current_color}, bgcolor={current_bgcolor}")
                print(f"    New: color={expected_colors['color']}, bgcolor={expected_colors['bgcolor']}")
                
                # Update colors
                node['color'] = expected_colors['color']
                node['bgcolor'] = expected_colors['bgcolor']
                changes_made += 1
    
    # Save if changes were made
    if changes_made > 0:
        with open(workflow_path, 'w') as f:
            json.dump(workflow, f, indent=4)
        print(f"  ✓ Updated {changes_made} nodes")
    else:
        print(f"  ✓ No changes needed")
    
    return changes_made

def main():
    """Fix colors in all workflow files."""
    print("DNNE Workflow Color Fixer")
    print("=" * 50)
    
    # Find all workflow files
    workflow_dir = Path("/home/asantanna/DNNE/DNNE-UI/user/default/workflows")
    workflow_files = list(workflow_dir.glob("*.json"))
    
    if not workflow_files:
        print("No workflow files found!")
        return
    
    print(f"Found {len(workflow_files)} workflow files")
    
    # Process each workflow
    total_changes = 0
    for workflow_path in workflow_files:
        changes = fix_workflow_colors(workflow_path)
        total_changes += changes
    
    # Summary
    print("\n" + "=" * 50)
    print(f"COMPLETE: Updated {total_changes} nodes across all workflows")
    
    # Validate JSON structure
    print("\nValidating JSON structure...")
    all_valid = True
    for workflow_path in workflow_files:
        try:
            with open(workflow_path, 'r') as f:
                json.load(f)
            print(f"  ✓ {workflow_path.name} - Valid JSON")
        except json.JSONDecodeError as e:
            print(f"  ✗ {workflow_path.name} - INVALID JSON: {e}")
            all_valid = False
    
    if all_valid:
        print("\n✓ All workflows have valid JSON structure")
    else:
        print("\n✗ Some workflows have invalid JSON - please check!")

if __name__ == "__main__":
    main()