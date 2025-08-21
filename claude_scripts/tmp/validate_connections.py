#!/usr/bin/env python3
"""
Validate workflow connections after context removal
"""

import json
from pathlib import Path

# Define expected input/output names for each node type (after context removal)
NODE_DEFINITIONS = {
    "IsaacGymEnvNode": {
        "inputs": [],  # No inputs for env node
        "outputs": ["sim_handle", "observations"]
    },
    "IsaacGymStepNode": {
        "inputs": ["sim_handle", "actions", "trigger"],
        "outputs": ["observations", "rewards", "done", "info", "next_observations"]
    },
    "CartpoleActionNode": {
        "inputs": ["network_output"],
        "outputs": ["action"]  
    },
    "PPOAgentNode": {
        "inputs": ["observations"],  # Only data input, widgets are not connections
        "outputs": ["policy_output", "model"]
    },
    "PPOTrainerNode": {
        "inputs": ["state", "policy_output", "reward", "done", "model"],  # Only data inputs
        "outputs": ["loss", "training_complete"]
    },
    "ORNode": {
        "inputs": ["input_a", "input_b", "input_c"],
        "outputs": ["output"]
    }
}

def validate_workflow_connections():
    """Validate all connections in the workflow are correct"""
    
    workflow_path = Path("user/default/workflows/Cartpole_PPO.json")
    
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    print("Validating Cartpole_PPO workflow connections...")
    print("=" * 50)
    
    # Build node lookup by ID
    nodes_by_id = {str(node['id']): node for node in workflow.get('nodes', [])}
    
    errors = []
    warnings = []
    
    # Validate each connection
    for link in workflow.get('links', []):
        if len(link) >= 6:
            link_id, from_node, from_slot, to_node, to_slot, connection_type = link
            from_node, to_node = str(from_node), str(to_node)
            
            print(f"Link {link_id}: {from_node}[{from_slot}] -> {to_node}[{to_slot}] ({connection_type})")
            
            # Validate source node and slot
            if from_node not in nodes_by_id:
                errors.append(f"  ❌ Source node {from_node} not found")
                continue
                
            from_node_data = nodes_by_id[from_node]
            from_node_type = from_node_data['type']
            
            if from_node_type not in NODE_DEFINITIONS:
                warnings.append(f"  ⚠️  Unknown node type: {from_node_type}")
                continue
                
            expected_outputs = NODE_DEFINITIONS[from_node_type]['outputs']
            if from_slot >= len(expected_outputs):
                errors.append(f"  ❌ Invalid output slot {from_slot} for {from_node_type} (max: {len(expected_outputs)-1})")
                continue
                
            expected_output_name = expected_outputs[from_slot]
            print(f"    Source: {from_node_type}.{expected_output_name}")
            
            # Validate destination node and slot
            if to_node not in nodes_by_id:
                errors.append(f"  ❌ Destination node {to_node} not found")
                continue
                
            to_node_data = nodes_by_id[to_node]
            to_node_type = to_node_data['type']
            
            if to_node_type not in NODE_DEFINITIONS:
                warnings.append(f"  ⚠️  Unknown node type: {to_node_type}")
                continue
                
            expected_inputs = NODE_DEFINITIONS[to_node_type]['inputs']
            if to_slot >= len(expected_inputs):
                errors.append(f"  ❌ Invalid input slot {to_slot} for {to_node_type} (max: {len(expected_inputs)-1})")
                continue
                
            expected_input_name = expected_inputs[to_slot]
            print(f"    Destination: {to_node_type}.{expected_input_name}")
            
            print(f"    ✓ Valid connection: {expected_output_name} -> {expected_input_name}")
            print()
    
    # Summary
    print("Validation Summary:")
    print("=" * 20)
    
    if errors:
        print("❌ ERRORS FOUND:")
        for error in errors:
            print(error)
        print()
    
    if warnings:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(warning)
        print()
    
    if not errors and not warnings:
        print("✅ All connections are valid!")
    elif not errors:
        print("✅ No errors found (only warnings)")
    else:
        print(f"❌ Found {len(errors)} errors")
    
    return len(errors) == 0

if __name__ == "__main__":
    validate_workflow_connections()