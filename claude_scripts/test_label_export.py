#!/usr/bin/env python3
"""
Test script for label-based connections in DNNE export system.
Creates a mock workflow with labels and tests export functionality.
"""

import sys
import json
import logging
from pathlib import Path

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from export_system.graph_exporter import GraphExporter

def create_test_workflow_with_labels():
    """Create a test workflow with label connections"""
    workflow = {
        "nodes": [
            {
                "id": 1,
                "type": "Tensor",
                "class_type": "Tensor",
                "pos": [100, 100],
                "size": [150, 50],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [{"name": "tensor", "type": "TENSOR", "links": []}],
                "properties": {},
                "widgets_values": [[1, 2, 3], "zeros", 0.0, "float32", 42]  # dims, fill_mode, custom_fill, dtype, seed
            },
            {
                "id": 2,
                "type": "Tensor",
                "class_type": "Tensor", 
                "pos": [100, 200],
                "size": [150, 50],
                "flags": {},
                "order": 1,
                "mode": 0,
                "inputs": [],
                "outputs": [{"name": "tensor", "type": "TENSOR", "links": []}],
                "properties": {},
                "widgets_values": [[10], "ones", 0.0, "int64", 43]  # dims, fill_mode, custom_fill, dtype, seed
            },
            {
                "id": 3,
                "type": "CrossEntropyLoss", 
                "class_type": "CrossEntropyLoss",
                "pos": [400, 150],
                "size": [150, 50],
                "flags": {},
                "order": 2,
                "mode": 0,
                "inputs": [
                    {"name": "predictions", "type": "TENSOR", "link": None},
                    {"name": "labels", "type": "TENSOR", "link": 2}  # Direct connection for labels
                ],
                "outputs": [{"name": "loss", "type": "TENSOR", "links": []}],
                "properties": {},
                "widgets_values": []
            },
            # Note: We don't include Label nodes in the export
            # They are purely UI elements
        ],
        "links": [
            # Direct connection for labels input (not using label system)
            [2, 2, 0, 3, 1, "TENSOR"]  # TensorNode(2) -> CrossEntropyLoss labels input
        ],
        "extra": {
            "labelDictionary": {
                "TensorNode(1).tensor": {
                    "nodeId": 1,
                    "slotName": "tensor",
                    "slotType": "TENSOR",
                    "direction": "output",
                    "anchorNodeId": 10
                },
                "CrossEntropyLoss(3).predictions": {
                    "nodeId": 3,
                    "slotName": "predictions",
                    "slotType": "TENSOR",
                    "direction": "input",
                    "anchorNodeId": 11,
                    "connectedToLabel": "TensorNode(1).tensor"
                }
            }
        },
        "metadata": {
            "workflow_name": "test_labels",
            "skip_slot_correction": True
        }
    }
    
    return workflow

def test_label_export():
    """Test exporting a workflow with label connections"""
    print("Creating test workflow with labels...")
    workflow = create_test_workflow_with_labels()
    
    print("\nLabel dictionary:")
    for label_name, label_data in workflow["extra"]["labelDictionary"].items():
        print(f"  {label_name}: {label_data}")
    
    print("\nInitializing exporter...")
    exporter = GraphExporter()
    
    print("\nGenerating label connections...")
    # Debug: Check if exporters are registered
    print(f"Registered node types: {list(exporter.node_registry.keys())[:10]}...")
    
    label_connections = exporter.generate_label_connections(workflow)
    
    print(f"\nFound {len(label_connections)} label connections:")
    for conn in label_connections:
        from_node, from_slot, to_node, to_slot = conn
        print(f"  Node {from_node}[{from_slot}] -> Node {to_node}[{to_slot}]")
    
    print("\nChecking workflow_labels global:")
    from export_system.graph_exporter import workflow_labels
    for key, value in workflow_labels.items():
        print(f"  {key}: {value}")
    
    # Try exporting the workflow
    print("\nAttempting export...")
    try:
        output_path = Path(__file__).parent.parent / "export_system" / "exports" / "test_labels"
        result = exporter.export_workflow(workflow, output_path)
        print(f"Export successful: {result}")
        
        # Check if connections were added to the exported code
        runner_file = output_path / "runner.py"
        if runner_file.exists():
            with open(runner_file, 'r') as f:
                content = f.read()
                if "wire_nodes" in content:
                    print("\nConnections found in runner.py!")
                    # Extract the wire_nodes section
                    import re
                    match = re.search(r'wire_nodes\(\[(.*?)\]\)', content, re.DOTALL)
                    if match:
                        connections_str = match.group(1)
                        print(f"Connections: {connections_str[:200]}...")
                
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_label_export()