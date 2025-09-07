#!/usr/bin/env python3
"""
Test script to verify training sequencer integration.
Creates a simple workflow with two networks and a training sequencer.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def create_test_workflow():
    """Create a minimal workflow with training sequencer"""
    
    workflow = {
        "id": "test-training-sequencer",
        "revision": 0,
        "last_node_id": 10,
        "last_link_id": 15,
        "nodes": [
            # Input tensor node
            {
                "id": 1,
                "type": "Tensor",
                "pos": [0, 0],
                "size": [270, 154],
                "widgets_values": ["10, 5", "randn", 0.0, "float32", 42]
            },
            # Network 1
            {
                "id": 2,
                "type": "Network",
                "pos": [300, 0],
                "size": [300, 170],
                "widgets_values": [False, "epoch", "50", False]
            },
            # Network 2
            {
                "id": 3,
                "type": "Network",
                "pos": [300, 200],
                "size": [300, 170],
                "widgets_values": [False, "epoch", "50", False]
            },
            # Linear layers for Network 1
            {
                "id": 4,
                "type": "LinearLayer",
                "pos": [100, 0],
                "size": [270, 154],
                "widgets_values": [3, True, "relu", 0, "auto"]
            },
            # Linear layers for Network 2
            {
                "id": 5,
                "type": "LinearLayer",
                "pos": [100, 200],
                "size": [270, 154],
                "widgets_values": [3, True, "relu", 0, "auto"]
            },
            # Loss computation 1
            {
                "id": 6,
                "type": "CustomComputation",
                "pos": [650, 0],
                "size": [270, 58],
                "widgets_values": ["simple_loss.py"]
            },
            # Loss computation 2
            {
                "id": 7,
                "type": "CustomComputation",
                "pos": [650, 200],
                "size": [270, 58],
                "widgets_values": ["simple_loss.py"]
            },
            # Training Sequencer
            {
                "id": 8,
                "type": "TrainingSequencer",
                "pos": [950, 100],
                "size": [270, 150],
                "widgets_values": ["1,2", True]
            },
            # SGD Optimizer 1
            {
                "id": 9,
                "type": "SGDOptimizer",
                "pos": [1250, 0],
                "size": [270, 150],
                "widgets_values": [0.001, 0.9, 0, False]
            },
            # SGD Optimizer 2
            {
                "id": 10,
                "type": "SGDOptimizer",
                "pos": [1250, 200],
                "size": [270, 150],
                "widgets_values": [0.001, 0.9, 0, False]
            }
        ],
        "links": [
            # Input to layers
            [1, 1, "tensor", 4, 0, "input", "*TENSOR"],
            [2, 1, "tensor", 5, 0, "input", "*TENSOR"],
            # Layers to networks
            [3, 4, "output", 2, 1, "to_output", "LAYER_TENSOR"],
            [4, 5, "output", 3, 1, "to_output", "LAYER_TENSOR"],
            # Input to networks
            [5, 1, "tensor", 2, 0, "input", "*TENSOR"],
            [6, 1, "tensor", 3, 0, "input", "*TENSOR"],
            # Networks to losses
            [7, 2, "output", 6, 0, "input", "NETWORK_OUTPUT_TENSOR"],
            [8, 3, "output", 7, 0, "input", "NETWORK_OUTPUT_TENSOR"],
            # Losses to sequencer
            [9, 6, "output", 8, 0, "loss1", "*LOSS_SCALAR"],
            [10, 7, "output", 8, 1, "loss2", "*LOSS_SCALAR"],
            # Sequencer to optimizers
            [11, 8, "to_opt1", 9, 1, "loss", "*LOSS_SCALAR"],
            [12, 8, "to_opt2", 10, 1, "loss", "*LOSS_SCALAR"],
            # Networks to optimizers (model connections)
            [13, 2, "model", 9, 0, "model", "NETWORK_MODEL_OBJ"],
            [14, 3, "model", 10, 0, "model", "NETWORK_MODEL_OBJ"]
        ],
        "config": {},
        "extra": {},
        "version": 1.0
    }
    
    return workflow

def create_simple_loss():
    """Create a simple loss computation function"""
    loss_code = '''
import torch

def compute(input_tensor):
    """Simple MSE loss against zeros"""
    target = torch.zeros_like(input_tensor)
    return torch.nn.functional.mse_loss(input_tensor, target)
'''
    
    loss_path = Path("/home/asantanna/DNNE/DNNE-UI/user/default/custom_compute_funcs/simple_loss.py")
    loss_path.parent.mkdir(parents=True, exist_ok=True)
    loss_path.write_text(loss_code)
    print(f"Created loss function: {loss_path}")

def main():
    """Test the training sequencer"""
    
    # Create simple loss function
    create_simple_loss()
    
    # Create test workflow
    workflow = create_test_workflow()
    
    # Save workflow
    workflow_path = Path("/home/asantanna/DNNE/DNNE-UI/user/default/workflows/test_training_sequencer.json")
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(workflow_path, 'w') as f:
        json.dump(workflow, f, indent=2)
    
    print(f"Created test workflow: {workflow_path}")
    print("\nWorkflow structure:")
    print("- 2 Networks with Linear layers")
    print("- 2 Loss computations")
    print("- 1 Training Sequencer coordinating both")
    print("- 2 SGD Optimizers")
    print("\nThe sequencer will:")
    print("1. Receive both losses")
    print("2. Execute backward passes in order (1, then 2)")
    print("3. Step both optimizers")
    
    # Now try to export it
    print("\nAttempting to export workflow...")
    
    from export_system.graph_exporter import GraphExporter
    
    # Add skip-slot-correction to workflow metadata
    workflow["metadata"] = {
        "skip-slot-correction": True,
        "workflow_name": "test_training_sequencer"
    }
    
    exporter = GraphExporter()
    try:
        export_dir = exporter.export_workflow(workflow, Path("export_system/exports/test_training_sequencer"))
        print(f"✅ Export successful to: {export_dir}")
        print("\nTo test the exported code:")
        print(f"cd {export_dir}")
        print("python runner.py --epochs 1")
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()