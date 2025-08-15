#!/usr/bin/env python3
"""
Test script for export system with more nodes
"""

# Use relative imports when run as module
from .graph_exporter import GraphExporter
from .node_exporters import register_all_exporters

def test_export_with_linear():
    # Create exporter
    exporter = GraphExporter()
    register_all_exporters(exporter)
    
    # Test workflow with linear layer
    test_workflow = {
        "nodes": [
            {
                "id": "1",
                "class_type": "MNISTDataset",
                "inputs": {
                    "data_path": "./data",
                    "train": True,
                    "download": True
                }
            },
            {
                "id": "2", 
                "class_type": "BatchSampler",
                "inputs": {
                    "batch_size": 32,
                    "shuffle": True
                }
            },
            {
                "id": "3",
                "class_type": "LinearLayer",
                "inputs": {
                    "output_size": 128,
                    "activation": "relu",
                    "dropout": 0.5
                }
            },
            {
                "id": "4",
                "class_type": "LinearLayer", 
                "inputs": {
                    "output_size": 10,
                    "activation": "none",
                    "dropout": 0.0
                }
            },
            {
                "id": "5",
                "class_type": "SGDOptimizer",
                "inputs": {
                    "learning_rate": 0.01,
                    "momentum": 0.9
                }
            }
        ],
        "links": [
            [1, "1", 0, "2", 0],     # Dataset -> Sampler
            [2, "2", 0, "3", 0],     # Sampler -> Linear1 (conceptual)
            [3, "3", 0, "4", 0],     # Linear1 -> Linear2
        ]
    }
    
    # Export
    script = exporter.export_workflow(test_workflow)
    
    # Save to file
    output_path = "export_system/test_linear_output.py"
    with open(output_path, "w") as f:
        f.write(script)
    
    print(f"Exported to {output_path}")
    print("\nGenerated script preview:")
    print("-" * 80)
    # Show last part of script
    lines = script.split('\n')
    for line in lines[-20:]:
        print(line)

if __name__ == "__main__":
    test_export_with_linear()