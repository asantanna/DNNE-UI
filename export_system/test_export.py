#!/usr/bin/env python3
"""
Test script for the export system
"""

# Use relative imports when run as module
from .graph_exporter import GraphExporter
from .node_exporters import register_all_exporters

def test_export():
    # Create exporter
    exporter = GraphExporter()
    register_all_exporters(exporter)
    
    # Test workflow (minimal MNIST)
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
            }
        ],
        "links": [
            [1, "1", 0, "2", 0]  # Dataset -> Sampler
        ]
    }
    
    # Export
    script = exporter.export_workflow(test_workflow)
    print("Generated script:")
    print("-" * 80)
    print(script)
    print("-" * 80)
    
    # Save to file
    output_path = "export_system/test_export_output.py"
    with open(output_path, "w") as f:
        f.write(script)
    print(f"\nSaved to {output_path}")

if __name__ == "__main__":
    test_export()