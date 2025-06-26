#!/usr/bin/env python3
"""
Simple test for queue export system
Save this in your DNNE-UI root directory and run it
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import
from export_system.graph_exporter import GraphExporter
from export_system.node_exporters import register_all_exporters

def main():
    print("Testing DNNE Queue Export System...")
    print("=" * 60)
    
    # Create exporter
    exporter = GraphExporter()
    register_all_exporters(exporter)
    
    print(f"✅ Loaded export system")
    print(f"✅ Registered {len(exporter.node_registry)} node types:")
    for node_type in sorted(exporter.node_registry.keys()):
        print(f"   - {node_type}")
    
    # Simple test workflow
    print("\n" + "=" * 60)
    print("Creating simple test workflow...")
    
    workflow = {
        "nodes": [
            {
                "id": "data",
                "class_type": "MNISTDataset",
                "inputs": {
                    "batch_size": 32,
                    "emit_rate": 2.0  # 2 batches per second
                }
            },
            {
                "id": "display",
                "class_type": "Display",
                "inputs": {
                    "display_type": "tensor_stats",
                    "log_interval": 1
                }
            }
        ],
        "links": [
            [1, "data", 0, "display", 0]  # Connect MNIST data output to display input
        ],
        "metadata": {
            "name": "Simple MNIST Display Test"
        }
    }
    
    # Export the workflow
    try:
        script = exporter.export_workflow(workflow)
        print("✅ Export successful!")
        
        # Save the script
        output_file = "test_generated_queue.py"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(script)
        
        print(f"✅ Saved to {output_file}")
        print(f"   Generated {len(script.splitlines())} lines of code")
        
        # Show first few lines
        print("\nFirst 30 lines of generated code:")
        print("-" * 60)
        lines = script.splitlines()
        for i, line in enumerate(lines[:30]):
            print(f"{i+1:3d}: {line}")
        print("-" * 60)
        
        print(f"\n✅ Test complete! You can now run:")
        print(f"   python {output_file}")
        print("   (Press Ctrl+C to stop it)")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()