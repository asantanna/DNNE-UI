#!/usr/bin/env python3
"""
Test script for Eat_N node export functionality
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from export_system.graph_exporter import GraphExporter

def create_test_workflow():
    """Create a minimal workflow with just Eat_N and Barrier nodes"""
    workflow = {
        "nodes": [
            {
                "id": "1", 
                "type": "Eat_N",
                "pos": [300, 100],
                "inputs": {},  # No input connection for this test
                "outputs": {
                    "output": {"connections": [{"node": "2", "input": "input"}]},
                    "trigger": {"connections": [{"node": "2", "input": "release"}]}
                },
                "widget_values": [2, "every_eat"]
            },
            {
                "id": "2",
                "type": "Barrier",
                "pos": [500, 100],
                "inputs": {
                    "input": {"connection": {"node": "1", "output": "output"}},
                    "release": {"connection": {"node": "1", "output": "trigger"}}
                },
                "outputs": {"output": {"connections": []}},
                "widget_values": ["FIFO"]
            }
        ],
        "links": [
            ["1", "output", "2", "input"],
            ["1", "trigger", "2", "release"]
        ],
        "metadata": {
            "workflow_name": "test_eat_n",
            "skip-slot-correction": True  # Skip slot correction for programmatic workflows
        }
    }
    return workflow

def test_eat_n_export():
    """Test exporting a workflow with Eat_N node"""
    print("Testing Eat_N node export...")
    
    # Create test workflow
    workflow = create_test_workflow()
    
    # Initialize exporter
    exporter = GraphExporter()
    
    try:
        # Export the workflow
        output_dir = Path("export_system/exports/test_eat_n")
        result = exporter.export_workflow(
            workflow,
            output_path=output_dir
        )
        
        if result:
            print(f"✓ Export successful to {output_dir}")
            
            # Check if files were created
            runner_file = output_dir / "runner.py"
            if runner_file.exists():
                print(f"✓ Runner file created: {runner_file}")
                
                # Check if Eat_N node code is in the file
                content = runner_file.read_text()
                if "Eat_NNode" in content:
                    print("✓ Eat_N node code found in runner.py")
                if "trigger_mode" in content:
                    print("✓ Trigger mode configuration found")
                if "num_to_eat" in content:
                    print("✓ num_to_eat configuration found")
                    
                return True
            else:
                print(f"✗ Runner file not created")
                return False
        else:
            print("✗ Export failed")
            return False
            
    except Exception as e:
        print(f"✗ Error during export: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_eat_n_export()
    sys.exit(0 if success else 1)