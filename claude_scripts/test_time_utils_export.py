#!/usr/bin/env python3
"""
Test that time_utils.py is properly exported with workflows.
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

sys.path.append('/home/asantanna/DNNE/DNNE-UI')

from export_system.graph_exporter import GraphExporter


def create_test_workflow():
    """Create a minimal workflow with SimulationTracker to test export."""
    return {
        "workflow_name": "test_time_utils",
        "last_node_id": 2,
        "last_link_id": 1,
        "nodes": [
            {
                "id": 1,
                "type": "SimulationTracker",
                "pos": [100, 100],
                "size": [200, 100],
                "inputs": [
                    {"name": "observation", "type": "*SIM_OBSERVATION_TENSOR", "link": None},
                    {"name": "done", "type": "*TRIGGER", "link": None}
                ],
                "outputs": [
                    {"name": "control_metrics", "type": "CONTROL_METRICS_PYDICT", "links": []}
                ],
                "properties": {},
                "widgets_values": [1000, 0.95, "time", "10s", True]
            }
        ],
        "links": [],
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4
    }


def test_time_utils_export():
    """Test that time_utils.py is included in the export."""
    print("\n=== Testing time_utils.py Export ===\n")
    
    # Create temporary directory for export
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = Path(temp_dir) / "test_export"
        
        # Create and save test workflow
        workflow = create_test_workflow()
        workflow_file = Path(temp_dir) / "test_workflow.json"
        workflow_file.write_text(json.dumps(workflow, indent=2))
        
        print(f"  Created test workflow: {workflow_file}")
        
        # Export the workflow
        try:
            exporter = GraphExporter()
            # Pass the workflow dict directly, not the file path
            exporter.export_workflow(workflow, export_path)
            print(f"  ✓ Export successful to: {export_path}")
        except Exception as e:
            print(f"  ✗ Export failed: {e}")
            return False
        
        # Check if time_utils.py was exported
        time_utils_path = export_path / "framework" / "time_utils.py"
        if time_utils_path.exists():
            print(f"  ✓ time_utils.py exported to framework/")
            
            # Verify content
            content = time_utils_path.read_text()
            if "def parse_duration" in content:
                print(f"  ✓ parse_duration function found in exported file")
            else:
                print(f"  ✗ parse_duration function not found in exported file")
                return False
        else:
            print(f"  ✗ time_utils.py NOT found in framework/")
            return False
        
        # Check if SimulationTracker imports it correctly
        sim_tracker_files = list(export_path.glob("nodes/simulationtracker*.py"))
        if sim_tracker_files:
            sim_file = sim_tracker_files[0]
            content = sim_file.read_text()
            if "from framework.time_utils import parse_duration" in content:
                print(f"  ✓ SimulationTracker correctly imports time_utils")
            else:
                print(f"  ✗ SimulationTracker missing time_utils import")
                return False
        else:
            print(f"  ⚠ No SimulationTracker node file found")
        
        # List all framework files
        print("\n  Framework files exported:")
        framework_dir = export_path / "framework"
        if framework_dir.exists():
            for file in sorted(framework_dir.glob("*.py")):
                print(f"    - {file.name}")
        
        return True


def main():
    """Run the test."""
    print("\n" + "="*60)
    print("Time Utils Export Test")
    print("="*60)
    
    success = test_time_utils_export()
    
    if success:
        print("\n✅ Test PASSED: time_utils.py is properly exported")
    else:
        print("\n❌ Test FAILED: time_utils.py export issue detected")
    
    print("="*60 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())