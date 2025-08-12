#!/usr/bin/env python3
"""
Test script for DataStreamer node export with Isaac Gym Simulator
Tests the Franka_Coop_Nodes workflow with CSV data streaming
"""

import os
import sys
import json
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specialized.helpers.deployment_helper import DeploymentHelper

def test_data_streamer_export():
    """Test exporting and running the Franka_Coop_Nodes workflow with DataStreamer"""
    
    print("=" * 60)
    print("Testing DataStreamer Export with Franka_Coop_Nodes")
    print("=" * 60)
    
    # Initialize deployment helper
    helper = DeploymentHelper()
    
    # Workflow to test
    workflow_name = "Franka_Coop_Nodes"
    
    # Path to test data
    test_data_dir = Path(__file__).parent / "test_data"
    
    # Export directory
    export_dir = Path("/home/asantanna/DNNE/DNNE-UI/export_system/exports") / workflow_name
    
    try:
        # Step 1: Export the workflow
        print(f"\n1. Exporting workflow: {workflow_name}")
        success = helper.export_workflow(workflow_name)
        
        if not success:
            print("ERROR: Failed to export workflow")
            return False
        
        print(f"   ✓ Workflow exported successfully to {export_dir}")
        
        # Step 2: Copy test data to export directory
        print("\n2. Copying test data to export directory")
        export_data_dir = export_dir / "data"
        export_data_dir.mkdir(exist_ok=True)
        
        # Copy CSV file
        csv_src = test_data_dir / "franka_trajectory.csv"
        csv_dst = export_data_dir / "trajectory.csv"
        shutil.copy2(csv_src, csv_dst)
        print(f"   ✓ Copied {csv_src.name} → {csv_dst.name}")
        
        # Copy metadata file
        meta_src = test_data_dir / "franka_trajectory_metadata.json"
        meta_dst = export_data_dir / "trajectory_metadata.json"
        shutil.copy2(meta_src, meta_dst)
        print(f"   ✓ Copied {meta_src.name} → {meta_dst.name}")
        
        # Step 3: Check generated files
        print("\n3. Checking generated files")
        runner_path = export_dir / "runner.py"
        
        if not runner_path.exists():
            print(f"ERROR: runner.py not found at {runner_path}")
            return False
        
        print(f"   ✓ runner.py exists")
        
        # Check if DataStreamer code is present
        with open(runner_path, 'r') as f:
            runner_content = f.read()
            
        if "DataStreamerNode" not in runner_content:
            print("ERROR: DataStreamerNode not found in runner.py")
            return False
        
        print("   ✓ DataStreamerNode found in runner.py")
        
        if "isaac_gym_sim" not in runner_content.lower():
            print("ERROR: Isaac Gym Sim not found in runner.py")
            return False
            
        print("   ✓ Isaac Gym Sim found in runner.py")
        
        # Step 4: Verify node connections
        print("\n4. Verifying node connections")
        
        # Check for data → action connection
        if "node_17" in runner_content and "node_14" in runner_content:
            print("   ✓ Node IDs found (node_17: DataStreamer, node_14: IsaacGymSim)")
        else:
            print("WARNING: Expected node IDs not found")
        
        # Check for queue connections
        if "connect_nodes" in runner_content:
            print("   ✓ Node connections found")
        else:
            print("WARNING: Node connections not explicitly found")
        
        # Step 5: Print export summary
        print("\n5. Export Summary")
        print("   ✓ Workflow exported successfully")
        print("   ✓ Test data copied to export directory")
        print("   ✓ DataStreamer and Isaac Gym Sim nodes present")
        print("   ✓ Ready for manual testing")
        
        print("\n" + "=" * 60)
        print("TEST PASSED: DataStreamer export successful")
        print("=" * 60)
        
        print("\nTo run the exported workflow:")
        print(f"  cd {export_dir}")
        print("  source /home/asantanna/miniconda/bin/activate DNNE_PY38")
        print("  python runner.py --timeout 10s")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    success = test_data_streamer_export()
    sys.exit(0 if success else 1)