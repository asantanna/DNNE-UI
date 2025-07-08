#!/usr/bin/env python3
"""
Test script for PPO workflow export functionality
"""

import sys
import os
import json

# Add the project root to the Python path
sys.path.insert(0, '/mnt/e/ALS-Projects/DNNE/DNNE-UI')

def test_ppo_export():
    """Test the PPO workflow export"""
    try:
        # Activate conda environment and import necessary modules
        print("🔄 Loading export system...")
        from export_system.graph_exporter import GraphExporter
        
        # Load the PPO workflow
        workflow_path = "/mnt/e/ALS-Projects/DNNE/DNNE-UI/user/default/workflows/Cartpole_PPO.json"
        print(f"📁 Loading workflow: {workflow_path}")
        
        with open(workflow_path, 'r') as f:
            workflow_data = json.load(f)
            
        print(f"✅ Workflow loaded successfully")
        print(f"   - Nodes: {len(workflow_data['nodes'])}")
        print(f"   - Links: {len(workflow_data['links'])}")
        
        # List the node types
        node_types = [node['type'] for node in workflow_data['nodes']]
        print(f"   - Node types: {node_types}")
        
        # Create exporter
        exporter = GraphExporter()
        
        # Test export
        export_name = "PPO_Cartpole_Test"
        print(f"\n🚀 Exporting workflow as: {export_name}")
        
        success = exporter.export_workflow(workflow_data, export_name)
        
        if success:
            print("✅ Export completed successfully!")
            
            # Check the exported files
            export_dir = f"/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/{export_name}"
            print(f"📂 Checking export directory: {export_dir}")
            
            if os.path.exists(export_dir):
                files = os.listdir(export_dir)
                print(f"   Generated files: {files}")
                
                # Check for runner.py
                runner_path = os.path.join(export_dir, "runner.py")
                if os.path.exists(runner_path):
                    print("✅ runner.py generated")
                    
                    # Show first few lines of runner.py
                    with open(runner_path, 'r') as f:
                        lines = f.readlines()[:20]
                        print("\n📝 First 20 lines of runner.py:")
                        for i, line in enumerate(lines, 1):
                            print(f"   {i:2d}: {line.rstrip()}")
                else:
                    print("❌ runner.py not found")
            else:
                print("❌ Export directory not created")
        else:
            print("❌ Export failed")
            
    except Exception as e:
        print(f"❌ Error during export test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return success

if __name__ == "__main__":
    print("🧪 Testing PPO Export Functionality")
    print("=" * 50)
    
    success = test_ppo_export()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 PPO export test PASSED!")
    else:
        print("💥 PPO export test FAILED!")
        sys.exit(1)