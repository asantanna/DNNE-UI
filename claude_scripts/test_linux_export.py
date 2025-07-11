#!/usr/bin/env python3
"""Test export with Linux repository setup"""

import os
import sys
import json

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

from export_system.graph_exporter import GraphExporter

def test_cartpole_export():
    """Test Cartpole export with Linux repository paths"""
    
    # Load workflow
    workflow_path = os.path.join(os.path.dirname(__file__), '..', 'user', 'default', 'workflows', 'Cartpole_PPO.json')
    if not os.path.exists(workflow_path):
        print(f"❌ Workflow not found: {workflow_path}")
        return False
        
    with open(workflow_path, 'r') as f:
        workflow_data = json.load(f)
    
    # Set output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'export_system', 'exports', 'Cartpole_PPO_LinuxRepo')
    
    # Create exporter and export
    exporter = GraphExporter()
    try:
        result = exporter.export_workflow(workflow_data, output_dir)
        print(f"✅ Export successful to: {output_dir}")
        return True
            
    except Exception as e:
        import traceback
        print(f"❌ Export exception: {e}")
        print(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_cartpole_export()
    sys.exit(0 if success else 1)