#!/usr/bin/env python3
"""
Test export of Cartpole RL Single workflow
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from export_system.graph_exporter import GraphExporter
from export_system.node_exporters import register_all_exporters

def test_cartpole_nodes():
    """Test that our Cartpole nodes are properly registered"""
    print("Testing Cartpole Node Registration...")
    print("-" * 40)
    
    # Create exporter
    exporter = GraphExporter()
    register_all_exporters(exporter)
    
    print(f"✓ Loaded export system with {len(exporter.node_registry)} node types")
    
    # Check if our new nodes are registered
    required_nodes = ["CartpoleActionNode", "CartpoleRewardNode"]
    missing_nodes = []
    
    for node_type in required_nodes:
        if node_type in exporter.node_registry:
            print(f"✓ {node_type} is registered")
        else:
            print(f"✗ {node_type} is NOT registered")
            missing_nodes.append(node_type)
    
    if missing_nodes:
        print(f"\n✗ Missing nodes: {missing_nodes}")
        return False
    else:
        print("\n✓ All Cartpole nodes are properly registered!")
        return True

def test_cartpole_export():
    """Test export of Cartpole RL workflow"""
    print("\nTesting Cartpole Workflow Export...")
    print("-" * 40)
    
    # Create exporter
    exporter = GraphExporter()
    register_all_exporters(exporter)
    
    # Load Cartpole workflow
    workflow_path = Path("user/default/workflows/Cartpole_RL_Single.json")
    if not workflow_path.exists():
        print(f"✗ Workflow not found: {workflow_path}")
        return False
    
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    print(f"✓ Loaded workflow with {len(workflow.get('nodes', []))} nodes")
    
    # List nodes for verification
    nodes = workflow.get('nodes', [])
    node_types = [node.get('type') for node in nodes]
    print(f"  Node types: {', '.join(node_types)}")
    
    # Export the workflow
    try:
        output_path = Path("export_system/exports/Cartpole-RL-Test")
        result = exporter.export_workflow(workflow, output_path)
        print("✓ Export successful!")
        print(f"✓ Generated package at: {output_path}")
        
        # Check if runner.py was created
        runner_path = output_path / "runner.py"
        if runner_path.exists():
            print("✓ runner.py generated successfully")
            
            # Check if our nodes are in the generated code
            with open(runner_path, 'r') as f:
                runner_content = f.read()
                
            cartpole_nodes_found = []
            if "CartpoleActionNode" in runner_content:
                cartpole_nodes_found.append("CartpoleActionNode")
            if "CartpoleRewardNode" in runner_content:
                cartpole_nodes_found.append("CartpoleRewardNode")
                
            if cartpole_nodes_found:
                print(f"✓ Found Cartpole nodes in generated code: {', '.join(cartpole_nodes_found)}")
                return True
            else:
                print("✗ Cartpole nodes not found in generated code")
                return False
        else:
            print("✗ runner.py not found in generated output")
            return False
            
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("DNNE Cartpole Export Test")
    print("=" * 60)
    
    success = True
    
    # Test 1: Node registration
    if not test_cartpole_nodes():
        success = False
    
    # Test 2: Workflow export
    if not test_cartpole_export():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("The Cartpole RL system is ready for use.")
    else:
        print("❌ Some tests failed.")
        print("Please check the implementation and try again.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)