#!/usr/bin/env python3
"""
Test script to verify SGD optimizer exporter fixes:
1. Properly traces network connections (not assuming single network)
2. Fails fast when no network is connected (no default fallback)
"""

import sys
import json
from pathlib import Path

# Add export system to path
sys.path.append(str(Path(__file__).parent.parent))

from export_system.node_exporters.sgd_optimizer_exporter import SGDOptimizerExporter

def test_multiple_networks():
    """Test that SGD optimizer correctly identifies its connected network when multiple exist"""
    print("Testing: Multiple networks in workflow")
    print("-" * 40)
    
    # Simulate workflow with 2 networks and 2 optimizers
    all_nodes = [
        {"id": 10, "class_type": "Network"},
        {"id": 20, "class_type": "Network"},
        {"id": 30, "class_type": "SGDOptimizer"},
        {"id": 40, "class_type": "SGDOptimizer"},
    ]
    
    # SGD optimizer 30 connected to Network 10
    node_data_30 = {
        "widgets_values": [0.01, 0.9, 0.0, True]  # lr, momentum, weight_decay, bootstrap
    }
    connections_30 = {
        "inputs": {
            "model": [{"from_node": "10", "from_slot": 2}],  # Connected to Network 10
            "loss": [{"from_node": "50", "from_slot": 0}]
        }
    }
    
    # SGD optimizer 40 connected to Network 20
    node_data_40 = {
        "widgets_values": [0.001, 0.95, 0.0001, False]
    }
    connections_40 = {
        "inputs": {
            "model": [{"from_node": "20", "from_slot": 2}],  # Connected to Network 20
            "loss": [{"from_node": "60", "from_slot": 0}]
        }
    }
    
    # Test optimizer 30
    vars_30 = SGDOptimizerExporter.prepare_template_vars(
        "30", node_data_30, connections_30, all_nodes=all_nodes
    )
    
    assert vars_30["NETWORK_NODE_ID"] == "10", f"Expected network 10, got {vars_30['NETWORK_NODE_ID']}"
    print(f"✓ Optimizer 30 correctly identified its network: {vars_30['NETWORK_NODE_ID']}")
    
    # Test optimizer 40
    vars_40 = SGDOptimizerExporter.prepare_template_vars(
        "40", node_data_40, connections_40, all_nodes=all_nodes
    )
    
    assert vars_40["NETWORK_NODE_ID"] == "20", f"Expected network 20, got {vars_40['NETWORK_NODE_ID']}"
    print(f"✓ Optimizer 40 correctly identified its network: {vars_40['NETWORK_NODE_ID']}")
    
    print("✅ Multiple networks test PASSED\n")
    return True

def test_no_network_fail_fast():
    """Test that SGD optimizer fails fast when no network is connected"""
    print("Testing: Fail-fast when no network connected")
    print("-" * 40)
    
    node_data = {
        "widgets_values": [0.01, 0.9, 0.0, True]
    }
    
    # No model connection
    connections = {
        "inputs": {
            "loss": [{"from_node": "50", "from_slot": 0}]
            # Note: no "model" connection!
        }
    }
    
    try:
        vars = SGDOptimizerExporter.prepare_template_vars(
            "30", node_data, connections, all_nodes=[]
        )
        print("❌ Should have raised ValueError but didn't!")
        return False
    except ValueError as e:
        expected_msg = "No Network node connected to 'model' input"
        if expected_msg in str(e):
            print(f"✓ Correctly raised ValueError: {e}")
            print("✅ Fail-fast test PASSED\n")
            return True
        else:
            print(f"❌ Wrong error message: {e}")
            return False

def test_wrong_node_type_connected():
    """Test that SGD optimizer rejects non-Network nodes on model input"""
    print("Testing: Reject non-Network node on model input")
    print("-" * 40)
    
    all_nodes = [
        {"id": 10, "class_type": "LinearLayer"},  # Wrong type!
        {"id": 30, "class_type": "SGDOptimizer"},
    ]
    
    node_data = {
        "widgets_values": [0.01, 0.9, 0.0, True]
    }
    
    connections = {
        "inputs": {
            "model": [{"from_node": "10", "from_slot": 0}],  # Connected to LinearLayer
            "loss": [{"from_node": "50", "from_slot": 0}]
        }
    }
    
    try:
        vars = SGDOptimizerExporter.prepare_template_vars(
            "30", node_data, connections, all_nodes=all_nodes
        )
        print("❌ Should have raised ValueError but didn't!")
        return False
    except ValueError as e:
        expected_msg = "but expected a Network node"
        if expected_msg in str(e):
            print(f"✓ Correctly rejected non-Network node: {e}")
            print("✅ Wrong node type test PASSED\n")
            return True
        else:
            print(f"❌ Wrong error message: {e}")
            return False

def main():
    print("=" * 60)
    print("Testing SGD Optimizer Exporter Fixes")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # Test 1: Multiple networks
    if not test_multiple_networks():
        all_passed = False
    
    # Test 2: No network fails fast
    if not test_no_network_fail_fast():
        all_passed = False
    
    # Test 3: Wrong node type
    if not test_wrong_node_type_connected():
        all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nThe SGD optimizer exporter now:")
        print("1. Correctly traces connections to find its specific network")
        print("2. Fails fast when no network is connected (no defaults)")
        print("3. Validates that connected node is actually a Network")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())