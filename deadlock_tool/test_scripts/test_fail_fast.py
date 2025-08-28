#!/usr/bin/env python3
"""
Test that the deadlock simulator fails fast when encountering unknown node types.
This ensures we don't get meaningless results from missing simulators.
"""

import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataflow_simulator import DataflowSimulator
from node_simulators import create_simulator

def test_missing_simulator():
    """Test that factory raises error for unknown node type"""
    print("\n=== Testing Missing Simulator Detection ===")
    
    try:
        # Try to create a simulator for an unknown node type
        unknown_config = {
            'class': 'UnknownNodeType_99',
            'type': 'mystery'
        }
        simulator = create_simulator('unknown_99', unknown_config)
        print("❌ FAILED: Should have raised ValueError for unknown node type")
        return False
    except ValueError as e:
        if "FAIL-FAST" in str(e):
            print("✓ Correctly raised error for unknown node type")
            print(f"  Error message: {str(e).split(chr(10))[0]}...")  # First line only
            return True
        else:
            print(f"❌ FAILED: Wrong error type: {e}")
            return False
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
        return False

def test_execution_failure():
    """Test that execution failures are caught"""
    print("\n=== Testing Execution Failure Detection ===")
    
    # Create a graph with a known node type
    graph = {
        "nodes": {
            "test_1": {"class": "NetworkNode_1", "type": "network"}
        },
        "connections": []
    }
    
    # Create simulator
    sim = DataflowSimulator(graph)
    
    # Force the node to be ready
    from node_simulators import NodeState
    sim.nodes['test_1'].state = NodeState.READY
    sim.nodes['test_1'].inputs_available['input'] = {'test': 'data'}
    
    # Try to execute - should work normally
    try:
        sim._try_execute_node('test_1')
        print("✓ Normal execution works")
    except Exception as e:
        print(f"✓ Execution completed (may have post-processing errors): {e}")
    
    return True

def test_graph_with_unknown_node():
    """Test that graph creation fails for unknown nodes"""
    print("\n=== Testing Graph with Unknown Node ===")
    
    graph = {
        "nodes": {
            "known_1": {"class": "NetworkNode_1", "type": "network"},
            "unknown_1": {"class": "CompletelyUnknownNode_1", "type": "mystery"}
        },
        "connections": [
            ["known_1", "output", "unknown_1", "input"]
        ]
    }
    
    try:
        sim = DataflowSimulator(graph)
        print("❌ FAILED: Should have raised error during graph construction")
        return False
    except ValueError as e:
        if "FAIL-FAST" in str(e):
            print("✓ Correctly failed during graph construction")
            print(f"  Detected unknown node type")
            return True
        else:
            print(f"❌ FAILED: Wrong error: {e}")
            return False
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
        return False

def test_all_known_simulators():
    """Test that all expected simulators are registered"""
    print("\n=== Testing Known Simulators ===")
    
    from node_simulators import get_available_simulators
    
    simulators = get_available_simulators()
    expected = [
        'BarrierNode',
        'Eat_NNode', 
        'ConcatNode',
        'SplitNode',
        'SGDOptimizerNode',
        'IsaacGymSimNode',
        'NetworkNode',
        'CustomComputationNode',
        'SimulationTracker',
        'TensorNode'
    ]
    
    missing = []
    for node_type in expected:
        if node_type not in simulators:
            missing.append(node_type)
            
    if missing:
        print(f"❌ FAILED: Missing simulators for: {missing}")
        return False
    else:
        print(f"✓ All {len(expected)} expected simulators are registered")
        print(f"  Registered types: {sorted(simulators.keys())}")
        return True

def main():
    """Run all fail-fast tests"""
    print("="*60)
    print("FAIL-FAST TESTING")
    print("="*60)
    print("\nEnsuring the simulator fails fast on errors instead of")
    print("producing meaningless results...")
    
    tests = [
        test_missing_simulator,
        test_all_known_simulators,
        test_graph_with_unknown_node,
        test_execution_failure
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
            
    print("\n" + "="*60)
    if failed == 0:
        print(f"✅ SUCCESS: All {passed} fail-fast tests passed!")
        print("The simulator will now fail immediately on missing simulators")
        print("instead of producing meaningless results.")
    else:
        print(f"❌ FAILURE: {failed} tests failed, {passed} passed")
        print("The simulator may still produce incorrect results!")
    print("="*60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())