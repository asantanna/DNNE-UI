#!/usr/bin/env python3
"""
Test basic node simulators to verify they work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from node_simulators import (
    BarrierNodeSimulator,
    EatNNodeSimulator,
    ConcatNodeSimulator,
    SGDOptimizerSimulator,
    IsaacGymSimulator,
    NetworkNodeSimulator,
    create_simulator
)

def test_barrier_node():
    """Test barrier holds data until triggered"""
    print("\n=== Testing Barrier Node ===")
    barrier = BarrierNodeSimulator('barrier_1', {'class': 'BarrierNode_1'})
    
    # Send data - should not be ready
    barrier.process_input('input', {'data': 'test_data'}, timestamp=1.0)
    assert not barrier.can_execute(), "Barrier should not execute without trigger"
    print("✓ Barrier correctly holds data without trigger")
    
    # Send trigger - should now be ready
    barrier.process_input('release', {'signal': 'trigger'}, timestamp=2.0)
    assert barrier.can_execute(), "Barrier should execute with data and trigger"
    print("✓ Barrier ready after receiving trigger")
    
    # Execute and check output
    output = barrier.execute()
    assert 'output' in output, "Barrier should produce output"
    assert output['output']['data'] == 'test_data', "Barrier should output held data"
    print("✓ Barrier correctly releases held data")
    
    # After execution, should reset
    barrier.post_execute()
    assert not barrier.can_execute(), "Barrier should reset after execution"
    print("✓ Barrier resets correctly after execution")

def test_eat_n_node():
    """Test Eat_N consumes then passes through"""
    print("\n=== Testing Eat_N Node ===")
    eat_n = EatNNodeSimulator('eat_n_1', {'class': 'Eat_NNode_1', 'n': 2})
    
    # First input - consume, no output
    eat_n.process_input('input', {'data': 'first'}, timestamp=1.0)
    assert eat_n.can_execute(), "Eat_N should be ready to consume"
    output = eat_n.execute()
    assert output == {}, "Eat_N should not output on first consumption"
    print(f"✓ Eat_N consumed 1/2 inputs without output")
    
    # Reset for second input
    eat_n.post_execute()
    
    # Second input - consume and switch to passthrough
    eat_n.process_input('input', {'data': 'second'}, timestamp=2.0)
    output = eat_n.execute()
    assert 'output' in output, "Eat_N should output when switching modes"
    assert 'trigger' in output, "Eat_N should send trigger when satisfied"
    print(f"✓ Eat_N consumed 2/2 inputs and switched to passthrough")
    
    # Reset for third input
    eat_n.post_execute()
    
    # Third input - passthrough mode
    eat_n.process_input('input', {'data': 'third'}, timestamp=3.0)
    output = eat_n.execute()
    assert 'output' in output, "Eat_N should output in passthrough"
    assert 'trigger' not in output, "Eat_N should not trigger again"
    print("✓ Eat_N correctly passes through subsequent inputs")

def test_concat_node():
    """Test Concat waits for all inputs"""
    print("\n=== Testing Concat Node ===")
    concat = ConcatNodeSimulator('concat_1', {'class': 'ConcatNode_1'})
    
    # Set expected inputs
    concat.set_expected_inputs({'input_a', 'input_b', 'input_c'})
    
    # Send first input - should not be ready
    concat.process_input('input_a', {'data': 'a'}, timestamp=1.0)
    assert not concat.can_execute(), "Concat should wait for all inputs"
    print(f"✓ Concat waiting (1/3 inputs)")
    
    # Send second input - still not ready
    concat.process_input('input_b', {'data': 'b'}, timestamp=2.0)
    assert not concat.can_execute(), "Concat should wait for all inputs"
    print(f"✓ Concat waiting (2/3 inputs)")
    
    # Send third input - now ready
    concat.process_input('input_c', {'data': 'c'}, timestamp=3.0)
    assert concat.can_execute(), "Concat should execute with all inputs"
    print(f"✓ Concat ready with all inputs")
    
    # Execute
    output = concat.execute()
    assert 'output' in output, "Concat should produce output"
    assert output['output']['count'] == 3, "Concat should combine all inputs"
    print("✓ Concat successfully concatenated 3 inputs")

def test_sgd_optimizer():
    """Test SGD optimizer with bootstrap"""
    print("\n=== Testing SGD Optimizer ===")
    
    # Test with bootstrap enabled
    sgd = SGDOptimizerSimulator('sgd_1', {
        'class': 'SGDOptimizerNode_1',
        'bootstrap': True,
        'no_bootstrap_trigger': False
    })
    
    # Check bootstrap capability
    assert sgd.should_bootstrap(), "SGD should be able to bootstrap"
    bootstrap_output = sgd.send_bootstrap()
    assert 'step_complete' in bootstrap_output, "SGD should send bootstrap signal"
    print("✓ SGD sent bootstrap signal")
    
    # Normal execution with loss
    sgd.process_input('loss', {'value': 0.5}, timestamp=1.0)
    assert sgd.can_execute(), "SGD should execute with loss"
    output = sgd.execute()
    assert 'step_complete' in output, "SGD should send step_complete"
    assert output['step_complete']['step'] == 1, "SGD should track steps"
    print("✓ SGD performed optimization step")
    
    # Test with bootstrap disabled
    sgd_no_boot = SGDOptimizerSimulator('sgd_2', {
        'class': 'SGDOptimizerNode_2',
        'no_bootstrap_trigger': True
    })
    assert not sgd_no_boot.should_bootstrap(), "SGD should not bootstrap when disabled"
    print("✓ SGD respects no_bootstrap_trigger flag")

def test_isaac_gym():
    """Test Isaac Gym with null action bootstrap"""
    print("\n=== Testing Isaac Gym Simulator ===")
    isaac = IsaacGymSimulator('isaac_1', {
        'class': 'IsaacGymSimNode_1',
        'num_envs': 4
    })
    
    # Should be able to bootstrap
    assert isaac.should_bootstrap(), "IsaacGym should bootstrap without action"
    assert isaac.can_execute(), "IsaacGym should execute via bootstrap"
    
    # Bootstrap execution
    output = isaac.bootstrap()
    assert 'observation' in output, "IsaacGym should produce observation"
    assert 'done' in output, "IsaacGym should produce done signal"
    assert output['observation']['metadata']['bootstrap'], "Should mark as bootstrap"
    print("✓ IsaacGym bootstrapped with null action")
    
    # Normal execution with action
    isaac.post_execute()
    isaac.process_input('action', {'data': 'action_tensor'}, timestamp=1.0)
    assert isaac.can_execute(), "IsaacGym should execute with action"
    output = isaac.execute()
    assert output['observation']['step'] == 2, "IsaacGym should track steps"
    print("✓ IsaacGym stepped with action")

def test_factory():
    """Test simulator factory"""
    print("\n=== Testing Simulator Factory ===")
    
    # Test barrier creation
    barrier = create_simulator('barrier_74', {'class': 'BarrierNode_74'})
    assert isinstance(barrier, BarrierNodeSimulator), "Factory should create BarrierNodeSimulator"
    print("✓ Factory created correct barrier simulator")
    
    # Test Eat_N creation (special underscore case)
    eat_n = create_simulator('eat_n_73', {'class': 'Eat_NNode_73'})
    assert isinstance(eat_n, EatNNodeSimulator), "Factory should create EatNNodeSimulator"
    print("✓ Factory handled Eat_NNode naming correctly")
    
    # Test unknown node type
    unknown = create_simulator('custom_99', {'class': 'CustomNode_99'})
    assert unknown.__class__.__name__ == 'BaseNodeSimulator', "Factory should use base for unknown"
    print("✓ Factory used base simulator for unknown type")

def run_all_tests():
    """Run all test functions"""
    print("=" * 60)
    print("Running Node Simulator Tests")
    print("=" * 60)
    
    test_functions = [
        test_barrier_node,
        test_eat_n_node,
        test_concat_node,
        test_sgd_optimizer,
        test_isaac_gym,
        test_factory
    ]
    
    failed = []
    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            failed.append((test_func.__name__, str(e)))
            print(f"✗ {test_func.__name__} failed: {e}")
    
    print("\n" + "=" * 60)
    if not failed:
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(failed)} test(s) failed:")
        for name, error in failed:
            print(f"  - {name}: {error}")
    print("=" * 60)
    
    return len(failed) == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)