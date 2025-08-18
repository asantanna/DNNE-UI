#!/usr/bin/env python3
"""
Test suite for Custom Computation node with dynamic types.
Tests the script loader utility and dynamic type resolution.
"""

import sys
import os
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from custom_nodes.utils.script_loader import load_custom_script, clear_script_cache

def test_script_loading():
    """Test that scripts can be loaded and validated"""
    print("Testing script loading...")
    
    # Test loading example_identity.py
    try:
        module = load_custom_script("example_identity.py")
        print(f"✓ Loaded example_identity.py")
        print(f"  - Output type: {module.get_output_type()}")
        
        # Test initial schema
        initial_schema = module.get_script_output_schema(initial=True)
        print(f"  - Initial schema: {initial_schema['outputs']['output']['type']}")
        
        # Test resolved schema
        test_input_schema = {
            "input": {
                "type": "tensor",
                "shape": [10, 20],
                "flattened_size": 200,
                "dtype": "float32"
            }
        }
        resolved_schema = module.get_script_output_schema(initial=False, input_schema=test_input_schema)
        print(f"  - Resolved schema: flattened_size={resolved_schema['outputs']['output']['flattened_size']}")
        
    except Exception as e:
        print(f"✗ Failed to load example_identity.py: {e}")
        return False
    
    # Test loading example_filter.py
    try:
        module = load_custom_script("example_filter.py")
        print(f"✓ Loaded example_filter.py")
        print(f"  - Output type: {module.get_output_type()}")
    except Exception as e:
        print(f"✗ Failed to load example_filter.py: {e}")
        return False
    
    # Test loading example_sink.py
    try:
        module = load_custom_script("example_sink.py")
        print(f"✓ Loaded example_sink.py")
        print(f"  - Output type: {module.get_output_type()}")
    except Exception as e:
        print(f"✗ Failed to load example_sink.py: {e}")
        return False
    
    # Test loading franks_coop_nodes_loss.py
    try:
        module = load_custom_script("franks_coop_nodes_loss.py")
        print(f"✓ Loaded franks_coop_nodes_loss.py")
        print(f"  - Output type: {module.get_output_type()}")
    except Exception as e:
        print(f"✗ Failed to load franks_coop_nodes_loss.py: {e}")
        return False
    
    return True

def test_cache_functionality():
    """Test script cache functionality"""
    print("\nTesting cache...")
    
    # Test that cache returns same module
    module1 = load_custom_script("example_identity.py")
    module2 = load_custom_script("example_identity.py")
    if module1 is module2:
        print("✓ Cache working - same module returned")
    else:
        print("✗ Cache not working - different modules")
        return False
    
    # Test cache clearing
    clear_script_cache()
    module3 = load_custom_script("example_identity.py")
    if module1 is not module3:
        print("✓ Cache cleared successfully")
    else:
        print("✗ Cache clear failed")
        return False
    
    return True

def test_missing_functions():
    """Test that scripts without required functions fail"""
    print("\nTesting validation of missing functions...")
    
    # Create a test script without required functions
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        test_script = f.name
        f.write("""
import torch

def compute(input):
    return input
# Missing get_output_type and get_script_output_schema
""")
    
    try:
        module = load_custom_script(test_script)
        print("✗ Should have failed - script missing required functions")
        return False
    except AttributeError as e:
        if "missing required function" in str(e):
            print(f"✓ Correctly rejected script: {e}")
            return True
        else:
            print(f"✗ Wrong error: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(test_script):
            os.remove(test_script)

def test_schema_resolution():
    """Test two-phase schema resolution"""
    print("\nTesting schema resolution...")
    
    # Test identity function schema resolution
    module = load_custom_script("example_identity.py")
    
    # Phase 1: Initial schema (with None values)
    initial_schema = module.get_script_output_schema(initial=True)
    output_info = initial_schema['outputs']['output']
    
    if output_info['shape'] is None and output_info['flattened_size'] is None:
        print("✓ Initial schema has unresolved fields")
    else:
        print("✗ Initial schema should have None for unresolved fields")
        return False
    
    # Phase 2: Resolved schema
    test_input = {
        "input": {
            "type": "tensor",
            "shape": [32, 64],
            "flattened_size": 2048,
            "dtype": "float32"
        }
    }
    resolved_schema = module.get_script_output_schema(initial=False, input_schema=test_input)
    resolved_output = resolved_schema['outputs']['output']
    
    # For identity function, output should match input
    if (resolved_output['shape'] == [32, 64] and 
        resolved_output['flattened_size'] == 2048 and
        resolved_output['dtype'] == 'float32'):
        print("✓ Schema correctly resolved from input")
    else:
        print(f"✗ Schema resolution failed: {resolved_output}")
        return False
    
    return True

def test_node_dynamic_types():
    """Test that CustomComputationNode updates RETURN_TYPES dynamically"""
    print("\nTesting node dynamic type updates...")
    
    from custom_nodes.custom_computation_visnode import CustomComputationNode
    
    # Test with empty src_path - should use wildcard
    inputs = {
        "required": {
            "input": ("*TENSOR", {}),
            "src_path": {"default": ""}
        }
    }
    CustomComputationNode._update_return_type(inputs)
    
    if CustomComputationNode.RETURN_TYPES == ("*",):
        print("✓ Empty src_path uses wildcard type")
    else:
        print(f"✗ Expected ('*',), got {CustomComputationNode.RETURN_TYPES}")
        return False
    
    # Test with valid script
    inputs["required"]["src_path"]["default"] = "example_identity.py"
    CustomComputationNode._update_return_type(inputs)
    
    if CustomComputationNode.RETURN_TYPES == ("*TENSOR",):
        print("✓ Valid script sets correct output type")
    else:
        print(f"✗ Expected ('*TENSOR',), got {CustomComputationNode.RETURN_TYPES}")
        return False
    
    # Test with sink script
    inputs["required"]["src_path"]["default"] = "example_sink.py"
    CustomComputationNode._update_return_type(inputs)
    
    if CustomComputationNode.RETURN_TYPES == ("VOID",):
        print("✓ Sink script sets VOID output type")
    else:
        print(f"✗ Expected ('VOID',), got {CustomComputationNode.RETURN_TYPES}")
        return False
    
    return True

def test_reshape_example():
    """Test the reshape example script that performs actual computation"""
    print("\nTesting reshape example with actual transformation...")
    
    import torch
    import io
    import sys
    
    # Capture prints from the script
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    
    try:
        # Load the reshape script
        module = load_custom_script("example_reshape.py")
        
        # Restore stdout to show our test output
        sys.stdout = old_stdout
        
        # Test output type
        output_type = module.get_output_type()
        if output_type != "RESHAPED_TENSOR":
            print(f"✗ Expected output type 'RESHAPED_TENSOR', got '{output_type}'")
            return False
        print("✓ Reshape script has correct output type")
        
        # Test initial schema
        initial_schema = module.get_script_output_schema(initial=True)
        output_info = initial_schema['outputs']['output']
        if output_info['shape'] is None and output_info['flattened_size'] is None:
            print("✓ Initial schema has unresolved fields")
        else:
            print("✗ Initial schema should have None for unresolved fields")
            return False
        
        # Test resolved schema with specific input
        input_schema = {
            "input": {
                "type": "tensor",
                "shape": [32, 64, 3],
                "flattened_size": 6144,
                "dtype": "float32"
            }
        }
        resolved_schema = module.get_script_output_schema(initial=False, input_schema=input_schema)
        resolved_output = resolved_schema['outputs']['output']
        
        # Check that reshape doubles first dim and halves second
        expected_shape = [64, 32, 3]  # [32*2, 64/2, 3]
        if resolved_output['shape'] == expected_shape:
            print(f"✓ Schema correctly computed reshape: [32,64,3] -> {expected_shape}")
        else:
            print(f"✗ Expected shape {expected_shape}, got {resolved_output['shape']}")
            return False
        
        # Test actual computation
        test_tensor = torch.randn(32, 64, 3)
        result = module.compute(test_tensor)
        
        # Verify the reshape
        if result.shape == torch.Size([64, 32, 3]):
            print(f"✓ Tensor correctly reshaped: {test_tensor.shape} -> {result.shape}")
        else:
            print(f"✗ Expected shape torch.Size([64, 32, 3]), got {result.shape}")
            return False
        
        # Verify element count preserved
        if test_tensor.numel() == result.numel():
            print(f"✓ Element count preserved: {test_tensor.numel()}")
        else:
            print(f"✗ Element count not preserved: {test_tensor.numel()} != {result.numel()}")
            return False
        
        # Test edge case: tensor that can't be reshaped evenly
        test_tensor_odd = torch.randn(31, 63, 3)  # 31 * 2 = 62, 63 / 2 = 31.5
        result_odd = module.compute(test_tensor_odd)
        
        if test_tensor_odd.numel() == result_odd.numel():
            print(f"✓ Handled odd dimensions gracefully: {test_tensor_odd.shape} -> {result_odd.shape}")
        else:
            print(f"✗ Failed to preserve elements for odd dimensions")
            return False
        
    finally:
        # Always restore stdout
        sys.stdout = old_stdout
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Custom Computation Dynamic Types")
    print("=" * 60)
    
    tests = [
        ("Script Loading", test_script_loading),
        ("Cache Functionality", test_cache_functionality),
        ("Missing Functions Validation", test_missing_functions),
        ("Schema Resolution", test_schema_resolution),
        ("Node Dynamic Types", test_node_dynamic_types),
        ("Reshape Example (Actual Computation)", test_reshape_example),
    ]
    
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            if not test_func():
                failed_tests.append(test_name)
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            failed_tests.append(test_name)
    
    # Summary
    print("\n" + "=" * 60)
    if not failed_tests:
        print("✓ ALL TESTS PASSED")
        print(f"  {len(tests)} tests completed successfully")
    else:
        print(f"✗ {len(failed_tests)} TESTS FAILED:")
        for test in failed_tests:
            print(f"  - {test}")
    print("=" * 60)
    
    return 0 if not failed_tests else 1

if __name__ == "__main__":
    sys.exit(main())