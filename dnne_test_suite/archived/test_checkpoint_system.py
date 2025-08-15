#!/usr/bin/env python3
"""
Test script for DNNE checkpoint system functionality
Tests CheckpointManager, time parsing, and checkpoint integration
"""

import os
import sys
import time
import torch
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/mnt/e/ALS-Projects/DNNE/DNNE-UI')

def test_checkpoint_manager_basic():
    """Test basic CheckpointManager functionality"""
    print("🧪 Testing CheckpointManager basic functionality...")
    
    # Import CheckpointManager
    from export_system.templates.base.run_utils import CheckpointManager, validate_checkpoint_config
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize checkpoint manager (simplified API)
        manager = CheckpointManager(
            node_id="test_node",
            checkpoint_dir=temp_dir
        )
        
        # Test basic save/load with simplified structure
        model_state_dict = {'weight': torch.tensor([1.0, 2.0, 3.0])}
        
        metadata = {
            'trigger_type': 'external',
            'test_value': 'test_data',
            'step': 1,
            'value': 42.0
        }
        
        # Save checkpoint (only model weights + metadata)
        success = manager.save_checkpoint(model_state_dict, metadata=metadata)
        assert success is True, "Checkpoint save failed"
        
        # Verify files exist
        node_dir = Path(temp_dir) / "node_test_node"
        assert node_dir.exists(), "Node checkpoint directory doesn't exist"
        assert (node_dir / "model.pt").exists(), "Model file doesn't exist"
        assert (node_dir / "metadata.json").exists(), "Metadata file doesn't exist"
        
        # Load checkpoint
        loaded_data = manager.load_checkpoint()
        assert loaded_data is not None, "Checkpoint load failed"
        assert 'model_state_dict' in loaded_data, "Model state dict missing from loaded data"
        assert 'metadata' in loaded_data, "Metadata missing from loaded data"
        
        # Verify loaded data
        loaded_model_state = loaded_data['model_state_dict']
        assert torch.equal(loaded_model_state['weight'], torch.tensor([1.0, 2.0, 3.0])), "Model weights incorrect"
        
        loaded_metadata = loaded_data['metadata']
        assert loaded_metadata['trigger_type'] == 'external', "Trigger type incorrect"
        assert loaded_metadata['test_value'] == 'test_data', "Test value incorrect"
        assert loaded_metadata['step'] == 1, "Step value incorrect"
        assert loaded_metadata['value'] == 42.0, "Value incorrect"
        
        print("✅ Basic checkpoint save/load test passed")

def test_time_parsing():
    """Test time format parsing functionality"""
    print("🧪 Testing time format parsing...")
    
    from export_system.templates.base.run_utils import CheckpointManager
    
    # Create temporary checkpoint manager for testing (simplified API)
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = CheckpointManager("test_node", temp_dir)
        
        # Test various time formats
        test_cases = [
            ("30s", 30.0),
            ("5m", 300.0),
            ("1h", 3600.0),
            ("1h30m", 5400.0),
            ("2h45m30s", 2*3600 + 45*60 + 30),
            ("15m30s", 15*60 + 30),
            ("3h15s", 3*3600 + 15)
        ]
        
        for time_str, expected_seconds in test_cases:
            result = manager.parse_time_format(time_str)
            assert result == expected_seconds, f"Time parsing failed for '{time_str}': expected {expected_seconds}, got {result}"
            
        print("✅ Time format parsing tests passed")

def test_checkpoint_triggers():
    """Test different checkpoint trigger types"""
    print("🧪 Testing checkpoint trigger functionality...")
    
    from export_system.templates.base.run_utils import CheckpointManager
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = CheckpointManager("test_node", temp_dir)
        
        # Test epoch-based triggers
        assert manager.should_checkpoint("epoch", "10", current_epoch=10) == True
        assert manager.should_checkpoint("epoch", "10", current_epoch=20) == True
        assert manager.should_checkpoint("epoch", "10", current_epoch=5) == False
        
        # Test best metric triggers (first metric is always best)
        assert manager.should_checkpoint("best_metric", "min", current_metric=0.5) == True
        assert manager.should_checkpoint("best_metric", "min", current_metric=0.3) == True  # Better (lower)
        assert manager.should_checkpoint("best_metric", "min", current_metric=0.7) == False  # Worse (higher)
        
        print("✅ Checkpoint trigger tests passed")

def test_config_validation():
    """Test checkpoint configuration validation"""
    print("🧪 Testing checkpoint configuration validation...")
    
    from export_system.templates.base.run_utils import validate_checkpoint_config
    
    # Test valid configurations (simplified - no path required)
    valid_configs = [
        {
            'enabled': False,
            'trigger_type': 'epoch',
            'trigger_value': '10'
        },
        {
            'enabled': True,
            'trigger_type': 'epoch',
            'trigger_value': '5'
        },
        {
            'enabled': True,
            'trigger_type': 'time',
            'trigger_value': '1h30m'
        },
        {
            'enabled': True,
            'trigger_type': 'best_metric',
            'trigger_value': 'min'
        }
    ]
    
    for config in valid_configs:
        try:
            validate_checkpoint_config(config)
            print(f"  ✅ Valid config passed: {config['trigger_type']}")
        except ValueError as e:
            print(f"  ❌ Valid config failed: {e}")
            raise
    
    # Test invalid configurations (simplified)
    invalid_configs = [
        {
            'enabled': True,
            'trigger_type': 'invalid_type',  # Invalid trigger type
            'trigger_value': '10'
        },
        {
            'enabled': True,
            'trigger_type': 'epoch',
            'trigger_value': 'invalid'  # Invalid epoch value
        }
    ]
    
    for config in invalid_configs:
        try:
            validate_checkpoint_config(config)
            print(f"  ❌ Invalid config passed when it should have failed: {config}")
            raise AssertionError("Invalid config should have failed validation")
        except ValueError:
            print(f"  ✅ Invalid config correctly rejected: {config.get('trigger_type', 'missing')}")
    
    print("✅ Configuration validation tests passed")

def test_checkpoint_existence():
    """Test checkpoint existence checking functionality"""
    print("🧪 Testing checkpoint existence checking...")
    
    from export_system.templates.base.run_utils import CheckpointManager
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = CheckpointManager("test_node", temp_dir)
        
        # Initially no checkpoint should exist
        assert manager.checkpoint_exists() == False, "Checkpoint should not exist initially"
        
        # Save a checkpoint
        model_state_dict = {'weight': torch.tensor([1.0, 2.0])}
        metadata = {'step': 1}
        
        success = manager.save_checkpoint(model_state_dict, metadata=metadata)
        assert success == True, "Checkpoint save should succeed"
        
        # Now checkpoint should exist
        assert manager.checkpoint_exists() == True, "Checkpoint should exist after saving"
        
        # Test with different load directory
        other_temp_dir = tempfile.mkdtemp()
        try:
            assert manager.checkpoint_exists(other_temp_dir) == False, "Checkpoint should not exist in different directory"
        finally:
            shutil.rmtree(other_temp_dir)
        
        print("✅ Checkpoint existence tests passed")

def run_all_tests():
    """Run all checkpoint system tests"""
    print("🚀 Running DNNE Checkpoint System Tests")
    print("=" * 50)
    
    try:
        test_checkpoint_manager_basic()
        test_time_parsing()
        test_checkpoint_triggers()
        test_config_validation()
        test_checkpoint_existence()
        
        print("\n" + "=" * 50)
        print("✅ All checkpoint system tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)