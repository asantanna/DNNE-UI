#!/usr/bin/env python3
"""
Integration test for simplified command line checkpoint system
Tests the complete workflow from command line args to checkpoint save/load
"""

import os
import sys
import tempfile
import shutil
import torch
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/mnt/e/ALS-Projects/DNNE/DNNE-UI')

def test_command_line_checkpoint_integration():
    """Test the complete command line checkpoint workflow"""
    print("🧪 Testing command line checkpoint integration...")
    
    from export_system.templates.base.run_utils import CheckpointManager
    
    # Create temporary directories for save and load
    save_dir = tempfile.mkdtemp()
    load_dir = tempfile.mkdtemp()
    
    try:
        # Simulate runner.py setting command line args in builtins
        import builtins
        builtins.SAVE_CHECKPOINT_DIR = save_dir
        builtins.LOAD_CHECKPOINT_DIR = None  # No load initially
        
        print(f"📁 Save directory: {save_dir}")
        print(f"📁 Load directory: {load_dir}")
        
        # Test 1: Create checkpoint manager with command line save directory
        manager1 = CheckpointManager(node_id="ppo_trainer_5", checkpoint_dir=save_dir)
        
        # Create sample model state and metadata
        model_state = {
            'linear1.weight': torch.randn(64, 32),
            'linear1.bias': torch.randn(64),
            'linear2.weight': torch.randn(10, 64),
            'linear2.bias': torch.randn(10)
        }
        
        metadata = {
            'trigger_type': 'epoch',
            'trigger_value': '10',
            'training_step': 100,
            'loss': 0.234,
            'hyperparameters': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'horizon_length': 16
            }
        }
        
        # Save checkpoint
        success = manager1.save_checkpoint(model_state, metadata=metadata)
        assert success is True, "Checkpoint save should succeed"
        
        # Verify checkpoint structure
        node_dir = Path(save_dir) / "node_ppo_trainer_5"
        assert node_dir.exists(), "Node directory should exist"
        assert (node_dir / "model.pt").exists(), "Model file should exist"
        assert (node_dir / "metadata.json").exists(), "Metadata file should exist"
        
        print("✅ Checkpoint saved with correct structure")
        
        # Copy saved checkpoint to load directory
        load_node_dir = Path(load_dir) / "node_ppo_trainer_5"
        shutil.copytree(node_dir, load_node_dir)
        
        # Test 2: Load checkpoint from different directory
        manager2 = CheckpointManager(node_id="ppo_trainer_5", checkpoint_dir=None)
        
        # Load checkpoint from load directory
        loaded_data = manager2.load_checkpoint(load_dir)
        assert loaded_data is not None, "Checkpoint load should succeed"
        
        # Verify loaded data
        loaded_model = loaded_data['model_state_dict']
        loaded_metadata = loaded_data['metadata']
        
        # Check model weights
        for key in model_state:
            assert key in loaded_model, f"Model key {key} missing from loaded data"
            assert torch.equal(model_state[key], loaded_model[key]), f"Model weight {key} doesn't match"
        
        # Check metadata
        assert loaded_metadata['trigger_type'] == 'epoch', "Trigger type incorrect"
        assert loaded_metadata['training_step'] == 100, "Training step incorrect"
        assert abs(loaded_metadata['loss'] - 0.234) < 1e-6, "Loss value incorrect"
        assert loaded_metadata['hyperparameters']['learning_rate'] == 0.001, "Learning rate incorrect"
        
        print("✅ Checkpoint loaded successfully with all data intact")
        
        # Test 3: Test checkpoint existence checking
        assert manager1.checkpoint_exists() is True, "Checkpoint should exist in save directory"
        assert manager2.checkpoint_exists(load_dir) is True, "Checkpoint should exist in load directory"
        
        # Test with non-existent directory
        empty_dir = tempfile.mkdtemp()
        try:
            assert manager1.checkpoint_exists(empty_dir) is False, "Checkpoint should not exist in empty directory"
        finally:
            shutil.rmtree(empty_dir)
        
        print("✅ Checkpoint existence checking works correctly")
        
        # Test 4: Verify JSON structure is readable
        with open(node_dir / "metadata.json", 'r') as f:
            json_data = json.load(f)
            
        assert json_data['node_id'] == 'ppo_trainer_5', "Node ID in JSON incorrect"
        assert 'timestamp' in json_data, "Timestamp missing from JSON"
        assert json_data['hyperparameters']['batch_size'] == 32, "Batch size in JSON incorrect"
        
        print("✅ Metadata JSON structure is correct and readable")
        
        print("🎉 Command line checkpoint integration test passed!")
        
    finally:
        # Cleanup
        shutil.rmtree(save_dir)
        shutil.rmtree(load_dir)
        
        # Clean up builtins
        if hasattr(builtins, 'SAVE_CHECKPOINT_DIR'):
            delattr(builtins, 'SAVE_CHECKPOINT_DIR')
        if hasattr(builtins, 'LOAD_CHECKPOINT_DIR'):
            delattr(builtins, 'LOAD_CHECKPOINT_DIR')

def test_node_id_based_organization():
    """Test that different nodes create separate checkpoint directories"""
    print("🧪 Testing node ID based checkpoint organization...")
    
    from export_system.templates.base.run_utils import CheckpointManager
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create managers for different nodes
        ppo_manager = CheckpointManager(node_id="ppo_trainer_1", checkpoint_dir=temp_dir)
        network_manager = CheckpointManager(node_id="network_2", checkpoint_dir=temp_dir)
        
        # Save checkpoints for both nodes
        model_state = {'weight': torch.tensor([1.0])}
        
        ppo_success = ppo_manager.save_checkpoint(model_state, metadata={'node_type': 'ppo'})
        network_success = network_manager.save_checkpoint(model_state, metadata={'node_type': 'network'})
        
        assert ppo_success and network_success, "Both checkpoints should save successfully"
        
        # Verify separate directories
        ppo_dir = Path(temp_dir) / "node_ppo_trainer_1"
        network_dir = Path(temp_dir) / "node_network_2"
        
        assert ppo_dir.exists(), "PPO trainer directory should exist"
        assert network_dir.exists(), "Network directory should exist"
        assert ppo_dir != network_dir, "Directories should be different"
        
        # Verify independent loading
        ppo_data = ppo_manager.load_checkpoint()
        network_data = network_manager.load_checkpoint()
        
        assert ppo_data['metadata']['node_type'] == 'ppo', "PPO metadata incorrect"
        assert network_data['metadata']['node_type'] == 'network', "Network metadata incorrect"
        
        print("✅ Node ID based organization works correctly")

def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Running Simplified Checkpoint Integration Tests")
    print("=" * 60)
    
    try:
        test_command_line_checkpoint_integration()
        test_node_id_based_organization()
        
        print("\n" + "=" * 60)
        print("✅ All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)