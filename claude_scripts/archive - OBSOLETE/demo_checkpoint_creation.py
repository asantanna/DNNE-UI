#!/usr/bin/env python3
"""
Demo script to create checkpoints you can examine manually
These will NOT be automatically deleted
"""

import sys
import torch
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/mnt/e/ALS-Projects/DNNE/DNNE-UI')

def demo_checkpoint_creation():
    """Create some demo checkpoints you can examine"""
    
    from export_system.templates.base.run_utils import CheckpointManager
    
    # Create checkpoints in the project's temp directory
    demo_dir = Path.cwd() / "temp" / "demo_checkpoints"
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔐 Creating demo checkpoints in: {demo_dir.absolute()}")
    
    # Initialize checkpoint manager
    manager = CheckpointManager(
        workflow_name="demo_workflow",
        node_id="demo_node",
        checkpoint_dir=str(demo_dir)
    )
    
    # Create some demo checkpoints with different trigger types
    test_models = [
        {
            'model_state_dict': {'layer1.weight': torch.randn(10, 5), 'layer1.bias': torch.randn(10)},
            'epoch': 10,
            'loss': 0.5
        },
        {
            'model_state_dict': {'layer1.weight': torch.randn(10, 5), 'layer1.bias': torch.randn(10)},
            'epoch': 20, 
            'loss': 0.3
        },
        {
            'model_state_dict': {'layer1.weight': torch.randn(10, 5), 'layer1.bias': torch.randn(10)},
            'epoch': 30,
            'loss': 0.1
        }
    ]
    
    checkpoint_files = []
    
    for i, model_data in enumerate(test_models):
        metadata = {
            'trigger_type': 'epoch',
            'trigger_value': '10',
            'training_epoch': model_data['epoch'],
            'loss': model_data['loss'],
            'demo_info': f'This is demo checkpoint {i+1}'
        }
        
        filename = f"demo_checkpoint_epoch_{model_data['epoch']}.pt"
        checkpoint_file = manager.save_checkpoint(
            model_data, 
            metadata=metadata,
            filename=filename
        )
        checkpoint_files.append(checkpoint_file)
        
    print(f"\n✅ Created {len(checkpoint_files)} demo checkpoints:")
    for cp_file in checkpoint_files:
        file_size = Path(cp_file).stat().st_size
        print(f"  📄 {Path(cp_file).name} ({file_size} bytes)")
    
    # List all checkpoints
    print(f"\n📋 Checkpoint listing:")
    checkpoints = manager.list_checkpoints()
    for i, cp_info in enumerate(checkpoints):
        print(f"  {i+1}. {cp_info['filename']}")
        print(f"     Size: {cp_info['size']} bytes")
        print(f"     Modified: {cp_info['modified']}")
        print(f"     Epoch: {cp_info['metadata'].get('training_epoch', 'unknown')}")
        print(f"     Loss: {cp_info['metadata'].get('loss', 'unknown')}")
        print()
    
    print(f"💡 You can examine these files at: {demo_dir.absolute()}")
    print(f"💡 To load a checkpoint:")
    print(f"   manager.load_checkpoint('demo_checkpoint_epoch_30.pt')")
    print(f"💡 To clean up when done:")
    print(f"   rm -rf {demo_dir}")
    print(f"💡 Files are properly stored in project temp/ directory")

if __name__ == "__main__":
    demo_checkpoint_creation()