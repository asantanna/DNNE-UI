#!/usr/bin/env python3
"""
DNNE Runtime Utilities
Shared utilities for exported workflows including checkpoint management
"""

import os
import re
import time
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta


class CheckpointManager:
    """
    Manages checkpointing for DNNE workflows with command line directory support
    
    Features:
    - Node-based organization: checkpoint_dir/node_{id}/
    - Simple 2-file structure: model.pt + metadata.json
    - Flexible time format parsing ("1h30m", "45s", "2h", etc.)
    - Multiple trigger types: epoch-based, time-based, best metric
    - Command line directory specification
    """
    
    def __init__(self, node_id: str, checkpoint_dir: str = None):
        """
        Initialize checkpoint manager
        
        Args:
            node_id: ID of the node being checkpointed
            checkpoint_dir: Base directory from command line (None if not saving)
        """
        self.node_id = node_id
        self.checkpoint_dir = checkpoint_dir
        
        # Create node-specific directory if checkpoint_dir provided
        if self.checkpoint_dir:
            self.node_checkpoint_path = Path(self.checkpoint_dir) / f"node_{node_id}"
            self.node_checkpoint_path.mkdir(parents=True, exist_ok=True)
        else:
            self.node_checkpoint_path = None
            
        # Initialize state
        self.last_checkpoint_time = time.time()
        self.best_metric_value = None
        self.best_metric_type = None  # 'min' or 'max'
        
    def parse_time_format(self, time_str: str) -> float:
        """
        Parse flexible time format into seconds
        
        Supported formats:
        - "30s" -> 30 seconds
        - "5m" -> 5 minutes (300 seconds)
        - "1h" -> 1 hour (3600 seconds)
        - "1h30m" -> 1.5 hours (5400 seconds)
        - "2h45m30s" -> 2 hours 45 minutes 30 seconds
        
        Args:
            time_str: Time string in flexible format
            
        Returns:
            Total seconds as float
        """
        if not time_str:
            return 0.0
            
        # Remove whitespace
        time_str = time_str.strip().lower()
        
        # Pattern to match time components
        pattern = r'(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?'
        match = re.match(pattern, time_str)
        
        if not match:
            raise ValueError(f"Invalid time format: {time_str}. Use format like '1h30m', '45s', '2h'")
            
        hours, minutes, seconds = match.groups()
        
        total_seconds = 0.0
        if hours:
            total_seconds += int(hours) * 3600
        if minutes:
            total_seconds += int(minutes) * 60
        if seconds:
            total_seconds += int(seconds)
            
        if total_seconds == 0:
            raise ValueError(f"Invalid time format: {time_str}. Must specify at least one time component")
            
        return total_seconds
    
    def should_checkpoint(self, trigger_type: str, trigger_value: Any = None, 
                         current_epoch: int = None, current_metric: float = None) -> bool:
        """
        Check if checkpoint should be triggered
        
        Args:
            trigger_type: Type of trigger ('epoch', 'time', 'best_metric', 'external')
            trigger_value: Value for the trigger (depends on type)
            current_epoch: Current epoch number (for epoch-based triggers)
            current_metric: Current metric value (for best metric triggers)
            
        Returns:
            True if checkpoint should be created
        """
        current_time = time.time()
        
        if trigger_type == 'epoch':
            # trigger_value should be epoch interval (int)
            if current_epoch is None or trigger_value is None:
                return False
            return current_epoch % int(trigger_value) == 0
            
        elif trigger_type == 'time':
            # trigger_value should be time interval string
            if trigger_value is None:
                return False
            interval_seconds = self.parse_time_format(str(trigger_value))
            return (current_time - self.last_checkpoint_time) >= interval_seconds
            
        elif trigger_type == 'best_metric':
            # trigger_value should be dict with 'type' ('min' or 'max') and optionally 'threshold'
            if current_metric is None or trigger_value is None:
                return False
                
            if isinstance(trigger_value, str):
                # Simple format: just 'min' or 'max'
                metric_type = trigger_value.lower()
                threshold = None
            else:
                # Dict format: {'type': 'min', 'threshold': 0.01}
                metric_type = trigger_value.get('type', 'min').lower()
                threshold = trigger_value.get('threshold', 0.0)
            
            if metric_type not in ['min', 'max']:
                raise ValueError(f"Invalid metric type: {metric_type}. Must be 'min' or 'max'")
            
            # First metric is always best
            if self.best_metric_value is None:
                self.best_metric_value = current_metric
                self.best_metric_type = metric_type
                return True
            
            # Check if current metric is better
            is_better = False
            if metric_type == 'min':
                improvement = self.best_metric_value - current_metric
                is_better = improvement > (threshold or 0.0)
            else:  # max
                improvement = current_metric - self.best_metric_value
                is_better = improvement > (threshold or 0.0)
            
            if is_better:
                self.best_metric_value = current_metric
                return True
            return False
            
        else:
            raise ValueError(f"Unknown trigger type: {trigger_type}")
    
    def save_checkpoint(self, model_state_dict: Dict[str, Any], 
                       metadata: Dict[str, Any] = None) -> bool:
        """
        Save checkpoint with model weights and metadata
        
        Args:
            model_state_dict: PyTorch model state dictionary (weights only)
            metadata: Additional metadata (hyperparameters, training progress, etc.)
            
        Returns:
            True if saved successfully, False if no checkpoint directory
        """
        if not self.node_checkpoint_path:
            return False  # No checkpoint directory specified
            
        # Prepare paths
        model_path = self.node_checkpoint_path / "model.pt"
        metadata_path = self.node_checkpoint_path / "metadata.json"
        
        # Save model weights
        torch.save(model_state_dict, model_path)
        
        # Prepare metadata
        full_metadata = {
            'node_id': self.node_id,
            'timestamp': datetime.now().isoformat(),
            'best_metric_value': self.best_metric_value,
            'best_metric_type': self.best_metric_type,
            **(metadata or {})
        }
        
        # Save metadata as JSON
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        
        # Update state
        self.last_checkpoint_time = time.time()
        
        print(f"📄 Checkpoint saved: {self.node_checkpoint_path}")
        return True
    
    def load_checkpoint(self, load_checkpoint_dir: str = None) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from command line specified directory
        
        Args:
            load_checkpoint_dir: Directory to load from (overrides default)
            
        Returns:
            Dict with 'model_state_dict' and 'metadata' or None if not found
        """
        # Determine load path
        if load_checkpoint_dir:
            load_path = Path(load_checkpoint_dir) / f"node_{self.node_id}"
        elif self.node_checkpoint_path:
            load_path = self.node_checkpoint_path
        else:
            return None  # No load directory specified
            
        model_path = load_path / "model.pt"
        metadata_path = load_path / "metadata.json"
        
        # Check if files exist
        if not model_path.exists() or not metadata_path.exists():
            return None
            
        try:
            # Load model weights
            model_state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Update internal state from metadata
            self.best_metric_value = metadata.get('best_metric_value')
            self.best_metric_type = metadata.get('best_metric_type')
            
            print(f"📄 Checkpoint loaded: {load_path}")
            
            return {
                'model_state_dict': model_state_dict,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"⚠️ Error loading checkpoint {load_path}: {e}")
            return None
    
    def checkpoint_exists(self, load_checkpoint_dir: str = None) -> bool:
        """
        Check if checkpoint exists for this node
        
        Args:
            load_checkpoint_dir: Directory to check (overrides default)
            
        Returns:
            True if both model.pt and metadata.json exist
        """
        # Determine load path
        if load_checkpoint_dir:
            load_path = Path(load_checkpoint_dir) / f"node_{self.node_id}"
        elif self.node_checkpoint_path:
            load_path = self.node_checkpoint_path
        else:
            return False
            
        model_path = load_path / "model.pt"
        metadata_path = load_path / "metadata.json"
        
        return model_path.exists() and metadata_path.exists()


def validate_checkpoint_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize checkpoint configuration
    
    Args:
        config: Checkpoint configuration dictionary
        
    Returns:
        Validated and normalized config
        
    Raises:
        ValueError: If configuration is invalid
    """
    if not config.get('enabled', False):
        return config
    
    # Validate trigger type
    trigger_type = config.get('trigger_type', 'epoch')
    if trigger_type not in ['epoch', 'time', 'best_metric']:
        raise ValueError(f"Invalid trigger type: {trigger_type}")
    
    # Validate trigger value based on type
    trigger_value = config.get('trigger_value')
    if trigger_type == 'epoch':
        # Try to convert string to int if needed
        if isinstance(trigger_value, str):
            try:
                trigger_value = int(trigger_value)
            except ValueError:
                raise ValueError("Epoch trigger value must be a valid integer")
        if not isinstance(trigger_value, int) or trigger_value <= 0:
            raise ValueError("Epoch trigger value must be a positive integer")
    elif trigger_type == 'time':
        if not isinstance(trigger_value, str) or not trigger_value.strip():
            raise ValueError("Time trigger value must be a non-empty string")
        # Test parsing
        try:
            # Create temporary manager to test parsing
            temp_manager = CheckpointManager("temp")
            temp_manager.parse_time_format(trigger_value)
        except Exception as e:
            raise ValueError(f"Invalid time format: {e}")
    elif trigger_type == 'best_metric':
        if isinstance(trigger_value, str):
            if trigger_value.lower() not in ['min', 'max']:
                raise ValueError("Best metric trigger value must be 'min' or 'max'")
        elif isinstance(trigger_value, dict):
            if trigger_value.get('type', '').lower() not in ['min', 'max']:
                raise ValueError("Best metric trigger type must be 'min' or 'max'")
        else:
            raise ValueError("Best metric trigger value must be string or dict")
    
    return config