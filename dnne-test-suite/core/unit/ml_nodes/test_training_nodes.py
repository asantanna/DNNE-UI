"""
Unit tests for ML training nodes.

Tests CrossEntropyLoss, Accuracy, SGDOptimizer, TrainingStep, and EpochTracker
nodes for proper training coordination and trigger-based execution.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import Mock, patch, MagicMock
import time

# Import nodes to test
from custom_nodes.ml_nodes.training_nodes import (
    CrossEntropyLossNode, AccuracyNode, SGDOptimizerNode, 
    TrainingStepNode, EpochTrackerNode
)
from fixtures.node_data import (
    SGD_OPTIMIZER_DATA, CROSS_ENTROPY_LOSS_DATA, create_sample_batch,
    create_sample_tensor
)
from fixtures.test_utils import MockTorchModule, assert_tensor_shape


class TestCrossEntropyLossNode:
    """Test CrossEntropyLoss node for loss computation."""
    
    @pytest.mark.ml
    def test_input_types(self):
        """Test CrossEntropyLoss input type definition."""
        node = CrossEntropyLossNode()
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        all_params = {**input_types["required"], **input_types.get("optional", {})}
        
        # Should accept predictions and targets
        pred_found = any("pred" in k.lower() for k in all_params.keys())
        target_found = any("target" in k.lower() or "label" in k.lower() for k in all_params.keys())
        
        assert pred_found or target_found or len(all_params) >= 2, \
            "Should accept predictions and targets"
    
    @pytest.mark.ml
    def test_return_types(self):
        """Test CrossEntropyLoss return types."""
        node = CrossEntropyLossNode()
        
        assert hasattr(node, "RETURN_TYPES")
        assert hasattr(node, "RETURN_NAMES")
        
        return_types = node.RETURN_TYPES
        return_names = node.RETURN_NAMES
        
        # Should return loss and potentially accuracy
        assert len(return_types) == len(return_names)
        assert len(return_types) >= 1  # At least loss
    
    @pytest.mark.ml
    def test_cross_entropy_export_functionality(self):
        """Test CrossEntropyLoss export functionality."""
        node = CrossEntropyLossNode()
        
        # Test UI interface
        input_types = node.INPUT_TYPES()
        assert "required" in input_types
        required = input_types["required"]
        
        # Should accept predictions and labels/targets
        assert "predictions" in required
        assert "labels" in required
        
        # Test that exporter exists
        from export_system.node_exporters.ml_nodes import CrossEntropyLossExporter
        
        # Test template name
        template_name = CrossEntropyLossExporter.get_template_name()
        assert template_name == "nodes/cross_entropy_queue.py"
        
        # Test imports
        imports = CrossEntropyLossExporter.get_imports()
        assert "import torch" in imports
        assert "import torch.nn as nn" in imports
    
    @pytest.mark.ml
    def test_cross_entropy_template_variables(self):
        """Test CrossEntropyLoss template variable preparation."""
        from export_system.node_exporters.ml_nodes import CrossEntropyLossExporter
        
        # Mock node data
        mock_data = {"widgets_values": []}
        mock_connections = {
            "predictions": [("node_1", "output")],
            "labels": [("node_2", "labels")]
        }
        
        # Test template variable preparation
        template_vars = CrossEntropyLossExporter.prepare_template_vars(
            "test_1", mock_data, mock_connections
        )
        
        # Validate the variables
        assert template_vars["NODE_ID"] == "test_1"
        assert template_vars["CLASS_NAME"] == "LossNode"
    
    @pytest.mark.ml
    def test_cross_entropy_ui_interface(self):
        """Test CrossEntropyLoss UI interface and return types."""
        node = CrossEntropyLossNode()
        
        # Test return types
        return_types = node.RETURN_TYPES
        return_names = node.RETURN_NAMES
        
        # CrossEntropyLoss should return loss and accuracy
        assert len(return_types) == len(return_names)
        assert len(return_types) == 2  # Should return (loss, accuracy)
        
        # Test function name for export
        assert hasattr(node, 'FUNCTION')
        
        # Test category
        assert hasattr(node, 'CATEGORY')
        assert "ml" in node.CATEGORY.lower() or "training" in node.CATEGORY.lower()


class TestAccuracyNode:
    """Test Accuracy node for standalone accuracy computation."""
    
    @pytest.mark.ml
    def test_input_types(self):
        """Test Accuracy input type definition."""
        node = AccuracyNode()
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        all_params = {**input_types["required"], **input_types.get("optional", {})}
        assert len(all_params) >= 1
    
    @pytest.mark.ml
    def test_accuracy_export_functionality(self):
        """Test Accuracy node export functionality."""
        node = AccuracyNode()
        
        # Test UI interface
        input_types = node.INPUT_TYPES()
        assert "required" in input_types
        required = input_types["required"]
        
        # Should accept predictions and targets
        assert "predictions" in required or "input" in required
        
        # Test node structure for export
        assert hasattr(node, "RETURN_TYPES")
        assert hasattr(node, "RETURN_NAMES")
        assert hasattr(node, "FUNCTION")
        
        # Should return accuracy metric
        return_types = node.RETURN_TYPES
        assert len(return_types) >= 1
    
    @pytest.mark.ml
    def test_accuracy_ui_interface(self):
        """Test Accuracy node UI interface."""
        node = AccuracyNode()
        
        # Test return types and names
        return_types = node.RETURN_TYPES
        return_names = node.RETURN_NAMES
        
        # Should return accuracy value
        assert len(return_types) == len(return_names)
        assert len(return_types) >= 1
        
        # Test category
        assert hasattr(node, "CATEGORY")
        category = node.CATEGORY.lower()
        assert "ml" in category or "training" in category or "accuracy" in category


class TestSGDOptimizerNode:
    """Test SGDOptimizer node for gradient-based optimization."""
    
    @pytest.mark.ml
    def test_input_types(self):
        """Test SGDOptimizer input type definition."""
        node = SGDOptimizerNode()
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        all_params = {**input_types["required"], **input_types.get("optional", {})}
        
        # Should accept model and optimizer parameters
        lr_found = any("lr" in k.lower() or "learning" in k.lower() for k in all_params.keys())
        model_found = any("model" in k.lower() for k in all_params.keys())
        
        assert lr_found or model_found or len(all_params) >= 1
    
    @pytest.mark.ml
    def test_return_types(self):
        """Test SGDOptimizer return types."""
        node = SGDOptimizerNode()
        
        assert hasattr(node, "RETURN_TYPES")
        assert hasattr(node, "RETURN_NAMES")
        
        return_types = node.RETURN_TYPES
        return_names = node.RETURN_NAMES
        
        # Should return optimizer
        assert len(return_types) == len(return_names)
        assert len(return_types) >= 1
    
    @pytest.mark.ml
    def test_sgd_optimizer_export_functionality(self):
        """Test SGD optimizer export functionality."""
        node = SGDOptimizerNode()
        
        # Test UI interface
        input_types = node.INPUT_TYPES()
        assert "required" in input_types
        required = input_types["required"]
        
        # Should accept network/model connection
        assert "network" in required
        
        # Test that exporter exists
        from export_system.node_exporters.ml_nodes import SGDOptimizerExporter
        
        # Test template name
        template_name = SGDOptimizerExporter.get_template_name()
        assert template_name == "nodes/sgd_optimizer_queue.py"
        
        # Test imports
        imports = SGDOptimizerExporter.get_imports()
        assert "import torch.optim as optim" in imports
    
    @pytest.mark.ml
    def test_sgd_optimizer_template_variables(self):
        """Test SGD optimizer template variable preparation."""
        from export_system.node_exporters.ml_nodes import SGDOptimizerExporter
        
        # Mock node data with SGD parameters
        mock_data = {
            "widgets_values": [0.01, 0.9]  # learning_rate, momentum
        }
        mock_connections = {
            "network": [("node_1", "model")]
        }
        
        # Test template variable preparation
        template_vars = SGDOptimizerExporter.prepare_template_vars(
            "test_1", mock_data, mock_connections
        )
        
        # Validate the variables
        assert template_vars["NODE_ID"] == "test_1"
        assert template_vars["CLASS_NAME"] == "SGDOptimizerNode"
        assert template_vars["LEARNING_RATE"] == 0.01
        assert template_vars["MOMENTUM"] == 0.9


class TestTrainingStepNode:
    """Test TrainingStep node for gradient updates and trigger coordination."""
    
    @pytest.mark.ml
    def test_input_types(self):
        """Test TrainingStep input type definition."""
        node = TrainingStepNode()
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        all_params = {**input_types["required"], **input_types.get("optional", {})}
        
        # Should accept loss and optimizer
        loss_found = any("loss" in k.lower() for k in all_params.keys())
        opt_found = any("optim" in k.lower() for k in all_params.keys())
        
        assert loss_found or opt_found or len(all_params) >= 2
    
    @pytest.mark.ml
    def test_return_types(self):
        """Test TrainingStep return types."""
        node = TrainingStepNode()
        
        assert hasattr(node, "RETURN_TYPES")
        assert hasattr(node, "RETURN_NAMES")
        
        return_types = node.RETURN_TYPES
        return_names = node.RETURN_NAMES
        
        # Should return ready signal or sync
        assert len(return_types) == len(return_names)
        assert len(return_types) >= 1
    
    @pytest.mark.ml
    def test_training_step_export_functionality(self):
        """Test TrainingStep export functionality."""
        node = TrainingStepNode()
        
        # Test UI interface
        input_types = node.INPUT_TYPES()
        assert "required" in input_types
        required = input_types["required"]
        
        # Should accept loss and optimizer
        assert "loss" in required
        assert "optimizer" in required
        
        # Test that exporter exists
        from export_system.node_exporters.ml_nodes import TrainingStepExporter
        
        # Test template name
        template_name = TrainingStepExporter.get_template_name()
        assert template_name == "nodes/training_step_queue.py"
        
        # Test imports
        imports = TrainingStepExporter.get_imports()
        assert "import torch" in imports
        assert "import asyncio" in imports
    
    @pytest.mark.ml
    def test_training_step_template_variables(self):
        """Test TrainingStep template variable preparation."""
        from export_system.node_exporters.ml_nodes import TrainingStepExporter
        
        # Mock node data
        mock_data = {"widgets_values": []}
        mock_connections = {
            "loss": [("node_1", "loss")],
            "optimizer": [("node_2", "optimizer")]
        }
        
        # Test template variable preparation
        template_vars = TrainingStepExporter.prepare_template_vars(
            "test_1", mock_data, mock_connections
        )
        
        # Validate the variables
        assert template_vars["NODE_ID"] == "test_1"
        assert template_vars["CLASS_NAME"] == "TrainingStepNode"
    
    @pytest.mark.ml
    def test_training_step_ui_interface(self):
        """Test TrainingStep UI interface and return types."""
        node = TrainingStepNode()
        
        # Test return types
        return_types = node.RETURN_TYPES
        return_names = node.RETURN_NAMES
        
        # TrainingStep should return ready signals
        assert len(return_types) == len(return_names)
        assert len(return_types) >= 1
        
        # Test function name for export
        assert hasattr(node, 'FUNCTION')
        
        # Test category
        assert hasattr(node, 'CATEGORY')
        category = node.CATEGORY.lower()
        assert "ml" in category or "training" in category


class TestEpochTrackerNode:
    """Test EpochTracker node for training progress monitoring."""
    
    @pytest.mark.ml
    def test_input_types(self):
        """Test EpochTracker input type definition."""
        node = EpochTrackerNode()
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        all_params = {**input_types["required"], **input_types.get("optional", {})}
        assert len(all_params) >= 1
    
    @pytest.mark.ml
    def test_return_types(self):
        """Test EpochTracker return types."""
        node = EpochTrackerNode()
        
        assert hasattr(node, "RETURN_TYPES")
        assert hasattr(node, "RETURN_NAMES")
        
        return_types = node.RETURN_TYPES
        return_names = node.RETURN_NAMES
        
        # Should return epoch info, stats, completion status
        assert len(return_types) == len(return_names)
        assert len(return_types) >= 1
    
    @pytest.mark.ml
    def test_epoch_tracker_export_functionality(self):
        """Test EpochTracker export functionality."""
        node = EpochTrackerNode()
        
        # Test UI interface
        input_types = node.INPUT_TYPES()
        assert "required" in input_types
        required = input_types["required"]
        
        # Should accept epoch statistics (max_epochs is optional)
        assert "epoch_stats" in required
        assert "loss" in required
        assert "accuracy" in required
        
        # max_epochs should be in optional
        optional = input_types.get("optional", {})
        assert "max_epochs" in optional
        
        # Test that exporter exists
        from export_system.node_exporters.ml_nodes import EpochTrackerExporter
        
        # Test template name
        template_name = EpochTrackerExporter.get_template_name()
        assert template_name == "nodes/epoch_tracker_queue.py"
    
    @pytest.mark.ml
    def test_epoch_tracker_template_variables(self):
        """Test EpochTracker template variable preparation."""
        from export_system.node_exporters.ml_nodes import EpochTrackerExporter
        
        # Mock node data with max epochs
        mock_data = {
            "widgets_values": [100]  # max_epochs
        }
        mock_connections = {
            "epoch_stats": [("node_1", "epoch_stats")],
            "loss": [("node_2", "loss")],
            "accuracy": [("node_3", "accuracy")],
            "max_epochs": [("node_4", "max_epochs")]
        }
        
        # Test template variable preparation
        template_vars = EpochTrackerExporter.prepare_template_vars(
            "test_1", mock_data, mock_connections
        )
        
        # Validate the variables
        assert template_vars["NODE_ID"] == "test_1"
        assert template_vars["CLASS_NAME"] == "EpochTrackerNode"
        assert template_vars["MAX_EPOCHS"] == 100


class TestTrainingNodeIntegration:
    """Integration tests for training node coordination."""
    
    @pytest.mark.ml
    @pytest.mark.integration
    def test_training_loop_export_integration(self):
        """Test that all training nodes have consistent export interfaces."""
        # Test that all training node types have exporters
        training_nodes = [CrossEntropyLossNode, SGDOptimizerNode, TrainingStepNode, EpochTrackerNode]
        
        for node_class in training_nodes:
            node = node_class()
            
            # All nodes should have consistent interface
            assert hasattr(node, "INPUT_TYPES")
            assert hasattr(node, "RETURN_TYPES")
            assert hasattr(node, "RETURN_NAMES")
            assert hasattr(node, "FUNCTION")
            assert hasattr(node, "CATEGORY")
            
            # All should be in ml/training category
            category = node.CATEGORY.lower()
            assert any(keyword in category for keyword in ["ml", "training", "loss", "optim"])
    
    @pytest.mark.ml
    def test_training_node_exporter_consistency(self):
        """Test that training nodes have consistent exporter interfaces."""
        from export_system.node_exporters.ml_nodes import (
            CrossEntropyLossExporter, SGDOptimizerExporter, TrainingStepExporter, EpochTrackerExporter
        )
        
        exporters = [
            CrossEntropyLossExporter, SGDOptimizerExporter, 
            TrainingStepExporter, EpochTrackerExporter
        ]
        
        for exporter_class in exporters:
            # All exporters should have required methods
            assert hasattr(exporter_class, 'get_template_name')
            assert hasattr(exporter_class, 'prepare_template_vars')
            assert hasattr(exporter_class, 'get_imports')
            
            # Methods should be callable
            assert callable(exporter_class.get_template_name)
            assert callable(exporter_class.prepare_template_vars)
            assert callable(exporter_class.get_imports)
            
            # Template name should be valid
            template_name = exporter_class.get_template_name()
            assert isinstance(template_name, str)
            assert template_name.endswith('.py')
    
    @pytest.mark.ml
    @pytest.mark.performance
    def test_training_node_performance_interface(self):
        """Test that training nodes have proper interfaces for performance testing."""
        loss_node = CrossEntropyLossNode()
        
        # Test that nodes have proper UI interface for performance configuration
        input_types = loss_node.INPUT_TYPES()
        assert "required" in input_types
        
        # Should have input parameters for batch processing
        required = input_types["required"]
        assert "predictions" in required
        assert "labels" in required
        
        # Test node structure for export performance
        assert hasattr(loss_node, "RETURN_TYPES")
        assert hasattr(loss_node, "FUNCTION")
        assert hasattr(loss_node, "CATEGORY")
    
    @pytest.mark.ml
    def test_training_node_categories(self):
        """Test that all training nodes have appropriate categories."""
        nodes = [
            CrossEntropyLossNode(), AccuracyNode(), SGDOptimizerNode(),
            TrainingStepNode(), EpochTrackerNode()
        ]
        
        for node in nodes:
            assert hasattr(node, "CATEGORY")
            category = node.CATEGORY.lower()
            assert any(keyword in category for keyword in ["ml", "training", "loss", "optim", "dnne"])
    
    @pytest.mark.ml
    def test_training_node_parameter_validation_interface(self):
        """Test that training nodes have proper parameter validation interfaces."""
        training_nodes = [CrossEntropyLossNode, SGDOptimizerNode, TrainingStepNode, EpochTrackerNode]
        
        for node_class in training_nodes:
            node = node_class()
            
            # All nodes should have input type definitions for parameter validation
            input_types = node.INPUT_TYPES()
            assert "required" in input_types
            
            # Should have proper return type definitions
            assert hasattr(node, "RETURN_TYPES")
            return_types = node.RETURN_TYPES
            assert isinstance(return_types, (list, tuple))
            assert len(return_types) > 0
            
            # Should have return names matching return types
            assert hasattr(node, "RETURN_NAMES")
            return_names = node.RETURN_NAMES
            assert len(return_types) == len(return_names)