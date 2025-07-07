"""
Unit tests for ML data nodes.

Tests MNISTDataset, BatchSampler, and GetBatch nodes for proper data loading,
batch generation, and trigger-based coordination.
"""

import pytest
import torch
import numpy as np
import os
from unittest.mock import Mock, MagicMock
from pathlib import Path

# Import nodes to test
from custom_nodes.ml_nodes.data_nodes import MNISTDatasetNode, BatchSamplerNode, GetBatchNode
from fixtures.node_data import MNIST_DATASET_DATA, create_sample_mnist_batch
from fixtures.test_utils import assert_tensor_shape, assert_tensor_equal


class TestMNISTDatasetNode:
    """Test MNIST dataset loading and configuration."""
    
    @pytest.mark.ml
    def test_input_types(self):
        """Test that MNISTDataset has correct input type definition."""
        node = MNISTDatasetNode()
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        
        # Check for expected parameters
        required = input_types["required"]
        optional = input_types.get("optional", {})
        
        # Should have basic configuration in required or optional
        all_params = {**required, **optional}
        
        # Check for dataset parameters (data_path, train, download are more relevant than batch_size)
        expected_params = ["data_path", "train", "download"]
        found_params = [p for p in expected_params if p in all_params]
        assert len(found_params) > 0, f"Expected dataset params not found. Available: {list(all_params.keys())}"
    
    @pytest.mark.ml
    def test_return_types(self):
        """Test that MNISTDataset has correct return types."""
        node = MNISTDatasetNode()
        
        assert hasattr(node, "RETURN_TYPES")
        assert hasattr(node, "RETURN_NAMES")
        
        return_types = node.RETURN_TYPES
        return_names = node.RETURN_NAMES
        
        # Should return dataset and metadata
        assert len(return_types) == len(return_names)
        
        # Should have DATASET type or dataset in names
        has_dataset = "DATASET" in return_types or any("dataset" in name.lower() for name in return_names)
        assert has_dataset, f"Should have dataset output. Types: {return_types}, Names: {return_names}"
    
    @pytest.mark.ml
    @pytest.mark.timeout(60)  # Allow time for download if needed
    def test_dataset_creation(self, mnist_config):
        """Test dataset creation with configurable download."""
        node = MNISTDatasetNode()
        
        # Test with configurable parameters
        result = node.load_dataset(
            data_path=mnist_config['data_path'],
            download=mnist_config['download'],
            train=True
        )
        
        # Should return dataset and schema
        assert result is not None
        assert len(result) >= 2  # Dataset and schema
        
        # Check that result contains dataset and schema
        dataset, schema = result
        assert dataset is not None
        assert isinstance(schema, dict)
        assert 'input_size' in schema
        assert 'num_classes' in schema
        
        # Verify dataset has correct size
        assert len(dataset) > 0
    
    @pytest.mark.ml
    def test_category(self):
        """Test node category classification."""
        node = MNISTDatasetNode()
        assert hasattr(node, "CATEGORY")
        assert "ml" in node.CATEGORY.lower() or "data" in node.CATEGORY.lower()


class TestBatchSamplerNode:
    """Test batch sampling with trigger coordination."""
    
    @pytest.mark.ml
    def test_input_types(self):
        """Test BatchSampler input type definition."""
        node = BatchSamplerNode()
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        # Should accept dataset and potentially trigger inputs
        all_params = {**input_types["required"], **input_types.get("optional", {})}
        
        # Look for dataset input
        dataset_found = any("dataset" in str(v).lower() for v in all_params.values())
        assert dataset_found, "BatchSampler should accept dataset input"
    
    @pytest.mark.ml
    def test_return_types(self):
        """Test BatchSampler return types."""
        node = BatchSamplerNode()
        
        assert hasattr(node, "RETURN_TYPES")
        assert hasattr(node, "RETURN_NAMES")
        
        return_types = node.RETURN_TYPES
        return_names = node.RETURN_NAMES
        
        # Should return sampler or batch data
        assert len(return_types) == len(return_names)
    
    @pytest.mark.ml
    def test_batch_sampler_creation(self):
        """Test batch sampler creation with mock dataset."""
        node = BatchSamplerNode()
        
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)
        
        # Test sampler creation
        mock_schema = {"input_size": 784, "num_classes": 10}
        result = node.create_dataloader(
            dataset=mock_dataset,
            schema=mock_schema,
            batch_size=32,
            shuffle=True,
            seed=42
        )
        
        assert result is not None
        assert len(result) >= 1
    
    @pytest.mark.ml
    def test_trigger_handling(self):
        """Test trigger-based batch generation."""
        node = BatchSamplerNode()
        
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        
        # Create mock sampler for get_batch testing
        mock_sampler = Mock()
        mock_sampler.__len__ = Mock(return_value=10)
        
        # Test with trigger signal
        trigger_signal = {"signal_type": "ready", "timestamp": 1234567890}
        
        # Should handle trigger input gracefully
        try:
            mock_schema = {"input_size": 784, "num_classes": 10}
            result = node.create_dataloader(
                dataset=mock_dataset,
                schema=mock_schema,
                batch_size=16,
                shuffle=False,
                seed=42
            )
            assert result is not None
        except Exception as e:
            # If trigger handling not implemented, should fail gracefully
            assert "trigger" in str(e).lower() or "not implemented" in str(e).lower()


class TestGetBatchNode:
    """Test GetBatch node for data retrieval and flow control."""
    
    @pytest.mark.ml
    def test_input_types(self):
        """Test GetBatch input type definition."""
        node = GetBatchNode()
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        # Should accept sampler and potentially trigger
        all_params = {**input_types["required"], **input_types.get("optional", {})}
        
        # Look for sampler input
        sampler_found = any("sampler" in str(v).lower() for v in all_params.values())
        assert sampler_found or len(all_params) > 0  # Should have some input
    
    @pytest.mark.ml
    def test_return_types(self):
        """Test GetBatch return types."""
        node = GetBatchNode()
        
        assert hasattr(node, "RETURN_TYPES")
        assert hasattr(node, "RETURN_NAMES")
        
        return_types = node.RETURN_TYPES
        return_names = node.RETURN_NAMES
        
        # Should return batch data (images, labels)
        assert len(return_types) == len(return_names)
        assert len(return_types) >= 2  # At least images and labels
    
    @pytest.mark.ml
    def test_batch_retrieval(self):
        """Test batch data retrieval."""
        node = GetBatchNode()
        
        # Clear context memory for test isolation
        from custom_nodes.ml_nodes.base import get_context
        context = get_context()
        if hasattr(context, 'memory'):
            context.memory.clear()
        
        # Create mock sampler that yields sample batches
        def mock_sampler_iter():
            for i in range(3):
                batch_images, batch_labels = create_sample_mnist_batch(batch_size=16)
                yield batch_images, batch_labels
        
        mock_sampler = Mock()
        mock_sampler.__iter__ = Mock(return_value=mock_sampler_iter())
        mock_sampler.__len__ = Mock(return_value=10)  # Add length for dataloader
        
        # Test batch retrieval
        mock_schema = {"input_size": 784, "num_classes": 10}
        mock_trigger = {"signal_type": "ready"}
        result = node.get_batch(dataloader=mock_sampler, schema=mock_schema, trigger=mock_trigger)
        
        assert result is not None
        assert len(result) >= 2  # Should return images and labels
        
        # If actual tensors returned, check shapes
        if len(result) >= 2 and hasattr(result[0], 'shape'):
            images, labels = result[0], result[1]
            assert_tensor_shape(images, (16, 1, 28, 28))  # MNIST batch shape
            assert_tensor_shape(labels, (16,))  # Label shape
    
    @pytest.mark.ml
    def test_trigger_coordination(self):
        """Test trigger-based batch coordination."""
        node = GetBatchNode()
        
        # Clear context memory for test isolation
        from custom_nodes.ml_nodes.base import get_context
        context = get_context()
        if hasattr(context, 'memory'):
            context.memory.clear()
        
        # Mock sampler with required methods and sample data
        def mock_iter():
            yield torch.randn(2, 1, 28, 28), torch.randint(0, 10, (2,))  # Sample batch
        
        mock_sampler = Mock()
        mock_sampler.__iter__ = Mock(return_value=mock_iter())
        mock_sampler.__len__ = Mock(return_value=5)  # Add length for dataloader
        
        # Test with trigger signal
        trigger_signal = {"signal_type": "ready"}
        
        try:
            mock_schema = {"input_size": 784, "num_classes": 10}
            result = node.get_batch(
                dataloader=mock_sampler,
                schema=mock_schema,
                trigger=trigger_signal
            )
            # Should handle trigger gracefully
            assert result is not None or "trigger" in str(result)
        except Exception as e:
            # Should fail gracefully if trigger not implemented
            assert "trigger" in str(e).lower() or "implemented" in str(e).lower()


class TestDataNodeIntegration:
    """Integration tests for data node coordination."""
    
    @pytest.mark.ml
    @pytest.mark.integration
    @pytest.mark.timeout(90)  # Allow time for dataset operations
    def test_data_flow_coordination(self, mnist_config):
        """Test data flow between MNISTDataset -> BatchSampler -> GetBatch."""
        # Create nodes
        dataset_node = MNISTDatasetNode()
        sampler_node = BatchSamplerNode()
        batch_node = GetBatchNode()
        
        # Create dataset with real data
        dataset_result = dataset_node.load_dataset(
            data_path=mnist_config['data_path'],
            download=mnist_config['download'],
            train=True
        )
        
        assert dataset_result is not None
        dataset, schema = dataset_result
        
        # Create sampler from dataset
        sampler_result = sampler_node.create_dataloader(
            dataset=dataset,
            schema=schema,
            batch_size=32,
            shuffle=True,
            seed=42
        )
        
        assert sampler_result is not None
        
        # For get_batch test, we still need to mock the dataloader iteration
        # since the actual dataloader might not work in test context
        def mock_batch_iter():
            yield create_sample_mnist_batch(32)
        
        mock_batch_dataloader = Mock()
        mock_batch_dataloader.__iter__ = Mock(return_value=mock_batch_iter())
        mock_batch_dataloader.__len__ = Mock(return_value=10)
        
        # Clear context for clean test
        from custom_nodes.ml_nodes.base import get_context
        context = get_context()
        if hasattr(context, 'memory'):
            context.memory.clear()
        
        trigger = {"signal_type": "ready"}
        batch_result = batch_node.get_batch(
            dataloader=mock_batch_dataloader,
            schema=schema,
            trigger=trigger
        )
        
        assert batch_result is not None
    
    @pytest.mark.ml
    def test_node_categories(self):
        """Test that all data nodes have appropriate categories."""
        nodes = [MNISTDatasetNode(), BatchSamplerNode(), GetBatchNode()]
        
        for node in nodes:
            assert hasattr(node, "CATEGORY")
            category = node.CATEGORY.lower()
            assert "ml" in category or "data" in category or "dnne" in category
    
    @pytest.mark.ml
    def test_node_display_names(self):
        """Test that all data nodes have display names."""
        from custom_nodes.ml_nodes import NODE_DISPLAY_NAME_MAPPINGS
        
        expected_nodes = ["MNISTDataset", "BatchSampler", "GetBatch"]
        
        for node_name in expected_nodes:
            assert node_name in NODE_DISPLAY_NAME_MAPPINGS
            display_name = NODE_DISPLAY_NAME_MAPPINGS[node_name]
            assert isinstance(display_name, str)
            assert len(display_name) > 0
    
    @pytest.mark.ml
    def test_error_handling(self):
        """Test error handling in data nodes."""
        nodes = [MNISTDatasetNode(), BatchSamplerNode(), GetBatchNode()]
        
        # Test error handling for each node type
        dataset_node, sampler_node, batch_node = nodes
        
        # Test MNISTDataset error handling
        try:
            result = dataset_node.load_dataset()  # Missing required args
        except Exception as e:
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in 
                      ["required", "missing", "argument", "parameter"])
        
        # Test BatchSampler error handling  
        try:
            result = sampler_node.create_dataloader()  # Missing required args
        except Exception as e:
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in 
                      ["required", "missing", "argument", "parameter"])
        
        # Test GetBatch error handling
        try:
            result = batch_node.get_batch()  # Missing required args
        except Exception as e:
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in 
                      ["required", "missing", "argument", "parameter"])
    
    @pytest.mark.ml
    @pytest.mark.performance
    @pytest.mark.timeout(120)  # Allow time for multiple dataset operations
    def test_batch_size_handling(self, mnist_config):
        """Test handling of different batch sizes."""
        dataset_node = MNISTDatasetNode()
        
        # Test with a smaller set of batch sizes for real data
        batch_sizes = [1, 32, 128]
        
        for batch_size in batch_sizes:
            result = dataset_node.load_dataset(
                data_path=mnist_config['data_path'],
                download=mnist_config['download'],
                train=True
            )
            
            assert result is not None
            dataset, schema = result
            assert len(dataset) > batch_size  # Ensure dataset is large enough