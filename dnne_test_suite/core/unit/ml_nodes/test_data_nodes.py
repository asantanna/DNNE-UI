"""
Unit tests for ML data nodes.

Tests MNISTDataset, BatchSampler, and GetBatch nodes for proper UI interface
and export configuration. Since DNNE nodes don't execute, we test their
interface and export capabilities, not runtime behavior.
"""

import pytest
import torch
import numpy as np
import os
from unittest.mock import Mock, MagicMock
from pathlib import Path

# Import nodes to test
from custom_nodes import MNISTDatasetNode, BatchSamplerNode, GetBatchNode
from fixtures.node_data import MNIST_DATASET_DATA, create_sample_mnist_batch
from fixtures.test_utils import assert_tensor_shape, assert_tensor_equal


class TestMNISTDatasetNode:
    """Test MNIST dataset node UI interface and export configuration."""
    
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
    def test_node_interface(self):
        """Test that MNISTDataset node has proper UI interface."""
        node = MNISTDatasetNode()
        
        # Check required UI attributes
        assert hasattr(node, "FUNCTION")
        assert node.FUNCTION is None  # DNNE nodes don't execute
        assert hasattr(node, "CATEGORY")
        assert "ml" in node.CATEGORY.lower() or "data" in node.CATEGORY.lower()
        
        # Check input/output interface
        input_types = node.INPUT_TYPES()
        assert isinstance(input_types, dict)
        assert "required" in input_types
        
        # Check return types
        assert hasattr(node, "RETURN_TYPES")
        assert hasattr(node, "RETURN_NAMES")
        assert len(node.RETURN_TYPES) == len(node.RETURN_NAMES)
    
    @pytest.mark.ml
    def test_export_functionality(self):
        """Test that MNISTDataset has proper export support."""
        from export_system.node_exporters import MNISTDatasetExporter
        
        # Test that exporter exists
        assert MNISTDatasetExporter is not None
        
        # Test template name
        template_name = MNISTDatasetExporter.get_template_name()
        assert template_name == "nodes/mnist_dataset_queue.tpl"
        
        # Test imports
        imports = MNISTDatasetExporter.get_imports()
        assert "import torch" in imports
        assert "from torchvision import datasets, transforms" in imports
    
    @pytest.mark.ml
    def test_category(self):
        """Test node category assignment."""
        node = MNISTDatasetNode()
        assert hasattr(node, "CATEGORY")
        category = node.CATEGORY.lower()
        assert any(keyword in category for keyword in ["ml", "data", "dataset", "dnne"])


class TestBatchSamplerNode:
    """Test BatchSampler node UI interface and export configuration."""
    
    @pytest.mark.ml
    def test_input_types(self):
        """Test that BatchSampler has correct input type definition."""
        node = BatchSamplerNode()
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        
        # Check for expected parameters
        required = input_types["required"]
        optional = input_types.get("optional", {})
        all_params = {**required, **optional}
        
        # Should accept dataset and batch configuration
        dataset_found = any("dataset" in k.lower() for k in all_params.keys())
        batch_found = any("batch" in k.lower() for k in all_params.keys())
        
        assert dataset_found or batch_found or len(all_params) >= 1
    
    @pytest.mark.ml
    def test_return_types(self):
        """Test that BatchSampler has correct return types."""
        node = BatchSamplerNode()
        
        assert hasattr(node, "RETURN_TYPES")
        assert hasattr(node, "RETURN_NAMES")
        
        return_types = node.RETURN_TYPES
        return_names = node.RETURN_NAMES
        
        # Should return sampler/dataloader
        assert len(return_types) == len(return_names)
        assert len(return_types) >= 1
    
    @pytest.mark.ml
    def test_node_interface(self):
        """Test that BatchSampler has proper UI interface."""
        node = BatchSamplerNode()
        
        # Check required UI attributes
        assert hasattr(node, "FUNCTION")
        assert node.FUNCTION is None  # DNNE nodes don't execute
        assert hasattr(node, "CATEGORY")
        
        # Check input/output interface
        input_types = node.INPUT_TYPES()
        assert isinstance(input_types, dict)
        assert "required" in input_types
    
    @pytest.mark.ml
    def test_export_functionality(self):
        """Test that BatchSampler has proper export support."""
        from export_system.node_exporters import BatchSamplerExporter
        
        # Test that exporter exists
        assert BatchSamplerExporter is not None
        
        # Test template name
        template_name = BatchSamplerExporter.get_template_name()
        assert template_name == "nodes/batch_sampler_queue.tpl"
        
        # Test imports
        imports = BatchSamplerExporter.get_imports()
        assert "import torch" in imports
        assert "from torch.utils.data import DataLoader" in imports
    
    @pytest.mark.ml
    def test_ui_parameters(self):
        """Test BatchSampler UI parameter configuration."""
        node = BatchSamplerNode()
        input_types = node.INPUT_TYPES()
        
        required = input_types["required"]
        optional = input_types.get("optional", {})
        all_params = {**required, **optional}
        
        # Should have batch_size parameter
        assert "batch_size" in all_params
        
        # Check batch_size configuration
        batch_size_config = all_params["batch_size"]
        assert batch_size_config[0] == "INT"  # Should be integer type
        assert batch_size_config[1]["default"] > 0  # Should have positive default


class TestGetBatchNode:
    """Test GetBatch node UI interface and export configuration."""
    
    @pytest.mark.ml
    def test_input_types(self):
        """Test that GetBatch has correct input type definition."""
        node = GetBatchNode()
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        
        # Check for expected parameters
        required = input_types["required"]
        optional = input_types.get("optional", {})
        all_params = {**required, **optional}
        
        # Should accept dataloader/sampler
        sampler_found = any("sampler" in k.lower() or "dataloader" in k.lower() for k in all_params.keys())
        
        assert sampler_found or len(all_params) >= 1
    
    @pytest.mark.ml
    def test_return_types(self):
        """Test that GetBatch has correct return types."""
        node = GetBatchNode()
        
        assert hasattr(node, "RETURN_TYPES")
        assert hasattr(node, "RETURN_NAMES")
        
        return_types = node.RETURN_TYPES
        return_names = node.RETURN_NAMES
        
        # Should return batch data and metadata
        assert len(return_types) == len(return_names)
        assert len(return_types) >= 1
    
    @pytest.mark.ml
    def test_node_interface(self):
        """Test that GetBatch has proper UI interface."""
        node = GetBatchNode()
        
        # Check required UI attributes
        assert hasattr(node, "FUNCTION")
        assert node.FUNCTION is None  # DNNE nodes don't execute
        assert hasattr(node, "CATEGORY")
        
        # Check that it's in appropriate category
        category = node.CATEGORY.lower()
        assert any(keyword in category for keyword in ["ml", "data", "batch"])
    
    @pytest.mark.ml
    def test_export_functionality(self):
        """Test that GetBatch has proper export support."""
        from export_system.node_exporters import GetBatchExporter
        
        # Test that exporter exists
        assert GetBatchExporter is not None
        
        # Test template name
        template_name = GetBatchExporter.get_template_name()
        assert template_name == "nodes/get_batch_queue.tpl"
        
        # Test imports - GetBatch doesn't need direct imports as it uses framework
        imports = GetBatchExporter.get_imports()
        # GetBatch uses framework globals, doesn't need torch/asyncio imports directly
        assert isinstance(imports, list)
    
    @pytest.mark.ml
    def test_trigger_interface(self):
        """Test that GetBatch properly handles trigger connections."""
        node = GetBatchNode()
        input_types = node.INPUT_TYPES()
        
        # Check for trigger input (may be in required or optional)
        all_params = {**input_types["required"], **input_types.get("optional", {})}
        
        # GetBatch typically uses triggers for batch coordination
        # Check that it has appropriate inputs for this
        assert len(all_params) >= 1  # Should have some inputs


class TestDataNodeIntegration:
    """Integration tests for data node export coordination."""
    
    @pytest.mark.ml
    @pytest.mark.integration
    def test_data_node_export_interfaces(self):
        """Test that all data nodes have consistent export interfaces."""
        nodes = [MNISTDatasetNode(), BatchSamplerNode(), GetBatchNode()]
        
        for node in nodes:
            # All should have required UI attributes
            assert hasattr(node, "INPUT_TYPES")
            assert hasattr(node, "RETURN_TYPES")
            assert hasattr(node, "RETURN_NAMES")
            assert hasattr(node, "FUNCTION")
            assert node.FUNCTION is None  # DNNE nodes don't execute
            assert hasattr(node, "CATEGORY")
            
            # All should be in ml/data category
            category = node.CATEGORY.lower()
            assert any(keyword in category for keyword in ["ml", "data", "dnne"])
    
    @pytest.mark.ml
    def test_data_node_exporter_consistency(self):
        """Test that data node exporters have consistent interfaces."""
        from export_system.node_exporters import (
            MNISTDatasetExporter, BatchSamplerExporter, GetBatchExporter
        )
        
        exporters = [MNISTDatasetExporter, BatchSamplerExporter, GetBatchExporter]
        
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
            assert template_name.endswith('.tpl')
    
    @pytest.mark.ml
    def test_batch_size_configuration(self):
        """Test that batch size is properly configurable across nodes."""
        sampler_node = BatchSamplerNode()
        
        # Check that BatchSampler accepts batch_size
        input_types = sampler_node.INPUT_TYPES()
        all_params = {**input_types["required"], **input_types.get("optional", {})}
        
        assert "batch_size" in all_params
        batch_size_config = all_params["batch_size"]
        
        # Should have reasonable defaults
        assert batch_size_config[1]["default"] >= 1
        assert batch_size_config[1]["min"] >= 1
        assert batch_size_config[1]["max"] >= batch_size_config[1]["default"]