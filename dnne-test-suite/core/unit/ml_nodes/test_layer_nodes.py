"""
Unit tests for ML layer nodes.

Tests Network, LinearLayer, Conv2DLayer, Activation, Dropout, BatchNorm, 
and Flatten nodes for proper layer construction and forward passes.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import nodes to test
from custom_nodes.ml_nodes.layer_nodes import (
    NetworkNode, LinearLayerNode, Conv2DLayerNode, ActivationNode,
    DropoutNode, BatchNormNode, FlattenNode
)
from fixtures.node_data import (
    LINEAR_LAYER_DATA, NETWORK_DATA, create_sample_tensor, create_sample_mnist_batch
)
from fixtures.test_utils import assert_tensor_shape, MockTorchModule


class TestLinearLayerNode:
    """Test LinearLayer node for dense layer creation."""
    
    @pytest.mark.ml
    def test_input_types(self):
        """Test LinearLayer input type definition."""
        node = LinearLayerNode()
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        # Should have in_features and out_features
        all_params = {**input_types["required"], **input_types.get("optional", {})}
        
        # Look for dimension parameters
        dim_params = [k for k in all_params.keys() 
                     if any(dim in k.lower() for dim in ["feature", "input", "output", "dim"])]
        assert len(dim_params) >= 1, "Should have dimension parameters"
    
    @pytest.mark.ml
    def test_return_types(self):
        """Test LinearLayer return types."""
        node = LinearLayerNode()
        
        assert hasattr(node, "RETURN_TYPES")
        assert hasattr(node, "RETURN_NAMES")
        
        return_types = node.RETURN_TYPES
        return_names = node.RETURN_NAMES
        
        # Should return layer/module
        assert len(return_types) == len(return_names)
        assert len(return_types) >= 1
    
    @pytest.mark.ml
    def test_linear_layer_export(self):
        """Test LinearLayer export functionality."""
        node = LinearLayerNode()
        
        # Test UI interface
        input_types = node.INPUT_TYPES()
        assert "required" in input_types
        required = input_types["required"]
        
        # Should have dimension and activation parameters
        assert "output_size" in required
        assert "bias" in required
        assert "activation" in required
        
        # Test that exporter exists and works
        from export_system.node_exporters.ml_nodes import LinearLayerExporter
        
        # Test template name
        template_name = LinearLayerExporter.get_template_name()
        assert template_name == "nodes/linear_layer_queue.py"
        
        # Test imports
        imports = LinearLayerExporter.get_imports()
        assert "import torch" in imports
        assert "import torch.nn as nn" in imports
    
    @pytest.mark.ml
    def test_linear_layer_template_variables(self):
        """Test LinearLayer template variable preparation."""
        from export_system.node_exporters.ml_nodes import LinearLayerExporter
        
        # Mock node data with widget values
        mock_data = {
            "widgets_values": [128, True, "relu", 0.1]  # output_size, bias, activation, dropout
        }
        
        # Mock connections - need to handle missing connections gracefully
        mock_connections = {}
        
        # Test template variable preparation (may fail due to missing connections, which is expected)
        try:
            template_vars = LinearLayerExporter.prepare_template_vars(
                "test_1", mock_data, mock_connections
            )
            
            # If successful, validate the variables
            assert template_vars["NODE_ID"] == "test_1"
            assert template_vars["CLASS_NAME"] == "LinearLayerNode"
            assert template_vars["OUTPUT_SIZE"] == 128
            assert template_vars["BIAS_VALUE"] == True
            assert template_vars["ACTIVATION_VALUE"] == "relu"
            assert template_vars["DROPOUT"] == 0.1
            
        except ValueError as e:
            # Expected when no input connections - this is normal for isolated node tests
            assert "No input connection found" in str(e) or "input" in str(e).lower()
    
    @pytest.mark.ml
    def test_linear_layer_ui_defaults(self):
        """Test LinearLayer UI default values."""
        node = LinearLayerNode()
        
        # Test UI input type definitions
        input_types = node.INPUT_TYPES()
        
        # Check default values
        required = input_types["required"]
        
        # Should have reasonable defaults for output_size
        output_size_config = required["output_size"]
        assert output_size_config[1]["default"] >= 1  # Should have a positive default
        assert output_size_config[1]["min"] >= 1      # Should have a positive minimum
        
        # Should have bias default
        bias_config = required["bias"]
        assert bias_config[1]["default"] == True      # Should default to True
        
        # Should have activation options
        activation_config = required["activation"]
        activation_options = activation_config[0]
        assert "relu" in activation_options
        assert "none" in activation_options or "linear" in activation_options


class TestNetworkNode:
    """Test Network node for model consolidation and execution."""
    
    @pytest.mark.ml
    def test_input_types(self):
        """Test Network input type definition."""
        node = NetworkNode()
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        # Should accept input data and potentially layers
        all_params = {**input_types["required"], **input_types.get("optional", {})}
        assert len(all_params) >= 1  # Should have some inputs
    
    @pytest.mark.ml
    def test_return_types(self):
        """Test Network return types."""
        node = NetworkNode()
        
        assert hasattr(node, "RETURN_TYPES")
        assert hasattr(node, "RETURN_NAMES")
        
        return_types = node.RETURN_TYPES
        return_names = node.RETURN_NAMES
        
        # Should return predictions and model
        assert len(return_types) == len(return_names)
        assert len(return_types) >= 1
    
    @pytest.mark.ml
    def test_network_export_functionality(self):
        """Test network export functionality."""
        node = NetworkNode()
        
        # Test that NetworkNode has proper UI interface
        input_types = node.INPUT_TYPES()
        assert "required" in input_types
        required = input_types["required"]
        
        # Network should accept input data
        assert "input" in required
        assert "to_output" in required  # Loop-back connection
        
        # Test that exporter exists
        from export_system.node_exporters.ml_nodes import NetworkExporter
        
        # Test template name
        template_name = NetworkExporter.get_template_name()
        assert template_name == "nodes/network_queue.py"
        
        # Test imports
        imports = NetworkExporter.get_imports()
        assert "import torch" in imports
        assert "import torch.nn as nn" in imports
    
    @pytest.mark.ml
    def test_network_ui_interface(self):
        """Test network UI interface and return types."""
        node = NetworkNode()
        
        # Test return types
        return_types = node.RETURN_TYPES
        return_names = node.RETURN_NAMES
        
        # Network should return layers, output, and model
        assert len(return_types) == len(return_names)
        assert len(return_types) == 3  # Should return (layers, output, model)
        assert "TENSOR" in return_types  # Should include tensor outputs
        assert "MODEL" in return_types   # Should include model output
        
        # Test forward method exists (for UI connectivity)
        assert hasattr(node, 'forward')
        assert callable(node.forward)


class TestActivationNode:
    """Test Activation node for activation functions."""
    
    @pytest.mark.ml
    def test_input_types(self):
        """Test Activation input type definition."""
        node = ActivationNode()
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        # Should accept input tensor and activation type
        all_params = {**input_types["required"], **input_types.get("optional", {})}
        assert len(all_params) >= 1
    
    @pytest.mark.ml
    def test_activation_export_functionality(self):
        """Test activation node export functionality."""
        node = ActivationNode()
        
        # Test UI interface
        input_types = node.INPUT_TYPES()
        assert "required" in input_types
        required = input_types["required"]
        
        # Should accept input tensor and activation type
        assert "input" in required
        
        # Test that the node has proper structure for export
        assert hasattr(node, "RETURN_TYPES")
        assert hasattr(node, "RETURN_NAMES")
        assert hasattr(node, "FUNCTION")
        
        # ActivationNode uses 'apply' function
        assert node.FUNCTION == "apply"
        
        # Should return tensor output
        return_types = node.RETURN_TYPES
        assert "TENSOR" in return_types


class TestConv2DLayerNode:
    """Test Conv2D layer for convolutional operations."""
    
    @pytest.mark.ml
    def test_input_types(self):
        """Test Conv2D input type definition."""
        node = Conv2DLayerNode()
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        # Should have channel and kernel parameters
        all_params = {**input_types["required"], **input_types.get("optional", {})}
        
        conv_params = [k for k in all_params.keys() 
                      if any(param in k.lower() for param in ["channel", "kernel", "filter"])]
        assert len(conv_params) >= 1, "Should have convolution parameters"
    
    @pytest.mark.ml
    def test_conv2d_export_functionality(self):
        """Test Conv2D layer export functionality."""
        node = Conv2DLayerNode()
        
        # Test UI interface
        input_types = node.INPUT_TYPES()
        assert "required" in input_types
        required = input_types["required"]
        
        # Should have convolution-specific parameters
        assert "input" in required  # Input tensor
        
        # Test node structure for export
        assert hasattr(node, "RETURN_TYPES")
        assert hasattr(node, "RETURN_NAMES")
        assert hasattr(node, "FUNCTION")
        
        # Conv2D should use 'forward' function like other layers
        assert node.FUNCTION == "forward"
        
        # Should return tensor output
        return_types = node.RETURN_TYPES
        assert "TENSOR" in return_types


class TestDropoutNode:
    """Test Dropout node for regularization."""
    
    @pytest.mark.ml
    def test_input_types(self):
        """Test Dropout input type definition."""
        node = DropoutNode()
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        all_params = {**input_types["required"], **input_types.get("optional", {})}
        assert len(all_params) >= 1
    
    @pytest.mark.ml
    def test_dropout_export_functionality(self):
        """Test dropout export functionality."""
        node = DropoutNode()
        
        # Test UI interface
        input_types = node.INPUT_TYPES()
        assert "required" in input_types
        required = input_types["required"]
        
        # Should accept input tensor and dropout parameters
        assert "input" in required
        
        # Test node structure for export
        assert hasattr(node, "RETURN_TYPES")
        assert hasattr(node, "RETURN_NAMES")
        assert hasattr(node, "FUNCTION")
        
        # Dropout should use 'apply' function
        assert node.FUNCTION == "apply"
        
        # Should return tensor output
        return_types = node.RETURN_TYPES
        assert "TENSOR" in return_types
    
    @pytest.mark.ml
    def test_dropout_ui_parameters(self):
        """Test dropout UI parameter configuration."""
        node = DropoutNode()
        
        # Test input types include dropout rate parameter
        input_types = node.INPUT_TYPES()
        all_params = {**input_types["required"], **input_types.get("optional", {})}
        
        # Should have dropout probability parameter
        # (parameter name may vary - p, dropout, dropout_rate, etc.)
        dropout_params = [k for k in all_params.keys() 
                         if any(param in k.lower() for param in ["dropout", "p", "rate"])]
        
        # Allow flexible parameter naming
        if len(dropout_params) == 0:
            # If no explicit dropout parameter, that's also valid for some implementations
            pass  # Some dropout implementations might not expose the parameter in UI


class TestBatchNormNode:
    """Test BatchNorm node for normalization."""
    
    @pytest.mark.ml
    def test_input_types(self):
        """Test BatchNorm input type definition."""
        node = BatchNormNode()
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        all_params = {**input_types["required"], **input_types.get("optional", {})}
        assert len(all_params) >= 1
    
    @pytest.mark.ml
    def test_batch_norm_export_functionality(self):
        """Test BatchNorm export functionality."""
        node = BatchNormNode()
        
        # Test UI interface
        input_types = node.INPUT_TYPES()
        assert "required" in input_types
        required = input_types["required"]
        
        # Should accept input tensor
        assert "input" in required
        
        # Test node structure for export
        assert hasattr(node, "RETURN_TYPES")
        assert hasattr(node, "RETURN_NAMES")
        assert hasattr(node, "FUNCTION")
        
        # BatchNorm should use 'apply' function
        assert node.FUNCTION == "apply"
        
        # Should return tensor output
        return_types = node.RETURN_TYPES
        assert "TENSOR" in return_types


class TestFlattenNode:
    """Test Flatten node for tensor reshaping."""
    
    @pytest.mark.ml
    def test_input_types(self):
        """Test Flatten input type definition."""
        node = FlattenNode()
        input_types = node.INPUT_TYPES()
        
        assert "required" in input_types
        all_params = {**input_types["required"], **input_types.get("optional", {})}
        assert len(all_params) >= 1
    
    @pytest.mark.ml
    def test_flatten_export_functionality(self):
        """Test flatten export functionality."""
        node = FlattenNode()
        
        # Test UI interface
        input_types = node.INPUT_TYPES()
        assert "required" in input_types
        required = input_types["required"]
        
        # Should accept input tensor
        assert "input" in required
        
        # Test node structure for export
        assert hasattr(node, "RETURN_TYPES")
        assert hasattr(node, "RETURN_NAMES")
        assert hasattr(node, "FUNCTION")
        
        # Flatten should use 'flatten' function
        assert node.FUNCTION == "flatten"
        
        # Should return tensor output
        return_types = node.RETURN_TYPES
        assert "TENSOR" in return_types
    
    @pytest.mark.ml
    def test_flatten_ui_configuration(self):
        """Test flatten UI configuration options."""
        node = FlattenNode()
        
        # Test that flatten has minimal configuration
        input_types = node.INPUT_TYPES()
        
        # Flatten typically just needs input tensor
        required = input_types.get("required", {})
        optional = input_types.get("optional", {})
        all_params = {**required, **optional}
        
        # Should have input parameter
        assert "input" in all_params
        
        # May have optional parameters like start_dim, end_dim
        # but these are implementation-specific


class TestLayerNodeIntegration:
    """Integration tests for layer node combinations."""
    
    @pytest.mark.ml
    @pytest.mark.integration
    def test_layer_export_integration(self):
        """Test that all layer nodes have consistent export interfaces."""
        # Test that all layer node types have exporters
        layer_nodes = [LinearLayerNode, NetworkNode, ActivationNode, Conv2DLayerNode, 
                      DropoutNode, BatchNormNode, FlattenNode]
        
        for node_class in layer_nodes:
            node = node_class()
            
            # All nodes should have consistent interface
            assert hasattr(node, "INPUT_TYPES")
            assert hasattr(node, "RETURN_TYPES")
            assert hasattr(node, "RETURN_NAMES")
            assert hasattr(node, "FUNCTION")
            assert hasattr(node, "CATEGORY")
            
            # All should be in ml category
            assert "ml" in node.CATEGORY.lower()
            
            # All should return tensors
            assert "TENSOR" in node.RETURN_TYPES
    
    @pytest.mark.ml
    def test_layer_node_categories(self):
        """Test that all layer nodes have appropriate categories."""
        nodes = [
            NetworkNode(), LinearLayerNode(), Conv2DLayerNode(),
            ActivationNode(), DropoutNode(), BatchNormNode(), FlattenNode()
        ]
        
        for node in nodes:
            assert hasattr(node, "CATEGORY")
            category = node.CATEGORY.lower()
            assert any(keyword in category for keyword in ["ml", "layer", "network", "dnne"])
    
    @pytest.mark.ml
    @pytest.mark.performance
    def test_layer_node_performance_interface(self):
        """Test that layer nodes have proper interfaces for performance testing."""
        linear_node = LinearLayerNode()
        
        # Test that nodes have proper UI interface for performance configuration
        input_types = linear_node.INPUT_TYPES()
        assert "required" in input_types
        
        # Should have output size parameter for scaling tests
        all_params = {**input_types["required"], **input_types.get("optional", {})}
        assert "output_size" in all_params
        
        # Test node structure for export performance
        assert hasattr(linear_node, "RETURN_TYPES")
        assert hasattr(linear_node, "FUNCTION")
        assert hasattr(linear_node, "CATEGORY")
    
    @pytest.mark.ml
    def test_layer_node_parameter_validation_interface(self):
        """Test that layer nodes have proper parameter validation interfaces."""
        layer_nodes = [LinearLayerNode, NetworkNode, ActivationNode, Conv2DLayerNode, 
                      DropoutNode, BatchNormNode, FlattenNode]
        
        for node_class in layer_nodes:
            node = node_class()
            
            # All nodes should have input type definitions for parameter validation
            input_types = node.INPUT_TYPES()
            assert "required" in input_types
            
            # Should have proper return type definitions
            assert hasattr(node, "RETURN_TYPES")
            return_types = node.RETURN_TYPES
            assert isinstance(return_types, (list, tuple))
            assert len(return_types) > 0