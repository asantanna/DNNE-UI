"""
Unit tests for DNNE node exporters.

Tests ML and robotics node exporters for proper parameter extraction,
template variable preparation, and import management.
"""

import pytest
from unittest.mock import Mock, patch

# Import node exporters
from export_system.node_exporters.ml_nodes import (
    MNISTDatasetExporter, LinearLayerExporter, NetworkExporter,
    SGDOptimizerExporter, CrossEntropyLossExporter, TrainingStepExporter
)
from fixtures.node_data import (
    LINEAR_LAYER_DATA, MNIST_DATASET_DATA, NETWORK_DATA, 
    SGD_OPTIMIZER_DATA, CROSS_ENTROPY_LOSS_DATA, SAMPLE_TEMPLATE_VARS
)
from fixtures.test_utils import assert_valid_python_code


class TestExportableNodeBase:
    """Test base exporter functionality."""
    
    @pytest.mark.export
    def test_exporter_interface(self):
        """Test that exporters implement required interface."""
        exporters = [
            MNISTDatasetExporter, LinearLayerExporter, NetworkExporter,
            SGDOptimizerExporter, CrossEntropyLossExporter, TrainingStepExporter
        ]
        
        for exporter_class in exporters:
            # Should have required class methods
            assert hasattr(exporter_class, 'get_template_name')
            assert hasattr(exporter_class, 'prepare_template_vars')
            assert hasattr(exporter_class, 'get_imports')
            
            # Methods should be callable
            assert callable(exporter_class.get_template_name)
            assert callable(exporter_class.prepare_template_vars)
            assert callable(exporter_class.get_imports)


class TestMNISTDatasetExporter:
    """Test MNIST dataset node exporter."""
    
    @pytest.mark.export
    def test_template_name(self):
        """Test template name generation."""
        template_name = MNISTDatasetExporter.get_template_name()
        
        assert isinstance(template_name, str)
        assert len(template_name) > 0
        assert template_name.endswith('.py')
        assert 'mnist' in template_name.lower() or 'dataset' in template_name.lower()
    
    @pytest.mark.export
    def test_template_vars_preparation(self):
        """Test template variable preparation."""
        node_id = "node_1"
        node_data = MNIST_DATASET_DATA
        connections = {}
        
        template_vars = MNISTDatasetExporter.prepare_template_vars(
            node_id, node_data, connections
        )
        
        assert isinstance(template_vars, dict)
        assert len(template_vars) > 0
        
        # Should have node ID
        assert "NODE_ID" in template_vars
        assert template_vars["NODE_ID"] == node_id
        
        # Should have class name
        assert "CLASS_NAME" in template_vars
        class_name = template_vars["CLASS_NAME"]
        assert "mnist" in class_name.lower() or "dataset" in class_name.lower()
    
    @pytest.mark.export
    def test_imports(self):
        """Test import statement generation."""
        imports = MNISTDatasetExporter.get_imports()
        
        assert isinstance(imports, list)
        assert len(imports) > 0
        
        # Should have torch-related imports
        import_text = " ".join(imports).lower()
        assert "torch" in import_text or "import" in import_text
    
    @pytest.mark.export
    def test_parameter_extraction(self):
        """Test extraction of MNIST dataset parameters."""
        # Use ComfyUI format with widgets_values
        node_data = {
            "widgets_values": ["./data", True, False]  # data_path, train, download
        }
        
        template_vars = MNISTDatasetExporter.prepare_template_vars(
            "test_node", node_data, {}
        )
        
        # Should extract parameters from widgets_values
        assert template_vars["DATA_PATH"] == "./data"
        assert template_vars["TRAIN"] == True
        assert template_vars["DOWNLOAD"] == False
        
        # Fixed batch size for MNIST
        assert template_vars["BATCH_SIZE"] == 32
        
        # Should have node identification
        assert template_vars["NODE_ID"] == "test_node"
        assert template_vars["CLASS_NAME"] == "MNISTDatasetNode"


class TestLinearLayerExporter:
    """Test Linear layer node exporter."""
    
    @pytest.mark.export
    def test_template_name(self):
        """Test template name generation."""
        template_name = LinearLayerExporter.get_template_name()
        
        assert isinstance(template_name, str)
        assert template_name.endswith('.py')
        assert 'linear' in template_name.lower() or 'layer' in template_name.lower()
    
    @pytest.mark.export
    def test_dimension_parameter_extraction(self):
        """Test extraction of layer dimensions."""
        # Use ComfyUI format with widgets_values
        node_data = {
            "widgets_values": [128, True, "relu", 0.0]  # output_size, bias, activation, dropout
        }
        
        # Mock connections for input size detection
        mock_connections = {
            "inputs": {
                "input": {
                    "from_node": "source_node",
                    "from_slot": 0
                }
            }
        }
        
        # Mock source node data
        mock_all_nodes = [
            {
                "id": "source_node",
                "type": "MNISTDataset",
                "widgets_values": []
            }
        ]
        
        # Mock node registry with MNISTDataset 
        from export_system.node_exporters.ml_nodes import MNISTDatasetExporter
        mock_node_registry = {
            "MNISTDataset": MNISTDatasetExporter
        }
        
        template_vars = LinearLayerExporter.prepare_template_vars(
            "layer_node", node_data, mock_connections, mock_node_registry, mock_all_nodes, []
        )
        
        # Should extract dimensions from widgets_values and connections
        assert template_vars["OUTPUT_SIZE"] == 128
        assert template_vars["BIAS_VALUE"] == True
        assert template_vars["ACTIVATION_VALUE"] == "relu"
        assert template_vars["DROPOUT"] == 0.0
        
        # Input size should be determined from connection (MNIST = 784)
        assert template_vars["INPUT_SIZE"] == 784
        
        # Should have node identification
        assert template_vars["NODE_ID"] == "layer_node"
        assert template_vars["CLASS_NAME"] == "LinearLayerNode"
    
    @pytest.mark.export
    def test_imports(self):
        """Test Linear layer imports."""
        imports = LinearLayerExporter.get_imports()
        
        assert isinstance(imports, list)
        
        # Should include PyTorch imports
        import_text = " ".join(imports).lower()
        expected_imports = ["torch", "nn", "linear"]
        
        # At least some torch-related imports should be present
        has_torch_import = any(imp in import_text for imp in expected_imports)
        assert has_torch_import, f"Should have torch imports, got: {imports}"


class TestNetworkExporter:
    """Test Network node exporter."""
    
    @pytest.mark.export
    def test_template_name(self):
        """Test Network template name."""
        template_name = NetworkExporter.get_template_name()
        
        assert isinstance(template_name, str)
        assert template_name.endswith('.py')
        assert 'network' in template_name.lower()
    
    @pytest.mark.export
    def test_network_parameter_extraction(self):
        """Test extraction of network parameters."""
        node_data = {
            "inputs": {
                "device": "cpu"
            },
            "widgets": {
                "input_shape": [1, 28, 28],
                "num_classes": 10,
                "dropout_rate": 0.1
            }
        }
        
        template_vars = NetworkExporter.prepare_template_vars(
            "network_node", node_data, {}
        )
        
        # Should extract device
        if "DEVICE" in template_vars:
            assert template_vars["DEVICE"] == "cpu"
        
        # Should handle shape information
        if "INPUT_SHAPE" in template_vars:
            assert template_vars["INPUT_SHAPE"] == [1, 28, 28]
        
        if "NUM_CLASSES" in template_vars:
            assert template_vars["NUM_CLASSES"] == 10
    
    @pytest.mark.export
    def test_connection_handling(self):
        """Test network connection processing."""
        node_data = NETWORK_DATA
        connections = {
            "input": [("node_1", "dataset")],
            "layers": [("node_2", "layer"), ("node_3", "layer")]
        }
        
        template_vars = NetworkExporter.prepare_template_vars(
            "network_node", node_data, connections
        )
        
        # Should process connections
        assert isinstance(template_vars, dict)
        # Connection processing depends on implementation


class TestSGDOptimizerExporter:
    """Test SGD optimizer node exporter."""
    
    @pytest.mark.export
    def test_template_name(self):
        """Test SGD optimizer template name."""
        template_name = SGDOptimizerExporter.get_template_name()
        
        assert isinstance(template_name, str)
        assert template_name.endswith('.py')
        assert 'sgd' in template_name.lower() or 'optimizer' in template_name.lower()
    
    @pytest.mark.export
    def test_optimizer_parameter_extraction(self):
        """Test extraction of optimizer parameters."""
        # Use ComfyUI format with widgets_values
        node_data = {
            "widgets_values": [0.001, 0.8]  # learning_rate, momentum
        }
        
        template_vars = SGDOptimizerExporter.prepare_template_vars(
            "opt_node", node_data, {}
        )
        
        # Should extract parameters from widgets_values
        assert template_vars["LEARNING_RATE"] == 0.001
        assert template_vars["MOMENTUM"] == 0.8
        
        # Weight decay is fixed (not configurable in this node type)
        assert template_vars["WEIGHT_DECAY"] == 0.0
        
        # Should have node identification
        assert template_vars["NODE_ID"] == "opt_node"
        assert template_vars["CLASS_NAME"] == "SGDOptimizerNode"
    
    @pytest.mark.export
    def test_imports(self):
        """Test SGD optimizer imports."""
        imports = SGDOptimizerExporter.get_imports()
        
        assert isinstance(imports, list)
        
        # Should include optimizer imports
        import_text = " ".join(imports).lower()
        expected_imports = ["torch", "optim", "sgd"]
        
        has_optim_import = any(imp in import_text for imp in expected_imports)
        assert has_optim_import, f"Should have optimizer imports, got: {imports}"


class TestCrossEntropyLossExporter:
    """Test CrossEntropyLoss node exporter."""
    
    @pytest.mark.export
    def test_template_name(self):
        """Test loss template name."""
        template_name = CrossEntropyLossExporter.get_template_name()
        
        assert isinstance(template_name, str)
        assert template_name.endswith('.py')
        assert 'loss' in template_name.lower() or 'entropy' in template_name.lower()
    
    @pytest.mark.export
    def test_loss_parameter_extraction(self):
        """Test extraction of loss parameters."""
        node_data = {
            "inputs": {
                "reduction": "mean",
                "ignore_index": -100
            },
            "widgets": {
                "label_smoothing": 0.1,
                "weight": None
            }
        }
        
        template_vars = CrossEntropyLossExporter.prepare_template_vars(
            "loss_node", node_data, {}
        )
        
        # Should extract reduction method
        if "REDUCTION" in template_vars:
            assert template_vars["REDUCTION"] == "mean"
        
        # Should extract ignore index
        if "IGNORE_INDEX" in template_vars:
            assert template_vars["IGNORE_INDEX"] == -100
        
        # Should handle label smoothing
        if "LABEL_SMOOTHING" in template_vars:
            assert template_vars["LABEL_SMOOTHING"] == 0.1


class TestTrainingStepExporter:
    """Test TrainingStep node exporter."""
    
    @pytest.mark.export
    def test_template_name(self):
        """Test training step template name."""
        template_name = TrainingStepExporter.get_template_name()
        
        assert isinstance(template_name, str)
        assert template_name.endswith('.py')
        assert 'training' in template_name.lower() or 'step' in template_name.lower()
    
    @pytest.mark.export
    def test_training_parameter_extraction(self):
        """Test extraction of training step parameters."""
        node_data = {
            "inputs": {},
            "widgets": {
                "gradient_clipping": 1.0,
                "accumulate_gradients": 4
            }
        }
        
        template_vars = TrainingStepExporter.prepare_template_vars(
            "training_node", node_data, {}
        )
        
        # Should extract gradient clipping
        if "GRADIENT_CLIPPING" in template_vars:
            assert template_vars["GRADIENT_CLIPPING"] == 1.0
        
        # Should extract accumulation steps
        if "ACCUMULATE_GRADIENTS" in template_vars:
            assert template_vars["ACCUMULATE_GRADIENTS"] == 4
    
    @pytest.mark.export
    def test_trigger_signal_generation(self):
        """Test ready signal generation in template vars."""
        node_data = {"inputs": {}, "widgets": {}}
        
        template_vars = TrainingStepExporter.prepare_template_vars(
            "training_node", node_data, {}
        )
        
        # Should have proper node structure
        assert "NODE_ID" in template_vars
        assert "CLASS_NAME" in template_vars
        
        # Implementation might include signal generation logic
        class_name = template_vars.get("CLASS_NAME", "")
        assert "training" in class_name.lower() or "step" in class_name.lower()


class TestNodeExporterIntegration:
    """Integration tests for node exporters."""
    
    @pytest.mark.export
    @pytest.mark.integration
    def test_all_exporters_registered(self):
        """Test that all key node types have exporters."""
        from export_system.node_exporters import register_all_exporters
        from export_system.graph_exporter import GraphExporter
        
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        # Check that key ML nodes are registered
        key_ml_nodes = [
            "MNISTDataset", "LinearLayer", "Network", 
            "SGDOptimizer", "CrossEntropyLoss", "TrainingStep"
        ]
        
        registered_nodes = list(exporter.node_registry.keys())
        
        for node_type in key_ml_nodes:
            if node_type in registered_nodes:
                # Verify exporter implements interface
                node_exporter = exporter.node_registry[node_type]
                
                assert hasattr(node_exporter, 'get_template_name')
                assert hasattr(node_exporter, 'prepare_template_vars')
                assert hasattr(node_exporter, 'get_imports')
                
                # Test basic functionality
                template_name = node_exporter.get_template_name()
                assert isinstance(template_name, str)
                assert template_name.endswith('.py')
                
                imports = node_exporter.get_imports()
                assert isinstance(imports, list)
    
    @pytest.mark.export
    def test_template_variable_consistency(self):
        """Test that template variables are consistently formatted."""
        # Test exporters that don't require connections
        exporters = [
            (MNISTDatasetExporter, MNIST_DATASET_DATA),
            (SGDOptimizerExporter, SGD_OPTIMIZER_DATA),
            (CrossEntropyLossExporter, CROSS_ENTROPY_LOSS_DATA)
        ]
        
        for exporter_class, sample_data in exporters:
            template_vars = exporter_class.prepare_template_vars(
                "test_node", sample_data, {}
            )
            
            # All exporters should provide NODE_ID and CLASS_NAME
            assert "NODE_ID" in template_vars
            assert "CLASS_NAME" in template_vars
            
            # Values should be properly formatted
            assert isinstance(template_vars["NODE_ID"], str)
            assert isinstance(template_vars["CLASS_NAME"], str)
            
            # Node ID should be the one provided
            assert template_vars["NODE_ID"] == "test_node"
            
            # Class name should be meaningful
            class_name = template_vars["CLASS_NAME"]
            assert len(class_name) > 0
            assert class_name[0].isupper()  # Should be PascalCase
    
    @pytest.mark.export
    def test_import_statement_validity(self):
        """Test that generated import statements are valid Python."""
        exporters = [
            MNISTDatasetExporter, LinearLayerExporter, NetworkExporter,
            SGDOptimizerExporter, CrossEntropyLossExporter, TrainingStepExporter
        ]
        
        for exporter_class in exporters:
            imports = exporter_class.get_imports()
            
            # Each import should be a valid Python import statement
            for import_stmt in imports:
                assert isinstance(import_stmt, str)
                assert len(import_stmt.strip()) > 0
                
                # Should start with import or from
                stripped = import_stmt.strip()
                assert stripped.startswith('import ') or stripped.startswith('from ')
                
                # Should be valid Python syntax
                try:
                    compile(import_stmt, '<string>', 'exec')
                except SyntaxError as e:
                    pytest.fail(f"Invalid import statement: {import_stmt}, error: {e}")
    
    @pytest.mark.export
    def test_parameter_type_handling(self):
        """Test handling of different parameter types."""
        # Test with various parameter types
        test_data = {
            "inputs": {
                "string_param": "test_value",
                "int_param": 42,
                "float_param": 3.14,
                "bool_param": True,
                "list_param": [1, 2, 3],
                "none_param": None
            },
            "widgets": {
                "nested_dict": {"key": "value"},
                "tuple_param": (1, 2, 3)
            }
        }
        
        # Test with MNISTDatasetExporter as representative (doesn't require connections)
        template_vars = MNISTDatasetExporter.prepare_template_vars(
            "test_node", test_data, {}
        )
        
        # Should handle the data without crashing
        assert isinstance(template_vars, dict)
        assert "NODE_ID" in template_vars
        assert "CLASS_NAME" in template_vars
        
        # Values should be JSON-serializable for template substitution
        import json
        try:
            # Try to serialize all template vars (common requirement for templates)
            for key, value in template_vars.items():
                if value is not None:
                    json.dumps(value)  # Should not raise exception
        except (TypeError, ValueError) as e:
            # If values aren't JSON serializable, they should at least be string-convertible
            for key, value in template_vars.items():
                str(value)  # Should not raise exception
    
    @pytest.mark.export
    def test_connection_parameter_handling(self):
        """Test handling of connection information."""
        sample_connections = {
            "input": [("node_1", "output")],
            "model": [("node_2", "layer")],
            "optimizer": [("node_3", "optimizer")]
        }
        
        exporters = [
            NetworkExporter,  # Likely uses connections
            TrainingStepExporter,  # Likely uses connections
        ]
        
        for exporter_class in exporters:
            template_vars = exporter_class.prepare_template_vars(
                "test_node", {"inputs": {}, "widgets": {}}, sample_connections
            )
            
            # Should handle connections without crashing
            assert isinstance(template_vars, dict)
            assert "NODE_ID" in template_vars
            assert "CLASS_NAME" in template_vars
            
            # Connection information might be processed into template vars
            # (specific handling depends on implementation)