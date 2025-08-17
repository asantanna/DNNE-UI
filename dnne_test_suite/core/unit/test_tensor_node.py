#!/usr/bin/env python3
"""
Unit tests for TensorNode
"""

import unittest
import sys
import os
import json
import tempfile
import shutil
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from custom_nodes.tensor_visnode import TensorNode
from export_system.node_exporters.tensor_exporter import TensorExporter


class TestTensorNode(unittest.TestCase):
    """Test cases for TensorNode functionality"""
    
    def test_node_registration(self):
        """Test that the node is properly registered"""
        from custom_nodes.tensor_visnode import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        
        self.assertIn("Tensor", NODE_CLASS_MAPPINGS)
        self.assertEqual(NODE_CLASS_MAPPINGS["Tensor"], TensorNode)
        self.assertIn("Tensor", NODE_DISPLAY_NAME_MAPPINGS)
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["Tensor"], "Tensor")
    
    def test_node_properties(self):
        """Test node properties and configuration"""
        self.assertEqual(TensorNode.CATEGORY, "utils")
        self.assertEqual(TensorNode.RETURN_TYPES, ("TENSOR",))
        self.assertEqual(TensorNode.RETURN_NAMES, ("tensor",))
        self.assertIsNone(TensorNode.FUNCTION)
        
        # Test input types
        input_types = TensorNode.INPUT_TYPES()
        self.assertIn("required", input_types)
        required = input_types["required"]
        
        # Check all required widgets exist
        self.assertIn("tensor_dims", required)
        self.assertIn("fill_mode", required)
        self.assertIn("custom_fill", required)
        self.assertIn("dtype", required)
        self.assertIn("seed", required)
    
    def test_dimension_parsing(self):
        """Test various dimension string formats"""
        test_cases = [
            ("10", [10]),
            ("2,3", [2, 3]),
            ("[2,3,4]", [2, 3, 4]),
            ("  5  ", [5]),
            ("[10, 20, 30]", [10, 20, 30]),
            ("100", [100])
        ]
        
        for dims_str, expected in test_cases:
            node_data = {
                "widgets_values": {
                    "tensor_dims": dims_str,
                    "fill_mode": "zeros",
                    "custom_fill": 0.0,
                    "dtype": "float32",
                    "seed": -1
                }
            }
            
            template_vars = TensorExporter.prepare_template_vars(
                "test_node", node_data, {}
            )
            
            # Check that dimensions are properly formatted
            self.assertIn("TENSOR_DIMS", template_vars)
            # Should be wrapped in brackets
            self.assertTrue(template_vars["TENSOR_DIMS"].startswith("["))
            self.assertTrue(template_vars["TENSOR_DIMS"].endswith("]"))
    
    def test_fill_modes(self):
        """Test all supported fill modes"""
        fill_modes = [
            "zeros", "ones", "uniform", "normal",
            "kaiming_normal", "kaiming_uniform",
            "xavier_normal", "xavier_uniform", "custom"
        ]
        
        input_types = TensorNode.INPUT_TYPES()
        available_modes = input_types["required"]["fill_mode"][0]
        
        for mode in fill_modes:
            self.assertIn(mode, available_modes, f"Fill mode '{mode}' not in available modes")
    
    def test_dtype_options(self):
        """Test all supported data types"""
        dtypes = ["float32", "float64", "int32", "int64", "bool"]
        
        input_types = TensorNode.INPUT_TYPES()
        available_dtypes = input_types["required"]["dtype"][0]
        
        for dtype in dtypes:
            self.assertIn(dtype, available_dtypes, f"Data type '{dtype}' not in available types")
    
    def test_exporter_configuration(self):
        """Test TensorExporter configuration"""
        # Test template name
        self.assertEqual(TensorExporter.get_template_name(), "nodes/tensor_queue.tpl")
        
        # Test imports
        imports = TensorExporter.get_imports()
        self.assertIn("import torch", imports)
        self.assertIn("import torch.nn.init as init", imports)
        self.assertIn("import asyncio", imports)
        
        # Test input/output names
        self.assertEqual(TensorExporter.get_input_names(), [])
        self.assertEqual(TensorExporter.get_output_names(), ["tensor"])
    
    def test_template_variable_preparation(self):
        """Test template variable preparation with various configurations"""
        test_configs = [
            {
                "tensor_dims": "10",
                "fill_mode": "zeros",
                "custom_fill": 0.0,
                "dtype": "float32",
                "seed": -1
            },
            {
                "tensor_dims": "2,3,4",
                "fill_mode": "normal",
                "custom_fill": 1.5,
                "dtype": "float64",
                "seed": 42
            },
            {
                "tensor_dims": "[100, 200]",
                "fill_mode": "custom",
                "custom_fill": -0.5,
                "dtype": "int32",
                "seed": 12345
            }
        ]
        
        for config in test_configs:
            node_data = {"widgets_values": config}
            template_vars = TensorExporter.prepare_template_vars(
                "test_node", node_data, {}
            )
            
            # Check all required template variables are present
            self.assertIn("NODE_ID", template_vars)
            self.assertIn("CLASS_NAME", template_vars)
            self.assertIn("TENSOR_DIMS", template_vars)
            self.assertIn("FILL_MODE", template_vars)
            self.assertIn("CUSTOM_FILL", template_vars)
            self.assertIn("DTYPE", template_vars)
            self.assertIn("SEED", template_vars)
            
            # Check values match configuration
            self.assertEqual(template_vars["NODE_ID"], "test_node")
            self.assertEqual(template_vars["CLASS_NAME"], "TensorNode")
            self.assertEqual(template_vars["FILL_MODE"], config["fill_mode"])
            self.assertEqual(template_vars["DTYPE"], config["dtype"])
            self.assertEqual(template_vars["SEED"], str(config["seed"]))
            self.assertEqual(template_vars["CUSTOM_FILL"], str(config["custom_fill"]))
    
    def test_seed_reproducibility(self):
        """Test that seed parameter ensures reproducibility"""
        # This test would require actual execution of the generated code
        # For now, we just verify the seed is properly passed through
        node_data = {
            "widgets_values": {
                "tensor_dims": "10,10",
                "fill_mode": "normal",
                "custom_fill": 0.0,
                "dtype": "float32",
                "seed": 42
            }
        }
        
        template_vars = TensorExporter.prepare_template_vars(
            "test_node", node_data, {}
        )
        
        self.assertEqual(template_vars["SEED"], "42")
    
    def test_custom_fill_value(self):
        """Test custom fill value is properly handled"""
        test_values = [0.0, 1.0, -1.0, 3.14159, -999.99, 1e-6]
        
        for value in test_values:
            node_data = {
                "widgets_values": {
                    "tensor_dims": "5,5",
                    "fill_mode": "custom",
                    "custom_fill": value,
                    "dtype": "float32",
                    "seed": -1
                }
            }
            
            template_vars = TensorExporter.prepare_template_vars(
                "test_node", node_data, {}
            )
            
            self.assertEqual(template_vars["CUSTOM_FILL"], str(value))


if __name__ == "__main__":
    unittest.main()