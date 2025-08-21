#!/usr/bin/env python3
"""
Unit tests for BarrierNode
"""

import unittest
import sys
import os
import json
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from custom_nodes.barrier_visnode import BarrierNode
from export_system.node_exporters.barrier_exporter import BarrierExporter


class TestBarrierNode(unittest.TestCase):
    """Test cases for BarrierNode functionality"""
    
    def test_node_registration(self):
        """Test that the node is properly registered"""
        from custom_nodes.barrier_visnode import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        
        self.assertIn("Barrier", NODE_CLASS_MAPPINGS)
        self.assertEqual(NODE_CLASS_MAPPINGS["Barrier"], BarrierNode)
        self.assertIn("Barrier", NODE_DISPLAY_NAME_MAPPINGS)
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["Barrier"], "Barrier")
    
    def test_node_properties(self):
        """Test node properties and configuration"""
        self.assertEqual(BarrierNode.CATEGORY, "utility")
        self.assertEqual(BarrierNode.RETURN_TYPES, ("HELD_TENSOR",))
        self.assertEqual(BarrierNode.RETURN_NAMES, ("output",))
        self.assertIsNone(BarrierNode.FUNCTION)
        
        # Test input types
        input_types = BarrierNode.INPUT_TYPES()
        self.assertIn("required", input_types)
        self.assertIn("optional", input_types)
        
        required = input_types["required"]
        optional = input_types["optional"]
        
        # Check required widget
        self.assertIn("hold_mode", required)
        hold_mode_config = required["hold_mode"]
        self.assertEqual(hold_mode_config[0], ["FIFO"])
        self.assertEqual(hold_mode_config[1]["default"], "FIFO")
        
        # Check optional inputs
        self.assertIn("input", optional)
        self.assertIn("release", optional)
        
        # Verify input types
        input_config = optional["input"]
        self.assertEqual(input_config[0], "*TENSOR")
        
        release_config = optional["release"]
        self.assertEqual(release_config[0], "*TRIGGER")
    
    def test_exporter_template_name(self):
        """Test that exporter returns correct template name"""
        template_name = BarrierExporter.get_template_name()
        self.assertEqual(template_name, "nodes/barrier_node_queue.tpl")
    
    def test_exporter_input_output_names(self):
        """Test exporter input/output names"""
        input_names = BarrierExporter.get_input_names()
        self.assertEqual(input_names, ["input", "release"])
        
        output_names = BarrierExporter.get_output_names()
        self.assertEqual(output_names, ["output"])
    
    def test_exporter_imports(self):
        """Test that exporter provides necessary imports"""
        imports = BarrierExporter.get_imports()
        
        # Check for required imports
        self.assertIn("import torch", imports)
        self.assertIn("import asyncio", imports)
        self.assertIn("from collections import deque", imports)
        self.assertIn("from typing import Dict, Any, Optional", imports)
    
    def test_exporter_prepare_template_vars(self):
        """Test template variable preparation"""
        # Test with default hold_mode
        node_data = {
            "widget_values": ["FIFO"]
        }
        
        template_vars = BarrierExporter.prepare_template_vars(
            "barrier_1", node_data, {}, None, None, None
        )
        
        self.assertEqual(template_vars["NODE_ID"], "barrier_1")
        self.assertEqual(template_vars["CLASS_NAME"], "BarrierNode")
        self.assertEqual(template_vars["HOLD_MODE"], '"FIFO"')
        
        # Test with no widget values (should default to FIFO)
        node_data_empty = {"widget_values": []}
        template_vars_empty = BarrierExporter.prepare_template_vars(
            "barrier_2", node_data_empty, {}, None, None, None
        )
        
        self.assertEqual(template_vars_empty["HOLD_MODE"], '"FIFO"')
    
    def test_exporter_subsystem(self):
        """Test that exporter returns correct subsystem"""
        from export_system.subsystems import SUBSYSTEM_CONTROL
        subsystem = BarrierExporter.get_subsystem()
        self.assertEqual(subsystem, SUBSYSTEM_CONTROL)


class TestBarrierExportTemplate(unittest.TestCase):
    """Test the exported template code generation"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.template_path = os.path.join(
            os.path.dirname(__file__), 
            '../../../export_system/templates/nodes/barrier_node_queue.tpl'
        )
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_template_exists(self):
        """Test that the template file exists"""
        self.assertTrue(os.path.exists(self.template_path), 
                       f"Template file not found at {self.template_path}")
    
    def test_template_substitution(self):
        """Test that template variables are properly substituted"""
        with open(self.template_path, 'r') as f:
            template_content = f.read()
        
        # Check that template contains expected placeholders
        self.assertIn("{NODE_ID}", template_content)
        self.assertIn("{CLASS_NAME}", template_content)
        self.assertIn("{HOLD_MODE}", template_content)
        
        # Test substitution
        substituted = template_content.replace("{NODE_ID}", "test_barrier_1")
        substituted = substituted.replace("{CLASS_NAME}", "BarrierNode")
        substituted = substituted.replace("{HOLD_MODE}", '"FIFO"')
        
        # Verify substituted content
        self.assertIn("class BarrierNode_test_barrier_1(QueueNode):", substituted)
        self.assertIn('self.hold_mode = "FIFO"', substituted)
        self.assertIn("self.fifo_queue = deque()", substituted)
        self.assertIn("self.release_count = 0", substituted)


if __name__ == '__main__':
    unittest.main()