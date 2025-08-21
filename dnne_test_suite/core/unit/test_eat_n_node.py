#!/usr/bin/env python3
"""
Unit tests for the Eat_N synchronization node
"""

import pytest
from pathlib import Path
import sys
import importlib

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from custom_nodes.eat_n_visnode import Eat_NNode
from export_system.node_exporters.eat_n_exporter import Eat_NExporter


class TestEat_NNode:
    """Test suite for Eat_N node UI implementation"""
    
    def test_node_registration(self):
        """Test that Eat_N node is properly registered"""
        from custom_nodes import NODE_CLASS_MAPPINGS
        assert "Eat_N" in NODE_CLASS_MAPPINGS
        assert NODE_CLASS_MAPPINGS["Eat_N"] == Eat_NNode
    
    def test_node_properties(self):
        """Test Eat_N node properties and configuration"""
        assert Eat_NNode.CATEGORY == "utility"
        assert Eat_NNode.RETURN_TYPES == ("TENSOR", "EAT_N_TRIGGER")
        assert Eat_NNode.RETURN_NAMES == ("output", "trigger")
        assert Eat_NNode.FUNCTION is None  # DNNE nodes don't execute
    
    def test_input_types(self):
        """Test Eat_N node input configuration"""
        input_types = Eat_NNode.INPUT_TYPES()
        
        # Check required inputs
        assert "required" in input_types
        required = input_types["required"]
        assert "num_to_eat" in required
        assert "trigger_mode" in required
        
        # Check num_to_eat configuration
        num_to_eat_config = required["num_to_eat"]
        assert num_to_eat_config[0] == "INT"
        assert num_to_eat_config[1]["default"] == 1
        assert num_to_eat_config[1]["min"] == 1
        assert num_to_eat_config[1]["max"] == 100
        
        # Check trigger_mode configuration
        trigger_mode_config = required["trigger_mode"]
        assert trigger_mode_config[0] == ["every_eat", "last_only"]
        assert trigger_mode_config[1]["default"] == "every_eat"
        
        # Check optional inputs
        assert "optional" in input_types
        optional = input_types["optional"]
        assert "input" in optional
        assert optional["input"][0] == "*TENSOR"
    
    def test_node_colors(self):
        """Test that node has proper color configuration"""
        assert hasattr(Eat_NNode, "COLOR")
        assert hasattr(Eat_NNode, "BGCOLOR")
        # Colors should be from utility category
        from custom_nodes.utils.node_colors import get_node_colors
        utility_colors = get_node_colors("utility")
        assert Eat_NNode.COLOR == utility_colors["color"]
        assert Eat_NNode.BGCOLOR == utility_colors["bgcolor"]


class TestEat_NExporter:
    """Test suite for Eat_N node exporter"""
    
    def test_exporter_template_name(self):
        """Test that exporter specifies correct template"""
        assert Eat_NExporter.get_template_name() == "nodes/eat_n_node_queue.tpl"
    
    def test_exporter_input_output_names(self):
        """Test input and output name definitions"""
        assert Eat_NExporter.get_input_names() == ["input"]
        assert Eat_NExporter.get_output_names() == ["output", "trigger"]
    
    def test_exporter_imports(self):
        """Test that exporter provides necessary imports"""
        imports = Eat_NExporter.get_imports()
        assert "import torch" in imports
        assert "import asyncio" in imports
        assert "from typing import Dict, Any, Optional" in imports
    
    def test_exporter_subsystem(self):
        """Test that exporter belongs to correct subsystem"""
        from export_system.subsystems import SUBSYSTEM_CONTROL
        assert Eat_NExporter.get_subsystem() == SUBSYSTEM_CONTROL
    
    def test_exporter_prepare_template_vars(self):
        """Test template variable preparation"""
        # Mock node data
        node_data = {
            "widget_values": [3, "last_only"]
        }
        
        # Prepare template variables
        template_vars = Eat_NExporter.prepare_template_vars(
            node_id="test_node",
            node_data=node_data,
            connections={}
        )
        
        # Check template variables
        assert template_vars["NODE_ID"] == "test_node"
        assert template_vars["CLASS_NAME"] == "Eat_NNode"
        assert template_vars["NUM_TO_EAT"] == "3"
        assert template_vars["TRIGGER_MODE"] == '"last_only"'
    
    def test_exporter_default_values(self):
        """Test that exporter handles missing widget values with defaults"""
        # Mock node data with no widget values
        node_data = {"widget_values": []}
        
        # Prepare template variables
        template_vars = Eat_NExporter.prepare_template_vars(
            node_id="test_node",
            node_data=node_data,
            connections={}
        )
        
        # Should use defaults
        assert template_vars["NUM_TO_EAT"] == "1"
        assert template_vars["TRIGGER_MODE"] == '"every_eat"'


class TestEat_NExportTemplate:
    """Test suite for Eat_N export template"""
    
    def test_template_exists(self):
        """Test that the export template file exists"""
        template_path = Path("export_system/templates/nodes/eat_n_node_queue.tpl")
        assert template_path.exists(), f"Template not found at {template_path}"
    
    def test_template_substitution(self):
        """Test that template can be properly substituted"""
        template_path = Path("export_system/templates/nodes/eat_n_node_queue.tpl")
        template_content = template_path.read_text()
        
        # Check for required template variables
        assert "{NODE_ID}" in template_content
        assert "{CLASS_NAME}" in template_content
        assert "{NUM_TO_EAT}" in template_content
        assert "{TRIGGER_MODE}" in template_content
        
        # Test substitution
        substituted = template_content.replace("{NODE_ID}", "test_node")
        substituted = substituted.replace("{CLASS_NAME}", "Eat_NNode")
        substituted = substituted.replace("{NUM_TO_EAT}", "2")
        substituted = substituted.replace("{TRIGGER_MODE}", '"every_eat"')
        
        # Check that substitution worked
        assert "test_node" in substituted
        assert "Eat_NNode_test_node" in substituted
        assert "self.num_to_eat = 2" in substituted
        assert 'self.trigger_mode = "every_eat"' in substituted
    
    def test_template_structure(self):
        """Test that template has proper async queue structure"""
        template_path = Path("export_system/templates/nodes/eat_n_node_queue.tpl")
        template_content = template_path.read_text()
        
        # Check for queue node inheritance
        assert "class {CLASS_NAME}_{NODE_ID}(QueueNode):" in template_content
        
        # Check for required methods
        assert "async def run(self):" in template_content
        assert "async def handle_input(self, data):" in template_content
        assert "async def compute(self, **inputs)" in template_content
        
        # Check for state management
        assert "self.counter" in template_content
        assert "self.is_passthrough" in template_content
        
        # Check for trigger emission
        assert 'await self.send_output("trigger"' in template_content
        assert 'await self.send_output("output"' in template_content