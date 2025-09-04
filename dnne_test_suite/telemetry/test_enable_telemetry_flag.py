#!/usr/bin/env python3
"""
Test --enable-telemetry flag functionality.
Verifies that the flag correctly sets telemetry_level instead of telemetry_enabled.
"""

import unittest
import json
from pathlib import Path
import sys
import os
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from export_system.graph_exporter import GraphExporter

class TestEnableTelemetryFlag(unittest.TestCase):
    """Test that --enable-telemetry flag works correctly"""
    
    def setUp(self):
        """Create test workflow and exporter"""
        self.exporter = GraphExporter()
        # Use a test directory within the required export path
        self.test_dir = Path("export_system/exports/test_telemetry_flag_test")
        
        # Create a simple test workflow with EpochTracker
        # Since EpochTracker has no required inputs anymore (all optional),
        # we can have a standalone node
        self.workflow = {
            "id": "test-workflow",
            "metadata": {
                "workflow_name": "test_telemetry_workflow",
                "skip-slot-correction": True,  # Skip slot correction for test workflow
                "dnne_test": True,
                "created_at": "2025-09-03T12:00:00Z",
                "workflow_id": "test-telemetry-123"
            },
            "nodes": [
                {
                    "id": 68,
                    "type": "EpochTracker",
                    "inputs": [],  # No inputs connected - all are optional
                    "widgets_values": [10, "off"]  # max_epochs=10, telemetry_level="off"
                }
            ],
            "links": []
        }
    
    def tearDown(self):
        """Clean up test directory"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_enable_telemetry_sets_level_not_enabled(self):
        """Verify --enable-telemetry sets telemetry_level, not telemetry_enabled"""
        # Export the workflow
        output_path = self.test_dir
        self.exporter.export_workflow(self.workflow, output_path)
        
        # Read the generated runner.py
        runner_path = output_path / "runner.py"
        with open(runner_path, 'r') as f:
            runner_code = f.read()
        
        # Check that code sets telemetry_level, not telemetry_enabled
        self.assertIn("'telemetry_level'] = 'essential'", runner_code)
        self.assertNotIn("'telemetry_enabled'] = True", runner_code)
        self.assertNotIn("telemetry_enabled", runner_code)
        
        # Check the help text is updated
        self.assertIn("Enable telemetry reporting at essential level", runner_code)
    
    def test_enable_telemetry_checks_existing_level(self):
        """Verify --enable-telemetry doesn't override existing telemetry_level"""
        # Export the workflow
        output_path = self.test_dir
        self.exporter.export_workflow(self.workflow, output_path)
        
        # Read the generated runner.py
        runner_path = output_path / "runner.py"
        with open(runner_path, 'r') as f:
            runner_code = f.read()
        
        # Check that code checks for existing telemetry_level
        self.assertIn("if 'telemetry_level' not in node_configs.get(node_id, {})", runner_code)
        self.assertIn("# Only set telemetry_level if not already set", runner_code)
    
    def test_node_checks_telemetry_level(self):
        """Verify nodes check telemetry_level, not telemetry_enabled"""
        # Export the workflow
        output_path = self.test_dir
        self.exporter.export_workflow(self.workflow, output_path)
        
        # Read the generated EpochTracker node
        nodes_path = output_path / "nodes" / "epoch_tracker_queue.py"
        with open(nodes_path, 'r') as f:
            node_code = f.read()
        
        # Check that node uses telemetry_level
        self.assertIn("self.telemetry_level", node_code)
        self.assertIn('self.telemetry_level = "off"', node_code)  # From widget value
        self.assertIn("g.get_node_config(self.node_id, 'telemetry_level'", node_code)
        
        # Check that node doesn't use telemetry_enabled
        self.assertNotIn("telemetry_enabled", node_code)
    
    def test_runner_args_json_updated(self):
        """Verify runner_args.json has updated description"""
        runner_args_path = Path("export_system/templates/framework/runner_args.json")
        with open(runner_args_path, 'r') as f:
            runner_args = json.load(f)
        
        # Check enable_telemetry description is updated
        telemetry_desc = runner_args["arguments"]["enable_telemetry"]["description"]
        self.assertIn("essential level", telemetry_desc)
        self.assertIn("For other levels use --override", telemetry_desc)
    
    def test_arg_parser_help_updated(self):
        """Verify arg_parser.tpl has updated help text"""
        arg_parser_path = Path("export_system/templates/framework/arg_parser.tpl")
        with open(arg_parser_path, 'r') as f:
            arg_parser_code = f.read()
        
        # Check help text mentions essential level
        self.assertIn("Enable telemetry reporting at essential level", arg_parser_code)
        self.assertIn("For other levels use --override", arg_parser_code)

if __name__ == "__main__":
    unittest.main()