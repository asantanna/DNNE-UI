#!/usr/bin/env python3
"""
Unit tests for subsystem support in override and telemetry arguments.

Tests the parser's ability to handle subsystem names in addition to node IDs
for command-line arguments like --override and --enable-telemetry.
"""

import unittest
import sys
from pathlib import Path

# Add export system to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from export_system.templates.framework.override_parser import parse_override_args, parse_single_override, parse_value


class TestOverrideParser(unittest.TestCase):
    """Test override parser with subsystem support"""
    
    def setUp(self):
        """Set up test subsystem mapping"""
        self.subsystem_to_nodes = {
            "training": ["56", "67", "62"],
            "data": ["37", "50", "65"],
            "network": ["53", "54"],
            "rl": ["66", "68"],
            "robotics": [],  # Empty subsystem
        }
    
    def test_single_node_override(self):
        """Test override with single node ID"""
        configs, errors = parse_override_args("56:learning_rate=0.001")
        self.assertEqual(len(errors), 0)
        self.assertIn("56", configs)
        self.assertEqual(configs["56"]["learning_rate"], 0.001)
    
    def test_subsystem_override(self):
        """Test override with subsystem name"""
        configs, errors = parse_override_args("training:learning_rate=0.001", self.subsystem_to_nodes)
        self.assertEqual(len(errors), 0)
        # Should expand to all training nodes
        self.assertIn("56", configs)
        self.assertIn("67", configs)
        self.assertIn("62", configs)
        self.assertEqual(configs["56"]["learning_rate"], 0.001)
        self.assertEqual(configs["67"]["learning_rate"], 0.001)
        self.assertEqual(configs["62"]["learning_rate"], 0.001)
    
    def test_mixed_override(self):
        """Test override with mix of subsystems and node IDs"""
        override_str = "training:learning_rate=0.001,99:checkpoint_enabled=True"
        configs, errors = parse_override_args(override_str, self.subsystem_to_nodes)
        self.assertEqual(len(errors), 0)
        # Training nodes should have learning_rate
        self.assertEqual(configs["56"]["learning_rate"], 0.001)
        # Specific node should have checkpoint_enabled
        self.assertEqual(configs["99"]["checkpoint_enabled"], True)
    
    def test_multiple_subsystems(self):
        """Test override with multiple subsystems"""
        override_str = "training:lr=0.001,rl:gamma=0.99"
        configs, errors = parse_override_args(override_str, self.subsystem_to_nodes)
        self.assertEqual(len(errors), 0)
        # Training nodes
        self.assertEqual(configs["56"]["lr"], 0.001)
        self.assertEqual(configs["67"]["lr"], 0.001)
        # RL nodes
        self.assertEqual(configs["66"]["gamma"], 0.99)
        self.assertEqual(configs["68"]["gamma"], 0.99)
    
    def test_empty_subsystem(self):
        """Test override with empty subsystem"""
        result = parse_single_override("robotics:enable=True", self.subsystem_to_nodes)
        self.assertIsInstance(result, str)  # Should be error message
        self.assertIn("no nodes", result.lower())
    
    def test_unknown_subsystem(self):
        """Test override with unknown subsystem (treated as node ID)"""
        configs, errors = parse_override_args("unknown:param=value", self.subsystem_to_nodes)
        self.assertEqual(len(errors), 0)
        # Should treat "unknown" as a node ID
        self.assertIn("unknown", configs)
        self.assertEqual(configs["unknown"]["param"], "value")
    
    def test_value_parsing(self):
        """Test various value types"""
        # Boolean
        self.assertEqual(parse_value("True"), True)
        self.assertEqual(parse_value("true"), True)
        self.assertEqual(parse_value("FALSE"), False)
        
        # Integer
        self.assertEqual(parse_value("42"), 42)
        self.assertEqual(parse_value("-100"), -100)
        
        # Float
        self.assertEqual(parse_value("3.14"), 3.14)
        self.assertEqual(parse_value("0.001"), 0.001)
        
        # String
        self.assertEqual(parse_value("hello"), "hello")
        self.assertEqual(parse_value('"hello world"'), "hello world")
        self.assertEqual(parse_value("'quoted string'"), "quoted string")
    
    def test_complex_values(self):
        """Test complex value strings"""
        override_str = '56:message="Hello, World!",57:path="/home/user/file.txt"'
        configs, errors = parse_override_args(override_str)
        self.assertEqual(len(errors), 0)
        self.assertEqual(configs["56"]["message"], "Hello, World!")
        self.assertEqual(configs["57"]["path"], "/home/user/file.txt")
    
    def test_invalid_format(self):
        """Test invalid override formats"""
        # Missing equals
        configs, errors = parse_override_args("56:param")
        self.assertEqual(len(configs), 0)
        self.assertEqual(len(errors), 1)
        self.assertIn("Invalid", errors[0])
        
        # Missing colon
        configs, errors = parse_override_args("56=value")
        self.assertEqual(len(configs), 0)
        self.assertEqual(len(errors), 1)
        self.assertIn("Invalid", errors[0])
    
    def test_subsystem_same_param_different_values(self):
        """Test setting same parameter to different values for different subsystems"""
        override_str = "training:batch_size=32,data:batch_size=64"
        configs, errors = parse_override_args(override_str, self.subsystem_to_nodes)
        self.assertEqual(len(errors), 0)
        
        # Training nodes should have batch_size=32
        for node_id in self.subsystem_to_nodes["training"]:
            self.assertEqual(configs[node_id]["batch_size"], 32)
        
        # Data nodes should have batch_size=64
        for node_id in self.subsystem_to_nodes["data"]:
            self.assertEqual(configs[node_id]["batch_size"], 64)
    
    def test_node_override_after_subsystem(self):
        """Test that specific node override can override subsystem setting"""
        override_str = "training:learning_rate=0.001,56:learning_rate=0.01"
        configs, errors = parse_override_args(override_str, self.subsystem_to_nodes)
        self.assertEqual(len(errors), 0)
        
        # Node 56 should have the specific override
        self.assertEqual(configs["56"]["learning_rate"], 0.01)
        # Other training nodes should have subsystem value
        self.assertEqual(configs["67"]["learning_rate"], 0.001)
        self.assertEqual(configs["62"]["learning_rate"], 0.001)


class TestTelemetryExpansion(unittest.TestCase):
    """Test telemetry enablement with subsystem support"""
    
    def setUp(self):
        """Set up test environment"""
        self.workflow_nodes = {
            "56": {"type": "SGDOptimizerNode", "subsystem": "training"},
            "67": {"type": "EpochTrackerNode", "subsystem": "training"},
            "37": {"type": "MNISTDatasetNode", "subsystem": "data"},
            "50": {"type": "BatchSamplerNode", "subsystem": "data"},
        }
        
        self.subsystem_to_nodes = {
            "training": ["56", "67"],
            "data": ["37", "50"],
        }
    
    def test_enable_all_telemetry(self):
        """Test enabling telemetry for all nodes"""
        enable_telemetry = "all"
        node_configs = {}
        
        # Simulate the runner.py logic
        if enable_telemetry == "all":
            for node_id in self.workflow_nodes.keys():
                node_configs.setdefault(node_id, {})["telemetry_enabled"] = True
        
        # All nodes should have telemetry enabled
        for node_id in self.workflow_nodes:
            self.assertTrue(node_configs[node_id]["telemetry_enabled"])
    
    def test_enable_subsystem_telemetry(self):
        """Test enabling telemetry for specific subsystem"""
        enable_telemetry = "training"
        node_configs = {}
        
        # Simulate the runner.py logic
        if enable_telemetry in self.subsystem_to_nodes:
            for node_id in self.subsystem_to_nodes[enable_telemetry]:
                node_configs.setdefault(node_id, {})["telemetry_enabled"] = True
        
        # Only training nodes should have telemetry
        self.assertTrue(node_configs.get("56", {}).get("telemetry_enabled", False))
        self.assertTrue(node_configs.get("67", {}).get("telemetry_enabled", False))
        self.assertFalse(node_configs.get("37", {}).get("telemetry_enabled", False))
        self.assertFalse(node_configs.get("50", {}).get("telemetry_enabled", False))
    
    def test_mixed_telemetry(self):
        """Test enabling telemetry for mix of subsystems and nodes"""
        enable_telemetry = "training,37"
        node_configs = {}
        
        # Simulate the runner.py logic
        for target in enable_telemetry.split(","):
            target = target.strip()
            if target in self.subsystem_to_nodes:
                for node_id in self.subsystem_to_nodes[target]:
                    node_configs.setdefault(node_id, {})["telemetry_enabled"] = True
            elif target in self.workflow_nodes:
                node_configs.setdefault(target, {})["telemetry_enabled"] = True
        
        # Training nodes and node 37 should have telemetry
        self.assertTrue(node_configs.get("56", {}).get("telemetry_enabled", False))
        self.assertTrue(node_configs.get("67", {}).get("telemetry_enabled", False))
        self.assertTrue(node_configs.get("37", {}).get("telemetry_enabled", False))
        self.assertFalse(node_configs.get("50", {}).get("telemetry_enabled", False))


class TestBackwardsCompatibility(unittest.TestCase):
    """Test that old command patterns still work"""
    
    def test_pure_node_ids(self):
        """Test that pure node ID overrides still work"""
        override_str = "56:param1=value1,67:param2=value2"
        configs, errors = parse_override_args(override_str)
        self.assertEqual(len(errors), 0)
        self.assertEqual(configs["56"]["param1"], "value1")
        self.assertEqual(configs["67"]["param2"], "value2")
    
    def test_numeric_node_ids(self):
        """Test that numeric node IDs are handled correctly"""
        configs, errors = parse_override_args("123:test=456")
        self.assertEqual(len(errors), 0)
        self.assertEqual(configs["123"]["test"], 456)
    
    def test_old_telemetry_format(self):
        """Test old telemetry format with just node IDs"""
        enable_telemetry = "56,67,68"
        workflow_nodes = {"56": {}, "67": {}, "68": {}, "99": {}}
        node_configs = {}
        
        # Simulate old logic
        for node_id in enable_telemetry.split(","):
            node_id = node_id.strip()
            if node_id in workflow_nodes:
                node_configs.setdefault(node_id, {})["telemetry_enabled"] = True
        
        self.assertTrue(node_configs["56"]["telemetry_enabled"])
        self.assertTrue(node_configs["67"]["telemetry_enabled"])
        self.assertTrue(node_configs["68"]["telemetry_enabled"])
        self.assertNotIn("99", node_configs)


if __name__ == "__main__":
    unittest.main()