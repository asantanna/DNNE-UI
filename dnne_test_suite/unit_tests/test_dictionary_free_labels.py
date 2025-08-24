"""
Unit tests for dictionary-free label resolution system.
"""

import unittest
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from export_system.graph_exporter import GraphExporter


class TestDictionaryFreeLabels(unittest.TestCase):
    """Test the dictionary-free label resolution system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.exporter = GraphExporter()
    
    def test_basic_label_resolution(self):
        """Test that basic label connections are resolved correctly."""
        workflow = {
            "nodes": [
                # Source node (not needed for resolution but good for context)
                {
                    "id": 10,
                    "type": "TensorConstant",
                    "class_type": "TensorConstant"
                },
                # Output label
                {
                    "id": 20,
                    "type": "Label",
                    "properties": {
                        "labelName": "TensorConstant(10).output",
                        "labelDirection": "output",
                        "sourceNodeId": 10,
                        "sourceSlotIndex": 0,
                        "sourceSlotName": "output",
                        "sourceSlotType": "TENSOR"
                    }
                },
                # Input label
                {
                    "id": 30,
                    "type": "Label",
                    "properties": {
                        "labelName": "TensorConstant(10).output",
                        "labelDirection": "input",
                        "targetNodeId": 40,
                        "targetSlotIndex": 0,
                        "targetSlotName": "input",
                        "targetSlotType": "TENSOR",
                        "connectedToLabel": "TensorConstant(10).output"
                    }
                },
                # Target node (not needed for resolution but good for context)
                {
                    "id": 40,
                    "type": "Print",
                    "class_type": "Print"
                }
            ],
            "links": []
        }
        
        connections, connections_dict = self.exporter.generate_label_connections(workflow)
        
        # Should have one connection
        self.assertEqual(len(connections), 1)
        
        # Check the connection details
        from_node, from_slot, to_node, to_slot = connections[0]
        self.assertEqual(from_node, "10")
        self.assertEqual(from_slot, 0)
        self.assertEqual(to_node, "40")
        self.assertEqual(to_slot, 0)
        
        # Check the lookup dictionaries
        self.assertIn("40_0", connections_dict["by_input"])
        self.assertIn("10_0", connections_dict["by_output"])
    
    def test_one_to_many_labels(self):
        """Test that one output label can connect to multiple input labels."""
        workflow = {
            "nodes": [
                # Output label
                {
                    "id": 20,
                    "type": "Label",
                    "properties": {
                        "labelName": "Source.output",
                        "labelDirection": "output",
                        "sourceNodeId": 10,
                        "sourceSlotIndex": 0
                    }
                },
                # First input label
                {
                    "id": 30,
                    "type": "Label",
                    "properties": {
                        "labelName": "Source.output",
                        "labelDirection": "input",
                        "targetNodeId": 40,
                        "targetSlotIndex": 0,
                        "connectedToLabel": "Source.output"
                    }
                },
                # Second input label
                {
                    "id": 50,
                    "type": "Label",
                    "properties": {
                        "labelName": "Source.output",
                        "labelDirection": "input",
                        "targetNodeId": 60,
                        "targetSlotIndex": 1,
                        "connectedToLabel": "Source.output"
                    }
                }
            ],
            "links": []
        }
        
        connections, _ = self.exporter.generate_label_connections(workflow)
        
        # Should have two connections
        self.assertEqual(len(connections), 2)
        
        # Both should come from node 10
        sources = [conn[0] for conn in connections]
        self.assertTrue(all(s == "10" for s in sources))
        
        # Should go to different targets
        targets = [(conn[2], conn[3]) for conn in connections]
        self.assertIn(("40", 0), targets)
        self.assertIn(("60", 1), targets)
    
    def test_orphaned_output_label(self):
        """Test that orphaned output labels are detected."""
        workflow = {
            "nodes": [
                {
                    "id": 20,
                    "type": "Label",
                    "properties": {
                        "labelName": "Orphaned.output",
                        "labelDirection": "output"
                        # Missing sourceNodeId, sourceSlotIndex
                    }
                }
            ],
            "links": []
        }
        
        with self.assertRaises(ValueError) as context:
            self.exporter.generate_label_connections(workflow)
        
        self.assertIn("missing source connection info", str(context.exception))
        self.assertIn("node 20", str(context.exception))
    
    def test_orphaned_input_label(self):
        """Test that orphaned input labels are detected."""
        workflow = {
            "nodes": [
                {
                    "id": 30,
                    "type": "Label",
                    "properties": {
                        "labelName": "Orphaned.input",
                        "labelDirection": "input"
                        # Missing targetNodeId, targetSlotIndex
                    }
                }
            ],
            "links": []
        }
        
        with self.assertRaises(ValueError) as context:
            self.exporter.generate_label_connections(workflow)
        
        self.assertIn("missing target connection info", str(context.exception))
        self.assertIn("node 30", str(context.exception))
    
    def test_missing_output_label(self):
        """Test that input labels referencing missing output labels are detected."""
        workflow = {
            "nodes": [
                {
                    "id": 30,
                    "type": "Label",
                    "properties": {
                        "labelName": "Input.label",
                        "labelDirection": "input",
                        "targetNodeId": 40,
                        "targetSlotIndex": 0,
                        "connectedToLabel": "NonExistent.output"
                    }
                }
            ],
            "links": []
        }
        
        with self.assertRaises(ValueError) as context:
            self.exporter.generate_label_connections(workflow)
        
        self.assertIn("missing output label", str(context.exception))
        self.assertIn("NonExistent.output", str(context.exception))
    
    def test_duplicate_output_labels(self):
        """Test that duplicate output labels are detected."""
        workflow = {
            "nodes": [
                {
                    "id": 20,
                    "type": "Label",
                    "properties": {
                        "labelName": "Duplicate.output",
                        "labelDirection": "output",
                        "sourceNodeId": 10,
                        "sourceSlotIndex": 0
                    }
                },
                {
                    "id": 21,
                    "type": "Label",
                    "properties": {
                        "labelName": "Duplicate.output",
                        "labelDirection": "output",
                        "sourceNodeId": 11,
                        "sourceSlotIndex": 0
                    }
                }
            ],
            "links": []
        }
        
        with self.assertRaises(ValueError) as context:
            self.exporter.generate_label_connections(workflow)
        
        self.assertIn("Duplicate output label", str(context.exception))
        self.assertIn("Duplicate.output", str(context.exception))
    
    def test_empty_workflow(self):
        """Test that workflows without labels work correctly."""
        workflow = {
            "nodes": [
                {"id": 1, "type": "TensorConstant"},
                {"id": 2, "type": "Print"}
            ],
            "links": [[1, 1, 0, 2, 0]]
        }
        
        connections, connections_dict = self.exporter.generate_label_connections(workflow)
        
        # Should have no label connections
        self.assertEqual(len(connections), 0)
        self.assertEqual(len(connections_dict["by_input"]), 0)
        self.assertEqual(len(connections_dict["by_output"]), 0)
    
    def test_label_without_properties(self):
        """Test that labels without properties are handled correctly."""
        workflow = {
            "nodes": [
                {
                    "id": 20,
                    "type": "Label"
                    # No properties at all
                }
            ],
            "links": []
        }
        
        with self.assertRaises(ValueError) as context:
            self.exporter.generate_label_connections(workflow)
        
        self.assertIn("missing name or direction", str(context.exception))
        self.assertIn("node 20", str(context.exception))


if __name__ == '__main__':
    unittest.main()