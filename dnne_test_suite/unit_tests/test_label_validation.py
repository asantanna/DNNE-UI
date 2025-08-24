"""
Unit tests for label validation in the export system.
"""

import unittest
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from export_system.graph_exporter import GraphExporter


class TestLabelValidation(unittest.TestCase):
    """Test label validation error detection and reporting."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.exporter = GraphExporter()
    
    def test_validate_orphaned_output_label(self):
        """Test that orphaned output labels produce clear error messages."""
        workflow = {
            "nodes": [
                {
                    "id": 20,
                    "type": "Label",
                    "properties": {
                        "labelName": "TensorConstant(10).output",
                        "labelDirection": "output"
                        # Missing sourceNodeId, sourceSlotIndex
                    }
                }
            ],
            "links": []
        }
        
        with self.assertRaises(ValueError) as context:
            self.exporter.generate_label_connections(workflow)
        
        error_msg = str(context.exception)
        self.assertIn("orphaned", error_msg.lower())
        self.assertIn("missing source connection info", error_msg)
        self.assertIn("node 20", error_msg)
        self.assertIn("Please delete this orphaned label", error_msg)
    
    def test_validate_orphaned_input_label(self):
        """Test that orphaned input labels produce clear error messages."""
        workflow = {
            "nodes": [
                {
                    "id": 30,
                    "type": "Label",
                    "properties": {
                        "labelName": "SomeLabel",
                        "labelDirection": "input",
                        "connectedToLabel": "TensorConstant(10).output"
                        # Missing targetNodeId, targetSlotIndex
                    }
                }
            ],
            "links": []
        }
        
        with self.assertRaises(ValueError) as context:
            self.exporter.generate_label_connections(workflow)
        
        error_msg = str(context.exception)
        self.assertIn("orphaned", error_msg.lower())
        self.assertIn("missing target connection info", error_msg)
        self.assertIn("node 30", error_msg)
        self.assertIn("Please delete this orphaned label", error_msg)
    
    def test_validate_missing_output_label_reference(self):
        """Test that input labels referencing non-existent output labels are detected."""
        workflow = {
            "nodes": [
                {
                    "id": 30,
                    "type": "Label",
                    "properties": {
                        "labelName": "InputLabel",
                        "labelDirection": "input",
                        "targetNodeId": 40,
                        "targetSlotIndex": 0,
                        "targetSlotName": "input",
                        "connectedToLabel": "NonExistent.output"
                    }
                }
            ],
            "links": []
        }
        
        with self.assertRaises(ValueError) as context:
            self.exporter.generate_label_connections(workflow)
        
        error_msg = str(context.exception)
        self.assertIn("missing output label", error_msg)
        self.assertIn("NonExistent.output", error_msg)
        self.assertIn("node 30", error_msg)
        self.assertIn("Please create the output label or delete this input label", error_msg)
    
    def test_validate_duplicate_output_labels(self):
        """Test that duplicate output label names are detected."""
        workflow = {
            "nodes": [
                {
                    "id": 20,
                    "type": "Label",
                    "properties": {
                        "labelName": "TensorConstant(10).output",
                        "labelDirection": "output",
                        "sourceNodeId": 10,
                        "sourceSlotIndex": 0
                    }
                },
                {
                    "id": 21,
                    "type": "Label",
                    "properties": {
                        "labelName": "TensorConstant(10).output",  # Same name!
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
        
        error_msg = str(context.exception)
        self.assertIn("Duplicate output label", error_msg)
        self.assertIn("TensorConstant(10).output", error_msg)
        self.assertIn("Please remove one of the duplicate", error_msg)
    
    def test_validate_label_missing_name(self):
        """Test that labels without names are detected."""
        workflow = {
            "nodes": [
                {
                    "id": 20,
                    "type": "Label",
                    "properties": {
                        "labelDirection": "output",
                        "sourceNodeId": 10,
                        "sourceSlotIndex": 0
                        # Missing labelName
                    }
                }
            ],
            "links": []
        }
        
        with self.assertRaises(ValueError) as context:
            self.exporter.generate_label_connections(workflow)
        
        error_msg = str(context.exception)
        self.assertIn("missing name or direction", error_msg)
        self.assertIn("node 20", error_msg)
    
    def test_validate_label_missing_direction(self):
        """Test that labels without direction are detected."""
        workflow = {
            "nodes": [
                {
                    "id": 20,
                    "type": "Label",
                    "properties": {
                        "labelName": "SomeLabel",
                        "sourceNodeId": 10,
                        "sourceSlotIndex": 0
                        # Missing labelDirection
                    }
                }
            ],
            "links": []
        }
        
        with self.assertRaises(ValueError) as context:
            self.exporter.generate_label_connections(workflow)
        
        error_msg = str(context.exception)
        self.assertIn("missing name or direction", error_msg)
        self.assertIn("node 20", error_msg)
    
    def test_validate_multiple_errors(self):
        """Test that multiple validation errors are reported together."""
        workflow = {
            "nodes": [
                # Orphaned output label
                {
                    "id": 20,
                    "type": "Label",
                    "properties": {
                        "labelName": "Orphaned1",
                        "labelDirection": "output"
                    }
                },
                # Orphaned input label
                {
                    "id": 30,
                    "type": "Label",
                    "properties": {
                        "labelName": "Orphaned2",
                        "labelDirection": "input"
                    }
                },
                # Label without properties
                {
                    "id": 40,
                    "type": "Label",
                    "properties": {}
                }
            ],
            "links": []
        }
        
        with self.assertRaises(ValueError) as context:
            self.exporter.generate_label_connections(workflow)
        
        error_msg = str(context.exception)
        # Should report all orphaned labels
        self.assertIn("Found orphaned or invalid labels", error_msg)
        self.assertIn("node 20", error_msg)
        self.assertIn("node 30", error_msg)
        self.assertIn("node 40", error_msg)
    
    def test_valid_labels_no_errors(self):
        """Test that valid label configurations don't raise errors."""
        workflow = {
            "nodes": [
                {
                    "id": 20,
                    "type": "Label",
                    "properties": {
                        "labelName": "Valid.output",
                        "labelDirection": "output",
                        "sourceNodeId": 10,
                        "sourceSlotIndex": 0,
                        "sourceSlotName": "output"
                    }
                },
                {
                    "id": 30,
                    "type": "Label",
                    "properties": {
                        "labelName": "Valid.output",
                        "labelDirection": "input",
                        "targetNodeId": 40,
                        "targetSlotIndex": 0,
                        "targetSlotName": "input",
                        "connectedToLabel": "Valid.output"
                    }
                }
            ],
            "links": []
        }
        
        # Should not raise any errors
        try:
            connections, _ = self.exporter.generate_label_connections(workflow)
            self.assertEqual(len(connections), 1)
        except ValueError as e:
            self.fail(f"Valid labels should not raise error: {e}")


if __name__ == '__main__':
    unittest.main()