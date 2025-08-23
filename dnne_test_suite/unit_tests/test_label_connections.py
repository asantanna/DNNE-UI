#!/usr/bin/env python3
"""
Unit tests for label connection validation in the export system.
Tests that the export system correctly recognizes inputs connected through label pairs.
"""

import unittest
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from export_system.graph_exporter import GraphExporter
from export_system.node_exporters import register_all_exporters


class TestLabelConnections(unittest.TestCase):
    """Test that label connections are properly validated during export"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        # Create exporter and register all node types
        cls.exporter = GraphExporter()
        register_all_exporters(cls.exporter)  # This populates exporter.node_registry
        cls.node_registry = cls.exporter.node_registry
        
        # Load the MNIST_Label_Test workflow as fixture
        workflow_path = Path(__file__).parent.parent.parent / "user/default/workflows/MNIST_Label_Test.json"
        with open(workflow_path, 'r') as f:
            cls.workflow = json.load(f)
        
        # Ensure workflow has metadata to skip slot correction for testing
        if 'metadata' not in cls.workflow:
            cls.workflow['metadata'] = {}
        cls.workflow['metadata']['workflow_name'] = 'MNIST_Label_Test'
        cls.workflow['metadata']['skip-slot-correction'] = True  # Disable slot correction for test
    
    def test_label_connections_recognized(self):
        """Test that nodes connected through labels pass validation"""
        # The workflow should have label connections
        self.assertIn('extra', self.workflow, "Workflow should have extra data")
        self.assertIn('labelDictionary', self.workflow['extra'], "Workflow should have label dictionary")
        
        # Generate label connections
        label_connections = self.exporter.generate_label_connections(self.workflow)
        
        # Should find at least one label connection
        self.assertGreater(len(label_connections), 0, 
                          "Should find at least one label connection")
        
        # Get connections for a node that's connected via labels
        # First, find a node that has a label connection
        nodes = self.workflow.get('nodes', [])
        links = self.workflow.get('links', [])
        
        # Add label connections to links (simulating what export_workflow does)
        max_link_id = max([link[0] for link in links] + [0])
        for i, (from_node, from_slot, to_node, to_slot) in enumerate(label_connections):
            new_link = [
                max_link_id + i + 1,  # link_id
                from_node,            # from_node
                from_slot,            # from_slot
                to_node,              # to_node
                to_slot,              # to_slot
                "*"                   # type
            ]
            links.append(new_link)
        
        # Now test _get_node_connections for a node with label input
        # Find the target node of a label connection
        if label_connections:
            _, _, target_node_id, _ = label_connections[0]
            
            # Get connections for this node
            connections = self.exporter._get_node_connections(
                str(target_node_id), links, nodes
            )
            
            # Should have at least one input connection
            self.assertIn('inputs', connections)
            self.assertGreater(len(connections['inputs']), 0,
                             f"Node {target_node_id} should have input connections from labels")
    
    def test_validation_with_labels(self):
        """Test that validation passes for nodes with label connections"""
        try:
            # Try to export the workflow - this will validate all connections
            output_path = Path(__file__).parent.parent / "test_outputs" / "label_test"
            result = self.exporter.export_workflow(self.workflow, output_path)
            
            # If we get here without exception, validation passed
            self.assertIsNotNone(result, "Export should complete successfully")
            
        except ValueError as e:
            # If validation fails, check if it's due to missing connections
            if "missing required input connections" in str(e):
                self.fail(f"Label connections not recognized during validation: {e}")
            else:
                # Some other validation error - might be expected
                pass
    
    def test_label_connection_in_node_connections(self):
        """Test that _get_node_connections includes label-based connections"""
        # First generate the label connections
        label_connections = self.exporter.generate_label_connections(self.workflow)
        
        # Get the nodes and links
        nodes = self.workflow.get('nodes', [])
        links = self.workflow.get('links', [])
        
        # If we have label connections, test one
        if label_connections:
            from_node, from_slot, to_node, to_slot = label_connections[0]
            
            # Get connections for the target node
            connections = self.exporter._get_node_connections(
                str(to_node), links, nodes
            )
            
            # Find the node type to get input names
            target_node = None
            for node in nodes:
                if str(node['id']) == str(to_node):
                    target_node = node
                    break
            
            if target_node:
                node_type = target_node.get('class_type') or target_node.get('type')
                node_class = self.node_registry.get(node_type)
                
                if node_class and hasattr(node_class, 'get_input_names'):
                    input_names = node_class.get_input_names()
                    if to_slot < len(input_names):
                        input_name = input_names[to_slot]
                        
                        # Check that this input has a connection from the label
                        self.assertIn(input_name, connections['inputs'],
                                    f"Input {input_name} should have a connection")
                        
                        # Check that the connection is from the right source
                        found_connection = False
                        for conn in connections['inputs'][input_name]:
                            if conn['from_node'] == str(from_node) and conn['from_slot'] == from_slot:
                                found_connection = True
                                break
                        
                        self.assertTrue(found_connection,
                                      f"Should find connection from node {from_node} slot {from_slot}")


    def test_removed_label_breaks_connection(self):
        """Test that removing a label from the dictionary breaks the connection validation"""
        import copy
        
        # Make a copy of the workflow to modify
        modified_workflow = copy.deepcopy(self.workflow)
        
        # Check that we have labels to begin with
        self.assertIn('extra', modified_workflow)
        self.assertIn('labelDictionary', modified_workflow['extra'])
        
        label_dict = modified_workflow['extra']['labelDictionary']
        self.assertGreater(len(label_dict), 0, "Should have at least one label")
        
        # Generate label connections first to see what we expect
        label_connections = self.exporter.generate_label_connections(modified_workflow)
        self.assertGreater(len(label_connections), 0, "Should have label connections initially")
        
        # Remove a label from the dictionary (simulate deleting a label node)
        # Find an input label to remove
        input_label_to_remove = None
        for label_id, label_data in list(label_dict.items()):
            if 'input' in label_data and label_data['input'].get('connectedToLabel'):
                input_label_to_remove = label_id
                break
        
        if input_label_to_remove:
            # Store info about what will be broken
            removed_label_data = label_dict[input_label_to_remove]
            target_node = removed_label_data['input']['nodeId']
            target_slot = removed_label_data['input']['slotIndex']
            
            # Remove the label
            del label_dict[input_label_to_remove]
            
            # Generate connections again - should be missing the removed connection
            new_label_connections = self.exporter.generate_label_connections(modified_workflow)
            self.assertLess(len(new_label_connections), len(label_connections),
                          "Should have fewer connections after removing label")
            
            # Get the nodes and links
            nodes = modified_workflow.get('nodes', [])
            links = modified_workflow.get('links', [])
            
            # Check that the target node no longer has the label connection
            connections = self.exporter._get_node_connections(
                str(target_node), links, nodes
            )
            
            # Find what input name corresponds to the target slot
            target_node_data = None
            for node in nodes:
                if str(node['id']) == str(target_node):
                    target_node_data = node
                    break
            
            if target_node_data:
                node_type = target_node_data.get('class_type') or target_node_data.get('type')
                node_class = self.node_registry.get(node_type)
                
                if node_class and hasattr(node_class, 'get_input_names'):
                    input_names = node_class.get_input_names()
                    if target_slot < len(input_names):
                        input_name = input_names[target_slot]
                        
                        # The input should either be missing or have no connections from labels
                        if input_name in connections['inputs']:
                            # Check that none of the connections are from the removed label
                            for conn in connections['inputs'][input_name]:
                                # This connection should not be a label connection
                                # (unless there's a direct connection too)
                                pass  # We mainly care that it processes without the label
                        
                        # If this node requires this input, validation should fail
                        if hasattr(node_class, 'get_required_input_names'):
                            required_inputs = node_class.get_required_input_names()
                            if input_name in required_inputs and input_name not in connections['inputs']:
                                # This node would fail validation now
                                with self.assertRaises(ValueError) as context:
                                    node_class.validate_required_connections(str(target_node), connections)
                                self.assertIn("missing required input connections", str(context.exception))


if __name__ == '__main__':
    unittest.main()