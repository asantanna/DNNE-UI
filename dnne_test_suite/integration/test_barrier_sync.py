#!/usr/bin/env python3
"""
Integration test for Barrier node synchronization behavior
"""

import unittest
import sys
import os
import json
import tempfile
import shutil
import asyncio
from collections import deque
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from export_system.graph_exporter import GraphExporter


class TestBarrierSynchronization(unittest.TestCase):
    """Integration tests for Barrier node synchronization patterns"""
    
    def setUp(self):
        """Set up test environment"""
        # Use export_system/exports as the base directory
        self.export_base = Path("export_system/exports")
        self.test_dir = self.export_base / f"test_barrier_{os.getpid()}"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.exporter = GraphExporter()
    
    def tearDown(self):
        """Clean up test environment"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def create_barrier_workflow(self):
        """Create a test workflow with Barrier node"""
        workflow = {
            "metadata": {
                "workflow_name": "barrier_test",
                "skip-slot-correction": True
            },
            "nodes": [
                {
                    "id": 1,
                    "type": "Tensor",
                    "properties": {"Node name for S&R": "Tensor"},
                    "widgets_values": ["2,3", "ones", 0.0, "float32", -1]
                },
                {
                    "id": 2,
                    "type": "Tensor",
                    "properties": {"Node name for S&R": "Tensor"},
                    "widgets_values": ["1", "zeros", 0.0, "float32", -1]
                },
                {
                    "id": 3,
                    "type": "Barrier",
                    "properties": {"Node name for S&R": "Barrier"},
                    "widgets_values": ["FIFO"]
                }
            ],
            "links": [
                # Link data source to barrier input
                [1, 1, 0, 3, 0, "TENSOR"],
                # Link trigger source to barrier release
                [2, 2, 0, 3, 1, "TRIGGER"]
            ],
            "execution_order": {
                "EXECUTE": ["1", "2", "3"]
            }
        }
        return workflow
    
    def test_barrier_export(self):
        """Test that Barrier node exports correctly"""
        workflow = self.create_barrier_workflow()
        
        # Export the workflow
        output_dir = self.test_dir / "barrier_test"
        result = self.exporter.export_workflow(workflow, output_dir)
        
        self.assertTrue(result, "Export should succeed")
        
        # Check that runner.py was created
        runner_path = os.path.join(output_dir, "runner.py")
        self.assertTrue(os.path.exists(runner_path), "runner.py should be created")
        
        # Check that the Barrier node file was created
        barrier_path = os.path.join(output_dir, "nodes", "barriernode_3.py")
        self.assertTrue(os.path.exists(barrier_path), "barriernode_3.py should be created")
        
        # Check that the generated code contains Barrier node
        with open(barrier_path, 'r') as f:
            content = f.read()
        
        # Verify Barrier node class is present
        self.assertIn("class BarrierNode_3(QueueNode):", content)
        self.assertIn("self.fifo_queue = deque()", content)
        self.assertIn("self.release_count = 0", content)
        self.assertIn("async def handle_data_input", content)
        self.assertIn("async def handle_trigger_input", content)
        self.assertIn("async def process_releases", content)
    
    def test_barrier_template_variables(self):
        """Test that template variables are properly substituted"""
        workflow = self.create_barrier_workflow()
        
        # Export the workflow
        output_dir = self.test_dir / "barrier_vars"
        self.exporter.export_workflow(workflow, output_dir)
        
        # Read generated Barrier node code
        barrier_path = os.path.join(output_dir, "nodes", "barriernode_3.py")
        with open(barrier_path, 'r') as f:
            content = f.read()
        
        # Check specific substitutions
        self.assertIn('self.hold_mode = "FIFO"', content)
        self.assertIn("BarrierNode_3", content)
        self.assertNotIn("{NODE_ID}", content)
        self.assertNotIn("{CLASS_NAME}", content)
        self.assertNotIn("{HOLD_MODE}", content)
    
    def test_complex_barrier_workflow(self):
        """Test a more complex workflow with multiple connections"""
        workflow = {
            "metadata": {
                "workflow_name": "complex_barrier",
                "skip-slot-correction": True
            },
            "nodes": [
                {
                    "id": 1,
                    "type": "Tensor",
                    "properties": {"Node name for S&R": "Tensor"},
                    "widgets_values": ["10,5", "normal", 0.0, "float32", 42]
                },
                {
                    "id": 2,
                    "type": "Tensor",
                    "properties": {"Node name for S&R": "Tensor"},
                    "widgets_values": ["1", "ones", 0.0, "float32", -1]
                },
                {
                    "id": 3,
                    "type": "Barrier",
                    "properties": {"Node name for S&R": "Barrier"},
                    "widgets_values": ["FIFO"]
                },
                {
                    "id": 4,
                    "type": "Barrier",
                    "properties": {"Node name for S&R": "Barrier"},
                    "widgets_values": ["FIFO"]
                }
            ],
            "links": [
                # Data to first barrier
                [1, 1, 0, 3, 0, "TENSOR"],
                # Trigger to first barrier
                [2, 2, 0, 3, 1, "TRIGGER"],
                # First barrier output to second barrier input
                [3, 3, 0, 4, 0, "HELD_TENSOR"],
                # Same trigger to second barrier
                [4, 2, 0, 4, 1, "TRIGGER"]
            ],
            "execution_order": {
                "EXECUTE": ["1", "2", "3", "4"]
            }
        }
        
        # Export the workflow
        output_dir = self.test_dir / "complex_barrier"
        result = self.exporter.export_workflow(workflow, output_dir)
        
        self.assertTrue(result, "Complex workflow export should succeed")
        
        # Check generated Barrier node files
        barrier3_path = os.path.join(output_dir, "nodes", "barriernode_3.py")
        barrier4_path = os.path.join(output_dir, "nodes", "barriernode_4.py")
        
        with open(barrier3_path, 'r') as f:
            content3 = f.read()
        with open(barrier4_path, 'r') as f:
            content4 = f.read()
        
        # Verify both barrier nodes are present
        self.assertIn("class BarrierNode_3(QueueNode):", content3)
        self.assertIn("class BarrierNode_4(QueueNode):", content4)
        
        # Check runner.py for connections
        runner_path = os.path.join(output_dir, "runner.py")
        with open(runner_path, 'r') as f:
            content = f.read()
        
        # Verify connections
        self.assertIn('("3", "output", "4", "input")', content)


class TestBarrierQueueBehavior(unittest.TestCase):
    """Test the queue behavior patterns described in the spec"""
    
    def test_fifo_queue_imports(self):
        """Test that FIFO queue implementation uses collections.deque"""
        from export_system.node_exporters.barrier_exporter import BarrierExporter
        
        imports = BarrierExporter.get_imports()
        self.assertIn("from collections import deque", imports)
    
    def test_barrier_state_management(self):
        """Test that Barrier maintains proper state variables"""
        template_path = os.path.join(
            os.path.dirname(__file__),
            '../../export_system/templates/nodes/barrier_node_queue.tpl'
        )
        
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Check for state management
        self.assertIn("self.fifo_queue = deque()", content)
        self.assertIn("self.release_count = 0", content)
        self.assertIn("self.total_held = 0", content)
        self.assertIn("self.total_released = 0", content)
    
    def test_barrier_processing_algorithm(self):
        """Test that the processing algorithm matches the spec"""
        template_path = os.path.join(
            os.path.dirname(__file__),
            '../../export_system/templates/nodes/barrier_node_queue.tpl'
        )
        
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Check for correct processing methods
        self.assertIn("async def handle_data_input(self, data):", content)
        self.assertIn("async def handle_trigger_input(self, trigger):", content)
        self.assertIn("async def process_releases(self):", content)
        
        # Check FIFO behavior
        self.assertIn("self.fifo_queue.append(data)", content)
        self.assertIn("data = self.fifo_queue.popleft()", content)
        
        # Check release counting
        self.assertIn("self.release_count += 1", content)
        self.assertIn("self.release_count -= 1", content)
        
        # Check release loop condition
        self.assertIn("while self.release_count > 0 and self.fifo_queue:", content)


if __name__ == '__main__':
    unittest.main()