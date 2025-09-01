#!/usr/bin/env python3
"""
Unit tests for Split node functionality, especially the new index range features.

Tests cover:
1. Legacy integer split points
2. New range notation (inclusive)
3. Mixed range and single index notation
4. Error handling for invalid inputs
5. Export and template generation
"""

import unittest
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from export_system.node_exporters.split_exporter import SplitExporter
from export_system.graph_exporter import GraphExporter


class TestSplitNodeRangeParsing(unittest.TestCase):
    """Test the parse_index_ranges method for various input formats"""
    
    def test_single_index(self):
        """Test parsing single indices"""
        result = SplitExporter.parse_index_ranges("3", "test_node")
        self.assertEqual(result, [(3, 4)])
        
        result = SplitExporter.parse_index_ranges("0", "test_node")
        self.assertEqual(result, [(0, 1)])
    
    def test_multiple_single_indices(self):
        """Test parsing multiple single indices"""
        result = SplitExporter.parse_index_ranges("3, 5, 7", "test_node")
        self.assertEqual(result, [(3, 4), (5, 6), (7, 8)])
    
    def test_simple_range(self):
        """Test parsing simple range notation (inclusive)"""
        result = SplitExporter.parse_index_ranges("3:5", "test_node")
        self.assertEqual(result, [(3, 6)])  # 3:5 inclusive becomes [3, 6) exclusive
        
        result = SplitExporter.parse_index_ranges("0:2", "test_node")
        self.assertEqual(result, [(0, 3)])
    
    def test_bracketed_range(self):
        """Test parsing bracketed range notation"""
        result = SplitExporter.parse_index_ranges("[3:5]", "test_node")
        self.assertEqual(result, [(3, 6)])
        
        result = SplitExporter.parse_index_ranges("[10:18]", "test_node")
        self.assertEqual(result, [(10, 19)])
    
    def test_multiple_ranges(self):
        """Test parsing multiple ranges"""
        result = SplitExporter.parse_index_ranges("[3:5], [10:18]", "test_node")
        self.assertEqual(result, [(3, 6), (10, 19)])
        
        result = SplitExporter.parse_index_ranges("0:2, 5:7, 10:12", "test_node")
        self.assertEqual(result, [(0, 3), (5, 8), (10, 13)])
    
    def test_mixed_ranges_and_singles(self):
        """Test parsing mixed ranges and single indices"""
        result = SplitExporter.parse_index_ranges("3, [5:7], 10", "test_node")
        self.assertEqual(result, [(3, 4), (5, 8), (10, 11)])
        
        result = SplitExporter.parse_index_ranges("[0:2], 5, [10:15], 20", "test_node")
        self.assertEqual(result, [(0, 3), (5, 6), (10, 16), (20, 21)])
    
    def test_single_element_range(self):
        """Test range with same start and end (single element)"""
        result = SplitExporter.parse_index_ranges("[5:5]", "test_node")
        self.assertEqual(result, [(5, 6)])  # 5:5 inclusive is just element 5
    
    def test_whitespace_handling(self):
        """Test that whitespace is properly handled"""
        result = SplitExporter.parse_index_ranges(" [ 3 : 5 ] , [ 10 : 18 ] ", "test_node")
        self.assertEqual(result, [(3, 6), (10, 19)])
    
    def test_empty_input(self):
        """Test that empty input raises appropriate error"""
        with self.assertRaises(ValueError) as ctx:
            SplitExporter.parse_index_ranges("", "test_node")
        self.assertIn("resulted in empty list", str(ctx.exception))
    
    def test_invalid_range_format(self):
        """Test that invalid range formats raise errors"""
        # Too many colons
        with self.assertRaises(ValueError) as ctx:
            SplitExporter.parse_index_ranges("3:5:7", "test_node")
        self.assertIn("Invalid range format", str(ctx.exception))
        
        # Non-numeric values
        with self.assertRaises(ValueError) as ctx:
            SplitExporter.parse_index_ranges("abc", "test_node")
        self.assertIn("Invalid index", str(ctx.exception))
        
        # Non-numeric in range
        with self.assertRaises(ValueError) as ctx:
            SplitExporter.parse_index_ranges("[3:abc]", "test_node")
        self.assertIn("Invalid range values", str(ctx.exception))


class TestSplitNodeTemplateVars(unittest.TestCase):
    """Test template variable preparation for different split modes"""
    
    def test_by_index_with_ranges(self):
        """Test template vars for 'by index' mode with ranges"""
        node_data = {
            'id': 'split_1',
            'widgets_values': [
                1,  # dimension
                "by index",  # split_mode
                "[3:5], [10:18]"  # split_pos with ranges
            ]
        }
        
        vars = SplitExporter.prepare_template_vars(
            "split_1", node_data, {}
        )
        
        self.assertEqual(vars['NODE_ID'], "split_1")
        self.assertEqual(vars['SPLIT_MODE'], "by index")
        self.assertEqual(vars['SPLIT_VALUES'], [[3, 6], [10, 19]])
        self.assertEqual(vars['DIMENSION'], 1)  # Always 1 per tensor standards
    
    def test_by_index_with_singles(self):
        """Test template vars for 'by index' mode with single indices"""
        node_data = {
            'id': 'split_2',
            'widgets_values': [
                0,  # dimension (will be overridden to 1)
                "by index",
                "3, 5, 7"
            ]
        }
        
        vars = SplitExporter.prepare_template_vars(
            "split_2", node_data, {}
        )
        
        self.assertEqual(vars['SPLIT_VALUES'], [[3, 4], [5, 6], [7, 8]])
    
    def test_by_index_mixed(self):
        """Test template vars for 'by index' mode with mixed notation"""
        node_data = {
            'id': 'split_3',
            'widgets_values': [
                1,
                "by index",
                "0, [3:5], 10, [15:20]"
            ]
        }
        
        vars = SplitExporter.prepare_template_vars(
            "split_3", node_data, {}
        )
        
        self.assertEqual(vars['SPLIT_VALUES'], [[0, 1], [3, 6], [10, 11], [15, 21]])
    
    def test_by_size_mode(self):
        """Test that 'by size' mode still works with integer lists"""
        node_data = {
            'id': 'split_4',
            'widgets_values': [
                1,
                "by size",
                "10, 10, 10, 10"
            ]
        }
        
        vars = SplitExporter.prepare_template_vars(
            "split_4", node_data, {}
        )
        
        self.assertEqual(vars['SPLIT_MODE'], "by size")
        self.assertEqual(vars['SPLIT_VALUES'], [10, 10, 10, 10])
    
    def test_dimension_override(self):
        """Test that dimension is always set to 1 per tensor standards"""
        node_data = {
            'id': 'split_5',
            'widgets_values': [
                0,  # Try to set dimension to 0
                "by index",
                "5"
            ]
        }
        
        vars = SplitExporter.prepare_template_vars(
            "split_5", node_data, {}
        )
        
        # Dimension should be forced to 1 regardless of input
        self.assertEqual(vars['DIMENSION'], 1)


class TestSplitNodeExport(unittest.TestCase):
    """Test full export functionality with the Split node"""
    
    def setUp(self):
        """Create a temporary directory for test exports"""
        self.test_dir = tempfile.mkdtemp()
        self.export_dir = Path(self.test_dir) / "exports"
        self.export_dir.mkdir()
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.test_dir)
    
    def test_export_split_with_ranges(self):
        """Test exporting a workflow with Split node using ranges"""
        workflow = {
            "nodes": [
                {
                    "id": 1,
                    "type": "Tensor",
                    "widgets_values": ["20", "ones", 0.0, "float32", -1]
                },
                {
                    "id": 2,
                    "type": "Split",
                    "widgets_values": [1, "by index", "[3:5], [10:18]"]
                }
            ],
            "links": [
                [1, 1, 0, 2, 0, "TENSOR"]  # Tensor.output -> Split.input
            ]
        }
        
        # Test that we can prepare template vars without error
        # (we don't need the full GraphExporter for this test)
        try:
            # Process nodes
            for node in workflow['nodes']:
                if node['type'] == 'Split':
                    # Verify we can prepare template vars without error
                    vars = SplitExporter.prepare_template_vars(
                        str(node['id']), node, {}
                    )
                    self.assertEqual(vars['SPLIT_VALUES'], [[3, 6], [10, 19]])
            
            success = True
        except Exception as e:
            success = False
            print(f"Export failed: {e}")
        
        self.assertTrue(success, "Export should succeed with range notation")


class TestSplitNodeBackwardCompatibility(unittest.TestCase):
    """Test that old workflows still work with the new implementation"""
    
    def test_legacy_integer_splits(self):
        """Test that old-style integer split points are handled correctly"""
        # Old style: "10,20,30" meant split at those positions
        # New interpretation: these become single-element extractions
        result = SplitExporter.parse_index_ranges("10,20,30", "test_node")
        
        # Each integer becomes a single-element range
        self.assertEqual(result, [(10, 11), (20, 21), (30, 31)])
        
        # This is different from the old behavior but still valid
        # Users wanting the old split-at-position behavior would now use
        # the 'by size' mode or adjust their indices
    
    def test_single_integer(self):
        """Test single integer still works"""
        result = SplitExporter.parse_index_ranges("10", "test_node")
        self.assertEqual(result, [(10, 11)])


class TestSplitNodeEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def test_zero_index(self):
        """Test that index 0 works correctly"""
        result = SplitExporter.parse_index_ranges("0", "test_node")
        self.assertEqual(result, [(0, 1)])
        
        result = SplitExporter.parse_index_ranges("[0:0]", "test_node")
        self.assertEqual(result, [(0, 1)])
    
    def test_large_indices(self):
        """Test large index values"""
        result = SplitExporter.parse_index_ranges("[100:200]", "test_node")
        self.assertEqual(result, [(100, 201)])
        
        result = SplitExporter.parse_index_ranges("1000, [2000:2010]", "test_node")
        self.assertEqual(result, [(1000, 1001), (2000, 2011)])
    
    def test_adjacent_ranges(self):
        """Test adjacent ranges"""
        result = SplitExporter.parse_index_ranges("[0:2], [3:5], [6:8]", "test_node")
        self.assertEqual(result, [(0, 3), (3, 6), (6, 9)])
    
    def test_overlapping_ranges(self):
        """Test overlapping ranges (allowed, may extract same indices twice)"""
        result = SplitExporter.parse_index_ranges("[0:5], [3:8]", "test_node")
        self.assertEqual(result, [(0, 6), (3, 9)])
        # Note: overlapping is allowed, outputs will have overlapping data
    
    def test_max_four_outputs(self):
        """Test that we can specify up to 4 ranges (node has 4 outputs)"""
        result = SplitExporter.parse_index_ranges(
            "[0:2], [3:5], [6:8], [9:11]", "test_node"
        )
        self.assertEqual(result, [(0, 3), (3, 6), (6, 9), (9, 12)])
        
        # Fifth range would be parsed but ignored by the template
        result = SplitExporter.parse_index_ranges(
            "[0:2], [3:5], [6:8], [9:11], [12:14]", "test_node"
        )
        self.assertEqual(len(result), 5)  # Parser returns all, template handles limit


def run_tests():
    """Run all Split node tests and report results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSplitNodeRangeParsing))
    suite.addTests(loader.loadTestsFromTestCase(TestSplitNodeTemplateVars))
    suite.addTests(loader.loadTestsFromTestCase(TestSplitNodeExport))
    suite.addTests(loader.loadTestsFromTestCase(TestSplitNodeBackwardCompatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestSplitNodeEdgeCases))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return 0 for success, 1 for failure
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())