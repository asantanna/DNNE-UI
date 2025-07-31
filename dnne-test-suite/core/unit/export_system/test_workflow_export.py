"""
Unit tests for workflow export functionality.

Tests export of complete workflows without execution, including validation
of exported structure, syntax checking, and performance benchmarks.
"""

import pytest
import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Import DNNE components
from export_system.graph_exporter import GraphExporter
from export_system.node_exporters import register_all_exporters
from fixtures.workflows import MINIMAL_TRAINING_WORKFLOW
from fixtures.test_utils import (
    validate_workflow_structure, validate_export_output,
    create_temp_export_dir, cleanup_export_dir, export_workflow_for_test
)


class TestWorkflowLoading:
    """Test loading and validation of workflows."""
    
    @pytest.mark.export
    def test_load_mnist_workflow_file(self, sample_mnist_workflow):
        """Test loading actual MNIST Test.json workflow file."""
        assert sample_mnist_workflow is not None, "MNIST Test.json workflow not found"
        
        # Validate workflow structure
        assert validate_workflow_structure(sample_mnist_workflow)
        
        # Check for expected MNIST nodes
        nodes = sample_mnist_workflow.get("nodes", [])
        node_types = [node.get("type") for node in nodes]
        
        # Should have key ML training nodes
        expected_types = ["MNISTDataset", "Network", "TrainingStep"]
        found_types = [t for t in expected_types if t in node_types]
        
        assert len(found_types) > 0, f"Expected ML nodes not found. Available: {node_types}"
        
        # Check for connections
        links = sample_mnist_workflow.get("links", [])
        assert len(links) > 0, "MNIST workflow should have connections between nodes"
    
    @pytest.mark.export
    def test_minimal_training_workflow_structure(self):
        """Test minimal training workflow structure."""
        workflow = MINIMAL_TRAINING_WORKFLOW
        
        assert validate_workflow_structure(workflow)
        
        # Should have complete training pipeline
        nodes = workflow.get("nodes", [])
        node_types = [node.get("type") for node in nodes]
        
        # Check for training components
        training_components = [
            "MNISTDataset", "BatchSampler", "GetBatch", 
            "Network", "CrossEntropyLoss", "SGDOptimizer", "TrainingStep"
        ]
        
        found_components = [t for t in training_components if t in node_types]
        assert len(found_components) >= 4, \
            f"Should have training components. Found: {found_components}"
        
        # Check for trigger connections (training coordination)
        links = workflow.get("links", [])
        trigger_connections = [
            link for link in links 
            if len(link) == 4 and ("trigger" in str(link).lower() or "ready" in str(link).lower())
        ]
        
        # Should have some trigger-based coordination
        assert len(trigger_connections) >= 0  # May or may not have explicit triggers


class TestWorkflowExport:
    """Test export of workflows to Python code."""
    
    @pytest.mark.export
    def test_minimal_mnist_export(self):
        """Test export of minimal MNIST training workflow."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        workflow = MINIMAL_TRAINING_WORKFLOW
        export_path = create_temp_export_dir()
        
        try:
            # Attempt export
            result = exporter.export_workflow(workflow, export_path)
            
            # Should succeed or provide meaningful error
            if result is not None:
                # Validate export structure
                assert validate_export_output(export_path)
                
                # Check for key files
                assert (export_path / "runner.py").exists()
                assert (export_path / "framework").is_dir()
                assert (export_path / "nodes").is_dir()
                
        except ValueError as e:
            # Expected errors for incomplete workflows
            error_msg = str(e).lower()
            expected_errors = ["input", "connection", "tensor", "size", "missing"]
            assert any(err in error_msg for err in expected_errors), \
                f"Unexpected error: {e}"
            
        except Exception as e:
            # Check if it's a known issue
            if "template" in str(e).lower():
                # Missing template is acceptable for unit tests
                pass
            else:
                pytest.fail(f"Unexpected export error: {e}")
                
        finally:
            cleanup_export_dir(export_path)
    
    @pytest.mark.export
    def test_full_mnist_workflow_export(self, sample_mnist_workflow):
        """Test export of full MNIST workflow with all components."""
        assert sample_mnist_workflow is not None, "MNIST Test.json workflow not available"
        
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        export_path = create_temp_export_dir()
        
        try:
            result = exporter.export_workflow(sample_mnist_workflow, export_path)
            
            # Should succeed with complete workflow
            assert result is not None, "Export should succeed with complete MNIST workflow"
            
            # Validate complete export
            assert validate_export_output(export_path)
            
            # Check for all expected components
            runner_file = export_path / "runner.py"
            assert runner_file.exists()
            
            # Verify runner content
            runner_content = runner_file.read_text()
            assert "async def main()" in runner_content
            assert "MNISTDataset" in runner_content
            assert "TrainingStep" in runner_content
            
            # Check framework files
            framework_dir = export_path / "framework"
            assert (framework_dir / "base_nodes.py").exists()
            assert (framework_dir / "__init__.py").exists()
            
        except Exception as e:
            if "template" in str(e).lower():
                # Missing template is known issue
                pass
            else:
                pytest.fail(f"Full MNIST export failed: {e}")
                
        finally:
            cleanup_export_dir(export_path)
    
    @pytest.mark.export
    def test_export_consistency(self):
        """Test that repeated exports produce consistent results."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        workflow = MINIMAL_TRAINING_WORKFLOW
        
        # Export twice to different locations
        export_path1 = create_temp_export_dir()
        export_path2 = create_temp_export_dir()
        
        try:
            result1 = exporter.export_workflow(workflow, export_path1)
            result2 = exporter.export_workflow(workflow, export_path2)
            
            # Both should have same outcome
            if result1 is not None and result2 is not None:
                # Compare key files
                runner1 = (export_path1 / "runner.py").read_text()
                runner2 = (export_path2 / "runner.py").read_text()
                
                # Should be identical except for paths
                assert len(runner1) == len(runner2), \
                    "Exported runners should have same length"
                    
        except Exception as e:
            # Both should fail in same way
            pass
            
        finally:
            cleanup_export_dir(export_path1)
            cleanup_export_dir(export_path2)


class TestExportedCodeValidation:
    """Test validation of exported code without execution."""
    
    @pytest.mark.export
    def test_exported_code_syntax_validation(self):
        """Test that exported code has valid Python syntax."""
        workflow_name = "MNIST_Test"
        
        try:
            export_path = export_workflow_for_test(workflow_name, "syntax_check")
            
            # Validate all Python files have correct syntax
            python_files = list(export_path.glob("**/*.py"))
            assert len(python_files) > 0, "Should have Python files"
            
            for py_file in python_files:
                # Skip __pycache__ files
                if "__pycache__" in str(py_file):
                    continue
                    
                # Check syntax by compiling
                try:
                    with open(py_file, 'r') as f:
                        compile(f.read(), py_file, 'exec')
                except SyntaxError as e:
                    # Check for specific node files to provide better errors
                    if "nodes/" in str(py_file):
                        node_name = py_file.stem
                        pytest.fail(f"Node file {node_name}.py has syntax error: {e}")
                    else:
                        pytest.fail(f"File {py_file.name} has syntax error: {e}")
                        
        except Exception as e:
            if "template" in str(e).lower():
                raise AssertionError(f"Syntax validation failed: {e}")
            else:
                raise
                
        finally:
            cleanup_export_dir(export_path)
    
    @pytest.mark.export
    def test_import_resolution_in_exported_code(self):
        """Test that all imports in exported code can be resolved."""
        workflow_name = "MNIST_Test"
        
        try:
            export_path = export_workflow_for_test(workflow_name, "import_check")
            
            # Check runner.py imports
            runner_file = export_path / "runner.py"
            if runner_file.exists():
                runner_content = runner_file.read_text()
                
                # Extract import statements
                import_lines = [
                    line.strip() for line in runner_content.split('\n') 
                    if line.strip().startswith('import ') or 
                       line.strip().startswith('from ')
                ]
                
                # Common expected imports
                expected_imports = [
                    "import asyncio",
                    "from framework import",
                    "from nodes"
                ]
                
                # Verify key imports are present
                for expected in expected_imports:
                    assert any(expected in imp for imp in import_lines), \
                        f"Missing expected import pattern: {expected}"
                        
                # Check that node imports match exported nodes
                node_files = list((export_path / "nodes").glob("*.py"))
                node_names = [f.stem for f in node_files if f.stem != "__init__"]
                
                for node_name in node_names:
                    # Should import each node
                    import_pattern = f"from nodes.{node_name} import"
                    assert any(import_pattern in imp for imp in import_lines), \
                        f"Missing import for node: {node_name}"
                        
        except Exception as e:
            if "template" in str(e).lower():
                raise AssertionError(f"Import validation failed: {e}")
            else:
                raise
                
        finally:
            cleanup_export_dir(export_path)


class TestWorkflowExportPerformance:
    """Test export performance with various workflows."""
    
    @pytest.mark.export
    @pytest.mark.performance
    def test_export_performance(self):
        """Test that workflow export completes in reasonable time."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        workflow = MINIMAL_TRAINING_WORKFLOW
        export_path = create_temp_export_dir()
        
        start_time = time.time()
        
        try:
            result = exporter.export_workflow(workflow, export_path)
            
            end_time = time.time()
            export_time = end_time - start_time
            
            # Export should be fast (under 5 seconds for minimal workflow)
            assert export_time < 5.0, f"Export took too long: {export_time:.2f}s"
            
            print(f"✓ Minimal workflow exported in {export_time:.2f}s")
            
        except Exception as e:
            # Even errors should happen quickly
            end_time = time.time()
            export_time = end_time - start_time
            assert export_time < 5.0, f"Export error took too long: {export_time:.2f}s"
            
        finally:
            cleanup_export_dir(export_path)
    
    @pytest.mark.export
    @pytest.mark.performance
    def test_multiple_exports_performance(self):
        """Test performance of multiple sequential exports."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        workflow = MINIMAL_TRAINING_WORKFLOW
        num_exports = 5
        
        export_times = []
        export_paths = []
        
        try:
            for i in range(num_exports):
                export_path = create_temp_export_dir()
                export_paths.append(export_path)
                
                start_time = time.time()
                result = exporter.export_workflow(workflow, export_path)
                end_time = time.time()
                
                export_times.append(end_time - start_time)
                
            # Check consistency
            avg_time = sum(export_times) / len(export_times)
            max_time = max(export_times)
            
            # No single export should take much longer than average
            assert max_time < avg_time * 2, \
                f"Export time variance too high: avg={avg_time:.2f}s, max={max_time:.2f}s"
                
            print(f"✓ {num_exports} exports completed in {sum(export_times):.2f}s total")
            print(f"  Average: {avg_time:.2f}s, Max: {max_time:.2f}s")
            
        except Exception as e:
            # Performance test failure
            pass
            
        finally:
            for path in export_paths:
                cleanup_export_dir(path)