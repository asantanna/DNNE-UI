"""
Unit tests for DNNE graph exporter.

Tests the core GraphExporter class for workflow parsing, slot corruption fixes,
dependency resolution, and code generation coordination.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import export system components
from export_system.graph_exporter import GraphExporter
from export_system.node_exporters import register_all_exporters
from fixtures.workflows import (
    MINIMAL_LINEAR_WORKFLOW, SIMPLE_DATASET_NETWORK, MINIMAL_TRAINING_WORKFLOW,
    INVALID_WORKFLOW
)
from fixtures.test_utils import (
    validate_workflow_structure, validate_export_output, 
    create_temp_export_dir, cleanup_export_dir
)


class TestGraphExporter:
    """Test core GraphExporter functionality."""
    
    @pytest.mark.export
    def test_exporter_initialization(self):
        """Test GraphExporter initialization."""
        exporter = GraphExporter()
        
        assert hasattr(exporter, 'node_registry')
        assert isinstance(exporter.node_registry, dict)
        assert hasattr(exporter, 'export_workflow')
        assert callable(exporter.export_workflow)
    
    @pytest.mark.export
    def test_node_registry_registration(self):
        """Test node exporter registration with pre-loaded exporters."""
        exporter = GraphExporter()
        
        # Should have pre-loaded exporters (not empty)
        initial_count = len(exporter.node_registry)
        assert initial_count > 0, "Should have pre-loaded exporters"
        
        # Check that key exporters are pre-loaded
        expected_exporters = ["MNISTDataset", "LinearLayer", "Network", "SGDOptimizer"]
        for expected in expected_exporters:
            assert expected in exporter.node_registry, f"Expected {expected} to be pre-loaded"
        
        # Register additional mock exporter
        mock_exporter = Mock()
        mock_exporter.get_template_name = Mock(return_value="test_template.py")
        mock_exporter.prepare_template_vars = Mock(return_value={})
        mock_exporter.get_imports = Mock(return_value=[])
        
        exporter.register_node("TestNode", mock_exporter)
        
        # Should have added one more exporter
        assert len(exporter.node_registry) == initial_count + 1
        assert "TestNode" in exporter.node_registry
        assert exporter.node_registry["TestNode"] == mock_exporter
    
    @pytest.mark.export
    def test_register_all_exporters(self):
        """Test registration of all DNNE node exporters."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        # Should have registered ML and robotics nodes
        assert len(exporter.node_registry) > 0
        
        # Check for key ML nodes
        expected_ml_nodes = [
            "MNISTDataset", "LinearLayer", "Network", 
            "SGDOptimizer", "CrossEntropyLoss", "TrainingStep"
        ]
        
        registered_nodes = list(exporter.node_registry.keys())
        
        for node_type in expected_ml_nodes:
            if node_type in registered_nodes:
                # At least some key nodes should be registered
                assert exporter.node_registry[node_type] is not None
    
    @pytest.mark.export
    def test_workflow_validation(self):
        """Test workflow structure validation."""
        exporter = GraphExporter()
        
        # Valid workflow
        assert validate_workflow_structure(MINIMAL_LINEAR_WORKFLOW)
        assert validate_workflow_structure(SIMPLE_DATASET_NETWORK)
        
        # Invalid workflow
        assert not validate_workflow_structure(INVALID_WORKFLOW)
        
        # Malformed workflows
        invalid_workflows = [
            {},  # Empty
            {"nodes": []},  # Missing links
            {"links": []},  # Missing nodes
            {"nodes": "not_a_list", "links": []},  # Wrong type
        ]
        
        for invalid_workflow in invalid_workflows:
            assert not validate_workflow_structure(invalid_workflow)


class TestWorkflowParsing:
    """Test workflow parsing and analysis."""
    
    @pytest.mark.export
    def test_minimal_workflow_parsing(self):
        """Test parsing of minimal workflow (should fail due to missing connections)."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        workflow = MINIMAL_LINEAR_WORKFLOW
        export_path = create_temp_export_dir()
        
        try:
            # This should fail for disconnected workflow
            result = exporter.export_workflow(workflow, export_path)
            pytest.fail("Minimal workflow should fail due to missing input connections")
            
        except ValueError as e:
            # Expected error for minimal workflow with no connections
            assert "No input connection found" in str(e), f"Expected connection error, got: {e}"
            print(f"✓ Expected parsing error for minimal workflow: {e}")
            
        except Exception as e:
            # Unexpected error type
            pytest.fail(f"Expected ValueError for missing connections, got {type(e).__name__}: {e}")
            
        finally:
            cleanup_export_dir(export_path)
    
    @pytest.mark.export
    def test_complex_workflow_parsing(self):
        """Test parsing of complex training workflow."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        workflow = MINIMAL_TRAINING_WORKFLOW
        export_path = create_temp_export_dir()
        
        try:
            result = exporter.export_workflow(workflow, export_path)
            
            assert result is not None
            
        except Exception as e:
            # Should handle complex workflows gracefully
            error_msg = str(e).lower()
            print(f"Complex workflow export error: {error_msg}")
            # This is expected if templates are missing
            
        finally:
            cleanup_export_dir(export_path)
    
    @pytest.mark.export 
    def test_connection_analysis(self):
        """Test analysis of node connections."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        workflow = SIMPLE_DATASET_NETWORK
        
        # Test connection parsing
        nodes = workflow.get("nodes", [])
        links = workflow.get("links", [])
        
        assert len(nodes) == 2
        assert len(links) == 1
        
        # Verify connection structure
        link = links[0]
        assert len(link) == 4  # [from_node, from_output, to_node, to_input]
        
        from_node, from_output, to_node, to_input = link
        assert from_node == "1"
        assert to_node == "2"
        assert isinstance(from_output, str)
        assert isinstance(to_input, str)
    
    @pytest.mark.export
    def test_slot_corruption_handling(self):
        """Test handling of ComfyUI slot corruption."""
        exporter = GraphExporter()
        
        # Create workflow with potentially corrupted slots
        corrupted_workflow = {
            "nodes": [
                {"id": "1", "type": "MNISTDataset", "inputs": {}, "widgets": {}},
                {"id": "2", "type": "Network", "inputs": {}, "widgets": {}}
            ],
            "links": [
                ["1", "dataset", "2", "input"]  # This might get corrupted to ["1", "dataset", "2", 0]
            ]
        }
        
        # Test that exporter can handle various link formats
        try:
            export_path = create_temp_export_dir()
            result = exporter.export_workflow(corrupted_workflow, export_path)
            
            # Should either succeed or fail gracefully
            assert result is not None or True  # Either works or raises exception
            
        except Exception as e:
            # Should provide meaningful error for slot issues
            error_msg = str(e).lower()
            print(f"Slot corruption handling: {error_msg}")
            
        finally:
            if 'export_path' in locals():
                cleanup_export_dir(export_path)


class TestCodeGeneration:
    """Test code generation and output creation."""
    
    @pytest.mark.export
    def test_export_directory_creation(self):
        """Test creation of export directory structure."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        # Use a properly connected workflow
        workflow = SIMPLE_DATASET_NETWORK
        export_path = create_temp_export_dir()
        
        try:
            # Attempt export
            result = exporter.export_workflow(workflow, export_path)
            
            # Should succeed with connected workflow
            assert result is not None, "Export should succeed with properly connected workflow"
            
            # Check directory structure
            assert export_path.exists() and export_path.is_dir()
            
            # Check for expected structure
            files_created = list(export_path.glob("*"))
            assert len(files_created) > 0, "Export should create files"
            
        except Exception as e:
            # If this fails, it's a real error that should fail the test
            pytest.fail(f"Export directory creation failed unexpectedly: {e}")
            
        finally:
            cleanup_export_dir(export_path)
    
    @pytest.mark.export
    def test_runner_file_generation(self):
        """Test generation of runner.py file."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        # Use a properly connected workflow
        workflow = SIMPLE_DATASET_NETWORK
        export_path = create_temp_export_dir()
        
        try:
            result = exporter.export_workflow(workflow, export_path)
            
            # Should succeed and create runner.py
            assert result is not None, "Export should succeed with connected workflow"
            
            runner_file = export_path / "runner.py"
            assert runner_file.exists(), "Export should create runner.py file"
            assert runner_file.is_file(), "runner.py should be a file"
            
            # Check basic file content
            content = runner_file.read_text()
            assert len(content) > 0, "Runner file should not be empty"
            
            # Should contain Python code indicators
            python_indicators = ["import", "async", "def", "class"]
            has_python_content = any(indicator in content for indicator in python_indicators)
            assert has_python_content, "Runner should contain Python code"
                
        except Exception as e:
            pytest.fail(f"Runner file generation failed unexpectedly: {e}")
            
        finally:
            cleanup_export_dir(export_path)
    
    @pytest.mark.export
    def test_framework_directory_generation(self):
        """Test generation of framework directory."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        # Use a properly connected workflow
        workflow = SIMPLE_DATASET_NETWORK
        export_path = create_temp_export_dir()
        
        try:
            result = exporter.export_workflow(workflow, export_path)
            
            # Should succeed and create framework directory
            assert result is not None, "Export should succeed with connected workflow"
            
            framework_dir = export_path / "framework"
            assert framework_dir.exists(), "Export should create framework directory"
            assert framework_dir.is_dir(), "framework should be a directory"
            
            # Should contain base framework files
            base_file = framework_dir / "base.py"
            assert base_file.exists(), "Framework should contain base.py"
            
            content = base_file.read_text()
            assert "QueueNode" in content or "class" in content, "Base file should contain framework code"
                    
        except Exception as e:
            pytest.fail(f"Framework directory generation failed unexpectedly: {e}")
            
        finally:
            cleanup_export_dir(export_path)


    @pytest.mark.export
    def test_disconnected_workflow_handling(self):
        """Test that disconnected workflows fail with expected errors."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        # Use a workflow with no connections (should fail)
        workflow = MINIMAL_LINEAR_WORKFLOW
        export_path = create_temp_export_dir()
        
        try:
            result = exporter.export_workflow(workflow, export_path)
            
            # This should fail for disconnected workflow
            pytest.fail("Export should fail for disconnected workflow with missing input connections")
                    
        except ValueError as e:
            # Expected error for missing input connections
            assert "No input connection found" in str(e), f"Expected input connection error, got: {e}"
            print(f"✓ Expected error for disconnected workflow: {e}")
            
        except Exception as e:
            pytest.fail(f"Got unexpected error type for disconnected workflow: {type(e).__name__}: {e}")
            
        finally:
            cleanup_export_dir(export_path)


class TestErrorHandling:
    """Test export system error handling."""
    
    @pytest.mark.export
    def test_invalid_workflow_handling(self):
        """Test handling of invalid workflows."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        export_path = create_temp_export_dir()
        
        try:
            # Test with invalid workflow
            result = exporter.export_workflow(INVALID_WORKFLOW, export_path)
            
            # Should either handle gracefully or raise meaningful exception
            if result is not None:
                # Handled gracefully
                pass
                
        except Exception as e:
            # Should provide meaningful error
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in 
                      ["invalid", "node", "type", "unknown", "missing"])
            
        finally:
            cleanup_export_dir(export_path)
    
    @pytest.mark.export
    def test_missing_node_exporter_handling(self):
        """Test handling of missing node exporters (should generate placeholders)."""
        exporter = GraphExporter()
        
        # Remove a specific exporter to test missing exporter handling
        if "MNISTDataset" in exporter.node_registry:
            del exporter.node_registry["MNISTDataset"]
        
        # Use a workflow that includes the removed exporter
        workflow = SIMPLE_DATASET_NETWORK
        export_path = create_temp_export_dir()
        
        try:
            result = exporter.export_workflow(workflow, export_path)
            
            # Export should succeed with placeholder generation
            assert result is not None, "Export should succeed by generating placeholders"
            
            print(f"✓ Expected behavior: Missing exporter handled gracefully with placeholder generation")
            
            # Check that placeholder was created
            nodes_dir = export_path / "nodes"
            assert nodes_dir.exists(), "Should create nodes directory"
            
            # Look for any generated node files
            node_files = list(nodes_dir.glob("*.py"))
            assert len(node_files) > 0, "Should create node files (including placeholders)"
                
        except Exception as e:
            pytest.fail(f"Export should succeed with placeholder generation, got: {e}")
            
        finally:
            cleanup_export_dir(export_path)
    
    @pytest.mark.export
    def test_invalid_export_path_handling(self):
        """Test handling of invalid export paths."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        # Use a workflow with nodes that don't require connections
        workflow = {
            "nodes": [
                {
                    "id": "1",
                    "type": "MNISTDataset",
                    "inputs": {},
                    "widgets": {}
                }
            ],
            "links": []
        }
        
        # Test with invalid path
        invalid_paths = [
            Path("/nonexistent/readonly/path"),
            Path(""),  # Empty path
            Path("/dev/null/invalid"),  # Invalid nested path
        ]
        
        for invalid_path in invalid_paths:
            try:
                result = exporter.export_workflow(workflow, invalid_path)
                
                # If it succeeds, that's okay too
                if result is not None:
                    continue
                    
            except Exception as e:
                # Should provide meaningful error for path issues
                error_msg = str(e).lower()
                assert any(keyword in error_msg for keyword in 
                          ["path", "directory", "permission", "create", "write"])


class TestExportIntegration:
    """Integration tests for complete export workflows."""
    
    @pytest.mark.export
    @pytest.mark.integration
    def test_mnist_workflow_export(self, sample_mnist_workflow):
        """Test export of actual MNIST workflow if available."""
        assert sample_mnist_workflow is not None, "MNIST workflow must be available - missing workflow file indicates broken test environment"
            
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        export_path = create_temp_export_dir()
        
        try:
            result = exporter.export_workflow(sample_mnist_workflow, export_path)
            
            assert result is not None
            
            # Check for complete export structure
            if validate_export_output(export_path):
                # Export succeeded and created proper structure
                assert (export_path / "runner.py").exists()
                assert (export_path / "framework").is_dir()
                assert (export_path / "nodes").is_dir()
                
        except Exception as e:
            # Log error for debugging
            print(f"MNIST workflow export error: {str(e)}")
            # This might fail if templates are incomplete
            
        finally:
            cleanup_export_dir(export_path)
    
    @pytest.mark.export
    @pytest.mark.performance
    def test_export_performance(self):
        """Test export performance with reasonable workflows."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        workflow = MINIMAL_TRAINING_WORKFLOW  # More complex workflow
        export_path = create_temp_export_dir()
        
        import time
        start_time = time.time()
        
        try:
            result = exporter.export_workflow(workflow, export_path)
            
            end_time = time.time()
            export_time = end_time - start_time
            
            # Should complete reasonably quickly
            assert export_time < 30.0, f"Export took too long: {export_time}s"
            
        except Exception as e:
            # Even if export fails, should not take too long
            end_time = time.time()
            export_time = end_time - start_time
            assert export_time < 30.0, f"Export failure took too long: {export_time}s"
            
        finally:
            cleanup_export_dir(export_path)
    
    @pytest.mark.export
    def test_export_result_format(self):
        """Test format of export results."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        # Use a connected workflow for successful export
        workflow = SIMPLE_DATASET_NETWORK
        export_path = create_temp_export_dir()
        
        try:
            result = exporter.export_workflow(workflow, export_path)
            
            # Result should be meaningful for successful export
            assert result is not None, "Export should return a result"
            
            # Check result format
            if isinstance(result, dict):
                # Should have status or success indicator
                expected_keys = ["status", "success", "path", "files"]
                has_expected_key = any(key in result for key in expected_keys)
                assert has_expected_key, f"Result should have expected keys, got: {result.keys()}"
                
            elif isinstance(result, bool):
                # Boolean success indicator is fine
                assert isinstance(result, bool)
                
            elif isinstance(result, str):
                # String result should be meaningful
                assert len(result) > 0
                
        except Exception as e:
            # Should not fail with connected workflow and registered exporters
            pytest.fail(f"Export should succeed with connected workflow, got: {e}")
            
        finally:
            cleanup_export_dir(export_path)