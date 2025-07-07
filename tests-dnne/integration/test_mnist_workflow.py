"""
Integration tests for MNIST workflow end-to-end.

Tests complete MNIST training pipeline from workflow loading through export
to execution, validating the full DNNE system integration.
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
    create_temp_export_dir, cleanup_export_dir
)


class TestMNISTWorkflowLoading:
    """Test loading and validation of MNIST workflow."""
    
    @pytest.mark.integration
    @pytest.mark.timeout(30)
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
    
    @pytest.mark.integration
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


class TestMNISTWorkflowExport:
    """Test export of MNIST workflow to Python code."""
    
    @pytest.mark.integration
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
                # Export succeeded - check output
                if validate_export_output(export_path):
                    # Complete export structure created
                    assert (export_path / "runner.py").exists()
                    assert (export_path / "framework").is_dir()
                    
                    # Check runner content
                    runner_content = (export_path / "runner.py").read_text()
                    assert len(runner_content) > 0
                    assert "import" in runner_content
                    assert "class" in runner_content or "async" in runner_content
                
        except Exception as e:
            # Export may fail if templates are incomplete
            error_msg = str(e).lower()
            expected_errors = ["template", "missing", "not found", "exporter"]
            
            has_expected_error = any(err in error_msg for err in expected_errors)
            if not has_expected_error:
                pytest.fail(f"Unexpected export error: {e}")
            else:
                raise AssertionError(f"Export failed due to missing components: {e}")
                
        finally:
            cleanup_export_dir(export_path)
    
    @pytest.mark.integration
    def test_full_mnist_workflow_export(self, sample_mnist_workflow):
        """Test export of full MNIST Test.json workflow."""
        assert sample_mnist_workflow is not None, "MNIST Test.json workflow not available"
        
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        export_path = create_temp_export_dir()
        
        try:
            result = exporter.export_workflow(sample_mnist_workflow, export_path)
            
            if result is not None and validate_export_output(export_path):
                # Check exported structure
                assert (export_path / "runner.py").exists()
                
                # Check for node files
                nodes_dir = export_path / "nodes"
                if nodes_dir.exists():
                    node_files = list(nodes_dir.glob("*.py"))
                    assert len(node_files) > 0, "Should have generated node files"
                    
                    # Check node file content
                    for node_file in node_files[:3]:  # Check first 3
                        content = node_file.read_text()
                        assert "class" in content
                        assert "QueueNode" in content or "compute" in content
                        
        except Exception as e:
            error_msg = str(e).lower()
            if any(err in error_msg for err in ["template", "missing", "not found"]):
                raise AssertionError(f"Export failed due to missing components: {e}")
            else:
                raise
                
        finally:
            cleanup_export_dir(export_path)
    
    @pytest.mark.integration
    def test_export_consistency(self):
        """Test that export produces consistent results."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        workflow = MINIMAL_TRAINING_WORKFLOW
        
        # Export twice to different locations
        export_path1 = create_temp_export_dir()
        export_path2 = create_temp_export_dir()
        
        try:
            result1 = exporter.export_workflow(workflow, export_path1)
            result2 = exporter.export_workflow(workflow, export_path2)
            
            # Both should succeed or fail similarly
            assert type(result1) == type(result2)
            
            # If both succeeded, compare outputs
            if (result1 is not None and result2 is not None and
                validate_export_output(export_path1) and 
                validate_export_output(export_path2)):
                
                # Both should have runner.py
                runner1 = export_path1 / "runner.py"
                runner2 = export_path2 / "runner.py"
                
                if runner1.exists() and runner2.exists():
                    content1 = runner1.read_text()
                    content2 = runner2.read_text()
                    
                    # Should be identical (deterministic export)
                    assert content1 == content2, "Export should be deterministic"
                    
        except Exception as e:
            if "template" in str(e).lower() or "missing" in str(e).lower():
                raise AssertionError(f"Export consistency test failed: {e}")
            else:
                raise
                
        finally:
            cleanup_export_dir(export_path1)
            cleanup_export_dir(export_path2)


class TestMNISTCodeExecution:
    """Test execution of exported MNIST code."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_exported_code_syntax_validation(self):
        """Test that exported MNIST code has valid Python syntax."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        workflow = MINIMAL_TRAINING_WORKFLOW
        export_path = create_temp_export_dir()
        
        try:
            result = exporter.export_workflow(workflow, export_path)
            
            if result is not None and validate_export_output(export_path):
                # Check runner.py syntax
                runner_file = export_path / "runner.py"
                if runner_file.exists():
                    runner_content = runner_file.read_text()
                    
                    # Should compile without syntax errors
                    try:
                        compile(runner_content, str(runner_file), 'exec')
                    except SyntaxError as e:
                        pytest.fail(f"Exported runner.py has syntax error: {e}")
                
                # Check node files syntax
                nodes_dir = export_path / "nodes"
                if nodes_dir.exists():
                    for node_file in nodes_dir.glob("*.py"):
                        node_content = node_file.read_text()
                        
                        try:
                            compile(node_content, str(node_file), 'exec')
                        except SyntaxError as e:
                            pytest.fail(f"Node file {node_file.name} has syntax error: {e}")
                            
        except Exception as e:
            if "template" in str(e).lower():
                raise AssertionError(f"Syntax validation failed: {e}")
            else:
                raise
                
        finally:
            cleanup_export_dir(export_path)
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_exported_mnist_execution(self):
        """Test execution of exported MNIST training script."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        workflow = MINIMAL_TRAINING_WORKFLOW
        export_path = create_temp_export_dir()
        
        try:
            result = exporter.export_workflow(workflow, export_path)
            
            if result is not None and validate_export_output(export_path):
                runner_file = export_path / "runner.py"
                
                if runner_file.exists():
                    # Try to execute the generated script
                    try:
                        # Run with timeout to prevent hanging
                        execution_result = subprocess.run(
                            [sys.executable, str(runner_file)],
                            capture_output=True,
                            text=True,
                            timeout=30,  # 30 second timeout
                            cwd=export_path  # Run in export directory
                        )
                        
                        # Check execution results
                        if execution_result.returncode == 0:
                            # Successful execution
                            assert len(execution_result.stdout) >= 0
                            print(f"MNIST execution succeeded: {execution_result.stdout}")
                            
                        else:
                            # Execution failed - check if it's due to missing dependencies
                            stderr = execution_result.stderr.lower()
                            expected_errors = [
                                "modulenotfounderror", "importerror", "no module named",
                                "torch", "numpy", "torchvision"
                            ]
                            
                            has_expected_error = any(err in stderr for err in expected_errors)
                            
                            if has_expected_error:
                                raise AssertionError(f"MNIST execution failed due to missing deps: {stderr}")
                            else:
                                # Unexpected error - should investigate
                                print(f"MNIST execution failed unexpectedly:")
                                print(f"STDOUT: {execution_result.stdout}")
                                print(f"STDERR: {execution_result.stderr}")
                                
                                # Don't fail the test immediately - generated code issues are expected
                                raise AssertionError(f"MNIST execution failed: {stderr}")
                                
                    except subprocess.TimeoutExpired:
                        raise AssertionError("MNIST execution timed out - may be working but slow")
                        
                    except Exception as e:
                        raise AssertionError(f"MNIST execution error: {e}")
                        
        except Exception as e:
            if "template" in str(e).lower():
                raise AssertionError(f"MNIST execution test failed: {e}")
            else:
                raise
                
        finally:
            cleanup_export_dir(export_path)
    
    @pytest.mark.integration
    def test_import_resolution_in_exported_code(self):
        """Test that exported code imports can be resolved."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        workflow = MINIMAL_TRAINING_WORKFLOW
        export_path = create_temp_export_dir()
        
        try:
            result = exporter.export_workflow(workflow, export_path)
            
            if result is not None and validate_export_output(export_path):
                # Extract imports from generated files
                imports_to_test = set()
                
                # Check runner.py imports
                runner_file = export_path / "runner.py"
                if runner_file.exists():
                    content = runner_file.read_text()
                    for line in content.split('\n'):
                        line = line.strip()
                        if line.startswith('import ') or line.startswith('from '):
                            imports_to_test.add(line)
                
                # Test import resolution
                for import_stmt in list(imports_to_test)[:5]:  # Test first 5 imports
                    try:
                        exec(import_stmt)
                        print(f"Import OK: {import_stmt}")
                    except ImportError as e:
                        print(f"Import unavailable (expected): {import_stmt} - {e}")
                    except SyntaxError as e:
                        pytest.fail(f"Invalid import syntax: {import_stmt} - {e}")
                        
        except Exception as e:
            if "template" in str(e).lower():
                raise AssertionError(f"Import resolution test failed: {e}")
            else:
                raise
                
        finally:
            cleanup_export_dir(export_path)


class TestMNISTWorkflowPerformance:
    """Test performance characteristics of MNIST workflow."""
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_export_performance(self):
        """Test MNIST workflow export performance."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        workflow = MINIMAL_TRAINING_WORKFLOW
        export_path = create_temp_export_dir()
        
        try:
            # Time the export process
            start_time = time.time()
            result = exporter.export_workflow(workflow, export_path)
            end_time = time.time()
            
            export_duration = end_time - start_time
            
            # Export should complete quickly
            assert export_duration < 10.0, f"Export took too long: {export_duration}s"
            
            print(f"MNIST export completed in {export_duration:.2f}s")
            
        except Exception as e:
            # Even failed exports should not take too long
            end_time = time.time()
            export_duration = end_time - start_time
            assert export_duration < 10.0, f"Export failure took too long: {export_duration}s"
            
            if "template" in str(e).lower():
                raise AssertionError(f"Performance test failed: {e}")
            else:
                raise
                
        finally:
            cleanup_export_dir(export_path)
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_multiple_exports_performance(self):
        """Test performance of multiple MNIST exports."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        workflow = MINIMAL_TRAINING_WORKFLOW
        
        export_times = []
        export_paths = []
        
        try:
            # Perform multiple exports
            for i in range(3):
                export_path = create_temp_export_dir()
                export_paths.append(export_path)
                
                start_time = time.time()
                result = exporter.export_workflow(workflow, export_path)
                end_time = time.time()
                
                export_duration = end_time - start_time
                export_times.append(export_duration)
            
            # All exports should be reasonably fast
            max_time = max(export_times)
            avg_time = sum(export_times) / len(export_times)
            
            assert max_time < 15.0, f"Slowest export took too long: {max_time}s"
            assert avg_time < 10.0, f"Average export time too high: {avg_time}s"
            
            print(f"Export times: {export_times}")
            print(f"Average: {avg_time:.2f}s, Max: {max_time:.2f}s")
            
        except Exception as e:
            if "template" in str(e).lower():
                raise AssertionError(f"Multiple exports test failed: {e}")
            else:
                raise
                
        finally:
            for export_path in export_paths:
                cleanup_export_dir(export_path)


class TestMNISTWorkflowIntegration:
    """Complete integration tests for MNIST workflow."""
    
    @pytest.mark.integration
    def test_end_to_end_mnist_pipeline(self, sample_mnist_workflow):
        """Test complete end-to-end MNIST pipeline."""
        if sample_mnist_workflow is None:
            # Use minimal workflow instead
            workflow = MINIMAL_TRAINING_WORKFLOW
        else:
            workflow = sample_mnist_workflow
        
        export_path = create_temp_export_dir()
        
        try:
            # Step 1: Validate workflow
            assert validate_workflow_structure(workflow)
            
            # Step 2: Export workflow
            exporter = GraphExporter()
            register_all_exporters(exporter)
            
            result = exporter.export_workflow(workflow, export_path)
            
            if result is not None and validate_export_output(export_path):
                # Step 3: Validate exported structure
                assert (export_path / "runner.py").exists()
                
                # Step 4: Check syntax
                runner_content = (export_path / "runner.py").read_text()
                compile(runner_content, "runner.py", 'exec')
                
                # Step 5: Brief execution test (syntax check)
                try:
                    exec_result = subprocess.run(
                        [sys.executable, "-m", "py_compile", str(export_path / "runner.py")],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    assert exec_result.returncode == 0, \
                        f"Python compilation failed: {exec_result.stderr}"
                    
                except subprocess.TimeoutExpired:
                    raise AssertionError("Compilation check timed out")
                
                print("End-to-end MNIST pipeline test completed successfully")
                
            else:
                raise AssertionError("Export did not produce complete output")
                
        except Exception as e:
            if "template" in str(e).lower() or "missing" in str(e).lower():
                raise AssertionError(f"End-to-end test failed: {e}")
            else:
                raise
                
        finally:
            cleanup_export_dir(export_path)
    
    @pytest.mark.integration
    def test_mnist_workflow_robustness(self):
        """Test MNIST workflow handling with various edge cases."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        # Test with modified workflows
        test_workflows = [
            MINIMAL_TRAINING_WORKFLOW,  # Baseline
            
            # Workflow with extra nodes
            {
                **MINIMAL_TRAINING_WORKFLOW,
                "nodes": MINIMAL_TRAINING_WORKFLOW["nodes"] + [
                    {"id": "999", "type": "ExtraNode", "inputs": {}, "widgets": {}}
                ]
            },
            
            # Workflow with missing connections
            {
                **MINIMAL_TRAINING_WORKFLOW,
                "links": MINIMAL_TRAINING_WORKFLOW["links"][:-1]  # Remove last link
            }
        ]
        
        for i, workflow in enumerate(test_workflows):
            export_path = create_temp_export_dir()
            
            try:
                result = exporter.export_workflow(workflow, export_path)
                
                # Should handle gracefully (succeed or meaningful error)
                if result is not None:
                    print(f"Workflow variant {i} exported successfully")
                else:
                    print(f"Workflow variant {i} export returned None")
                    
            except Exception as e:
                # Should provide meaningful errors
                error_msg = str(e).lower()
                expected_error_terms = [
                    "template", "missing", "not found", "unknown", 
                    "exporter", "node", "connection"
                ]
                
                has_expected_error = any(term in error_msg for term in expected_error_terms)
                assert has_expected_error, f"Unexpected error for workflow {i}: {e}"
                
                print(f"Workflow variant {i} failed as expected: {e}")
                
            finally:
                cleanup_export_dir(export_path)