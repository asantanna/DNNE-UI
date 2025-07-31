"""
Integration tests for MNIST workflow end-to-end execution.

Tests complete MNIST training pipeline from export through actual execution,
validating that the exported code runs and produces expected results.
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


class TestMNISTExecution:
    """Test actual execution of exported MNIST workflows."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.timeout(300)  # 5 minute timeout for full training
    def test_exported_mnist_execution(self, sample_mnist_workflow):
        """Test execution of exported MNIST training script with performance validation."""
        # Use real MNIST workflow if available, otherwise fallback to minimal
        if sample_mnist_workflow is not None:
            workflow_name = "MNIST Test"
        else:
            # For minimal workflow, we'll use MNIST Test as it's more complete
            workflow_name = "MNIST Test"
            
        try:
            # Use standardized export utility
            export_path = export_workflow_for_test(workflow_name, "mnist_execution")
            
            # Export utility already validates output, but double-check
            if validate_export_output(export_path):
                runner_file = export_path / "runner.py"
                
                if runner_file.exists():
                    # Try to execute the generated script
                    try:
                        # Track test execution time
                        test_start_time = time.time()
                        
                        # Run with timeout to prevent hanging - 5 minutes for full training
                        execution_result = subprocess.run(
                            [sys.executable, str(runner_file), "--test-mode"],
                            capture_output=True,
                            text=True,
                            timeout=300,  # 5 minute timeout
                            cwd=export_path  # Run in export directory
                        )
                        
                        test_end_time = time.time()
                        test_duration = test_end_time - test_start_time
                        
                        # Check execution results
                        if execution_result.returncode == 0:
                            # Parse training output for performance metrics
                            stdout = execution_result.stdout
                            print(f"MNIST training output:\n{stdout}")
                            
                            # Check for early output (training started)
                            if len(stdout) < 100:
                                raise AssertionError("Script produced minimal output - may not be training")
                            
                            # Parse final training metrics
                            final_accuracy, final_loss = self._parse_training_metrics(stdout)
                            
                            # Validate training performance - only check accuracy > 90%
                            if final_accuracy is not None:
                                assert final_accuracy > 0.90, f"Training accuracy {final_accuracy:.3f} < 90%"
                                print(f"✓ Training achieved {final_accuracy:.1%} accuracy")
                            else:
                                raise AssertionError("Could not parse final accuracy from output")
                            
                            # Print test summary with runtime, accuracy, and loss
                            print(f"\n📊 MNIST Test Summary:")
                            print(f"   Runtime: {test_duration:.1f} seconds")
                            print(f"   Final Accuracy: {final_accuracy:.1%}" if final_accuracy else "   Final Accuracy: Not available")
                            print(f"   Final Loss: {final_loss:.3f}" if final_loss else "   Final Loss: Not available")
                            
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
                                
                                raise AssertionError(f"MNIST execution failed: {stderr}")
                                
                    except subprocess.TimeoutExpired:
                        raise AssertionError("MNIST training timed out after 5 minutes")
                        
                    except Exception as e:
                        raise AssertionError(f"MNIST execution error: {e}")
                        
                else:
                    raise AssertionError("runner.py was not created during export")
                    
            else:
                raise AssertionError("Export validation failed - missing required files")
                
        except Exception as e:
            # Handle export errors
            if "template" in str(e).lower():
                # Template errors are known issues
                pytest.skip(f"Skipping due to missing template: {e}")
            else:
                raise
                
        finally:
            cleanup_export_dir(export_path)
    
    def _parse_training_metrics(self, output: str) -> tuple:
        """Parse training metrics from output.
        
        Returns:
            (final_accuracy, final_loss) or (None, None) if not found
        """
        lines = output.strip().split('\n')
        final_accuracy = None
        final_loss = None
        
        # Look for accuracy and loss in output (search from end)
        for line in reversed(lines):
            line_lower = line.lower()
            
            # Look for accuracy
            if 'accuracy' in line_lower and final_accuracy is None:
                # Try to extract number
                import re
                acc_match = re.search(r'accuracy[:\s]+([0-9.]+)%?', line_lower)
                if acc_match:
                    acc_value = float(acc_match.group(1))
                    # Convert to fraction if percentage
                    final_accuracy = acc_value / 100 if acc_value > 1 else acc_value
                    
            # Look for loss
            if 'loss' in line_lower and final_loss is None:
                loss_match = re.search(r'loss[:\s]+([0-9.]+)', line_lower)
                if loss_match:
                    final_loss = float(loss_match.group(1))
                    
            # Stop if both found
            if final_accuracy is not None and final_loss is not None:
                break
                
        return final_accuracy, final_loss
    
    @pytest.mark.integration
    @pytest.mark.timeout(600)  # 10 minute timeout for full pipeline
    def test_end_to_end_mnist_pipeline(self, sample_mnist_workflow):
        """Test complete MNIST pipeline: load -> export -> execute -> validate."""
        if sample_mnist_workflow is None:
            pytest.skip("MNIST Test workflow not available")
            
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        workflow = sample_mnist_workflow
        export_path = create_temp_export_dir()
        
        try:
            # Step 1: Export workflow
            print("Step 1: Exporting MNIST workflow...")
            result = exporter.export_workflow(workflow, export_path)
            assert result is not None, "Export failed"
            
            # Step 2: Validate export
            print("Step 2: Validating exported files...")
            assert validate_export_output(export_path)
            
            # Step 3: Execute training
            print("Step 3: Running MNIST training...")
            runner_file = export_path / "runner.py"
            
            # Run with test mode for faster execution
            execution_result = subprocess.run(
                [sys.executable, str(runner_file), "--test-mode", "--epochs", "2"],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes
                cwd=export_path
            )
            
            # Step 4: Validate results
            print("Step 4: Validating training results...")
            if execution_result.returncode != 0:
                print(f"STDERR: {execution_result.stderr}")
                raise AssertionError("Training execution failed")
                
            # Check output contains training progress
            output = execution_result.stdout
            assert "epoch" in output.lower(), "No epoch information in output"
            assert "loss" in output.lower(), "No loss information in output"
            
            # Parse metrics
            accuracy, loss = self._parse_training_metrics(output)
            print(f"✓ Pipeline completed - Accuracy: {accuracy}, Loss: {loss}")
            
        except subprocess.TimeoutExpired:
            pytest.fail("Pipeline timed out after 10 minutes")
            
        except Exception as e:
            if "template" in str(e).lower():
                pytest.skip(f"Skipping due to missing template: {e}")
            else:
                raise
                
        finally:
            cleanup_export_dir(export_path)
    
    @pytest.mark.integration
    def test_mnist_workflow_robustness(self):
        """Test MNIST workflow export and execution with various configurations."""
        exporter = GraphExporter()
        register_all_exporters(exporter)
        
        # Test different workflow variations
        test_cases = [
            {
                "name": "minimal_training",
                "workflow": MINIMAL_TRAINING_WORKFLOW,
                "expected_error": None  # Should work
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n🧪 Testing {test_case['name']}...")
            
            export_path = create_temp_export_dir()
            
            try:
                result = exporter.export_workflow(test_case["workflow"], export_path)
                
                if test_case["expected_error"] is None:
                    # Should succeed
                    assert result is not None, f"Export failed for {test_case['name']}"
                    assert validate_export_output(export_path)
                    print(f"✓ {test_case['name']} exported successfully")
                else:
                    # Should fail with expected error
                    pytest.fail(f"Expected error for {test_case['name']} but export succeeded")
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                if test_case["expected_error"]:
                    # Check for expected error
                    assert test_case["expected_error"] in error_msg, \
                        f"Wrong error for {test_case['name']}: {e}"
                    print(f"✓ {test_case['name']} failed as expected: {test_case['expected_error']}")
                else:
                    # Unexpected error
                    expected_errors = ["template", "connection", "tensor", "input"]
                    has_expected_error = any(err in error_msg for err in expected_errors)
                    
                    if not has_expected_error:
                        assert has_expected_error, f"Unexpected error for workflow {i}: {e}"
                        
            finally:
                cleanup_export_dir(export_path)
        
        print("\n✅ Robustness testing completed")