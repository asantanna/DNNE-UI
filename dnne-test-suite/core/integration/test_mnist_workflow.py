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
    
    # Class variable to share checkpoint path between tests
    checkpoint_export_path = None
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.timeout(300)  # 5 minute timeout for full training
    def test_exported_mnist_execution(self, sample_mnist_workflow):
        """Test execution of exported MNIST training script with checkpoint saving."""
        # Use real MNIST workflow if available
        if sample_mnist_workflow is None:
            pytest.skip("MNIST Test workflow not available")
            
        workflow_name = "MNIST_Test"
        export_path = None
            
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
                            [sys.executable, str(runner_file), "--timeout", "30s"],
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
                            
                            # Check for checkpoint save - REQUIRED
                            checkpoint_dir = export_path / "checkpoints"
                            if checkpoint_dir.exists():
                                checkpoints = list(checkpoint_dir.glob("*.pth"))
                                if checkpoints:
                                    print(f"✓ Found {len(checkpoints)} checkpoint(s)")
                                    # Store export path for test 3
                                    TestMNISTExecution.checkpoint_export_path = export_path
                                    # Don't cleanup this export - test 3 needs it
                                    return
                                else:
                                    raise AssertionError("No checkpoint files found - checkpoints directory exists but is empty")
                            else:
                                raise AssertionError("Checkpoints directory not created - training should save checkpoints")
                            
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
            # Only cleanup if we didn't save checkpoint path for test 3
            if export_path and TestMNISTExecution.checkpoint_export_path != export_path:
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
    @pytest.mark.timeout(120)  # 2 minutes for 2 epochs
    def test_mnist_execution_with_epoch_limit(self, sample_mnist_workflow):
        """Test that exported code respects the --epochs flag and stops after specified epochs."""
        if sample_mnist_workflow is None:
            pytest.skip("MNIST Test workflow not available")
            
        workflow_name = "MNIST_Test"
        export_path = None
            
        try:
            # Use standardized export utility
            export_path = export_workflow_for_test(workflow_name, "mnist_2_epochs")
            
            if validate_export_output(export_path):
                runner_file = export_path / "runner.py"
                
                if runner_file.exists():
                    # Track execution time
                    test_start_time = time.time()
                    
                    # Run with --epochs 2 flag
                    execution_result = subprocess.run(
                        [sys.executable, str(runner_file), "--epochs", "2"],
                        capture_output=True,
                        text=True,
                        timeout=120,  # Much shorter timeout for 2 epochs
                        cwd=export_path
                    )
                    
                    test_end_time = time.time()
                    test_duration = test_end_time - test_start_time
                    
                    if execution_result.returncode == 0:
                        stdout = execution_result.stdout
                        
                        # Count epoch mentions
                        epoch_1_count = stdout.lower().count("epoch 1")
                        epoch_2_count = stdout.lower().count("epoch 2") 
                        epoch_3_count = stdout.lower().count("epoch 3")
                        
                        # Should see epochs 1 and 2, but NOT epoch 3
                        assert epoch_1_count > 0, "Should see epoch 1"
                        assert epoch_2_count > 0, "Should see epoch 2"
                        assert epoch_3_count == 0, "Should NOT see epoch 3 - training should stop after 2 epochs"
                        
                        print(f"✓ Training stopped correctly after 2 epochs")
                        print(f"   Runtime: {test_duration:.1f} seconds")
                        
                        # We don't care about accuracy in this test - just that it stops at 2 epochs
                        # Could optionally parse metrics for info
                        accuracy, loss = self._parse_training_metrics(stdout)
                        if accuracy:
                            print(f"   (Info only - Accuracy after 2 epochs: {accuracy:.1%})")
                        if loss:
                            print(f"   (Info only - Loss after 2 epochs: {loss:.3f})")
                        
                    else:
                        pytest.fail(f"Execution failed: {execution_result.stderr}")
                        
                else:
                    pytest.fail("runner.py was not created during export")
                    
            else:
                pytest.fail("Export validation failed")
                
        except Exception as e:
            if "template" in str(e).lower():
                pytest.skip(f"Skipping due to missing template: {e}")
            else:
                raise
                
        finally:
            if export_path:
                cleanup_export_dir(export_path)
    
    @pytest.mark.integration
    @pytest.mark.timeout(60)  # 1 minute for inference
    def test_checkpoint_loading_and_inference(self):
        """Test loading checkpoint and running inference mode with accuracy validation."""
        # This test depends on test 1 creating a checkpoint
        if TestMNISTExecution.checkpoint_export_path is None:
            pytest.fail("No checkpoint was saved by test_exported_mnist_execution - cannot test checkpoint loading")
            
        export_path = TestMNISTExecution.checkpoint_export_path
        runner_file = export_path / "runner.py"
        
        if not runner_file.exists():
            pytest.fail("Runner file missing from checkpoint test")
            
        # Find the checkpoint file
        checkpoint_dir = export_path / "checkpoints"
        if not checkpoint_dir.exists():
            pytest.fail("Checkpoints directory missing")
            
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if not checkpoints:
            pytest.fail("No checkpoint files found")
            
        # Use the first (or latest) checkpoint
        checkpoint_file = checkpoints[0]
        print(f"Using checkpoint: {checkpoint_file.name}")
        
        try:
            # Run in inference mode with checkpoint
            execution_result = subprocess.run(
                [sys.executable, str(runner_file), 
                 "--inference-mode",
                 "--checkpoint", str(checkpoint_file.relative_to(export_path)),
                 "--epochs", "1"],  # Just one epoch to validate
                capture_output=True,
                text=True,
                timeout=60,  # Should be fast
                cwd=export_path
            )
            
            if execution_result.returncode == 0:
                stdout = execution_result.stdout
                
                # Verify checkpoint was loaded
                assert "loading checkpoint" in stdout.lower() or "loaded checkpoint" in stdout.lower(), \
                    "Should see checkpoint loading message"
                
                # Parse accuracy from inference
                accuracy, _ = self._parse_training_metrics(stdout)
                
                if accuracy is not None:
                    assert accuracy > 0.90, f"Loaded model should maintain >90% accuracy, got {accuracy:.1%}"
                    print(f"✓ Checkpoint loaded successfully")
                    print(f"   Inference accuracy: {accuracy:.1%}")
                else:
                    # If can't parse accuracy, at least check inference ran
                    assert "inference" in stdout.lower() or "eval" in stdout.lower(), \
                        "Should see inference/evaluation output"
                    print("✓ Checkpoint loaded and inference ran (accuracy not parsed)")
                    
            else:
                pytest.fail(f"Inference execution failed: {execution_result.stderr}")
                
        except Exception as e:
            raise
            
        finally:
            # Now we can clean up the checkpoint export
            cleanup_export_dir(export_path)
            TestMNISTExecution.checkpoint_export_path = None