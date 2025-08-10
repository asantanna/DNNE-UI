"""
Integration tests for MNIST workflow end-to-end execution.

Tests complete MNIST training pipeline from export through actual execution,
validating that the exported code runs and produces expected results.
"""

import pytest
import json
import subprocess
import sys
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

# Import DNNE components
from export_system.graph_exporter import GraphExporter
from export_system.node_exporters import register_all_exporters
from fixtures.workflows import MINIMAL_TRAINING_WORKFLOW
from fixtures.test_utils import (
    validate_workflow_structure, validate_export_output,
    create_temp_export_dir, cleanup_export_dir, export_workflow_for_test
)


# Module-level variable to store checkpoint info file
CHECKPOINT_INFO_FILE = Path(tempfile.gettempdir()) / "dnne_test_checkpoint_info.json"


class TestMNISTExecution:
    """Test actual execution of exported MNIST workflows."""
    
    # Network node ID from MNIST_Test.json workflow
    NETWORK_NODE_ID = 56 
    
    @pytest.mark.integration
    @pytest.mark.timeout(120)  # 2 minutes for 2 epochs
    def test_1_mnist_execution_with_epoch_limit(self, sample_mnist_workflow):
        """Test that exported code respects the --epochs flag and stops after specified epochs.
        
        This is test 1: Verifies --epochs 1 flag works correctly.
        """
        
        if sample_mnist_workflow is None:
            pytest.fail("MNIST_Test workflow not available")
            
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
                    
                    # Run with --epochs 1 flag
                    cmd = [sys.executable, str(runner_file), "--epochs", "1"]
                    print(f"Running command: {' '.join(cmd)}")
                    print(f"Working directory: {export_path}")
                    print(f"Python executable: {sys.executable}")
                    
                    try:
                        execution_result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=120,  # Much shorter timeout for 2 epochs
                            cwd=export_path
                        )
                    except subprocess.TimeoutExpired as e:
                        print(f"Process timed out after 120 seconds")
                        print(f"Partial STDOUT: {e.stdout if e.stdout else 'None'}")
                        print(f"Partial STDERR: {e.stderr if e.stderr else 'None'}")
                        pytest.fail("MNIST training timed out - process may be stuck during initialization")
                    
                    test_end_time = time.time()
                    test_duration = test_end_time - test_start_time
                    
                    if execution_result.returncode == 0:
                        stdout = execution_result.stdout
                        
                        # Count epoch mentions
                        epoch_1_count = stdout.lower().count("epoch 1")
                        epoch_2_count = stdout.lower().count("epoch 2") 
                        epoch_3_count = stdout.lower().count("epoch 3")
                        
                        # Should see epoch 1, but NOT epoch 2
                        assert epoch_1_count > 0, "Should see epoch 1"
                        assert epoch_2_count == 0, "Should NOT see epoch 2 - training should stop after 1 epoch"
                        
                        print(f"✓ Training stopped correctly after 1 epoch")
                        print(f"   Runtime: {test_duration:.1f} seconds")
                        
                        # We don't care about accuracy in this test - just that it stops at 1 epoch
                        # Could optionally parse metrics for info
                        accuracy, loss = self._parse_training_metrics(stdout)
                        if accuracy:
                            print(f"   (Info only - Accuracy after 1 epoch: {accuracy:.1%})")
                        if loss:
                            print(f"   (Info only - Loss after 1 epoch: {loss:.3f})")
                        
                    else:
                        print(f"STDOUT: {execution_result.stdout}")
                        print(f"STDERR: {execution_result.stderr}")
                        pytest.fail(f"Execution failed with return code {execution_result.returncode}: {execution_result.stderr}")
                        
                else:
                    pytest.fail("runner.py was not created during export")
                    
            else:
                pytest.fail("Export validation failed")
                
        except Exception as e:
            if "template" in str(e).lower():
                pytest.fail(f"Missing template: {e}")
            else:
                raise
                
        finally:
            if export_path:
                cleanup_export_dir(export_path)

                
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.timeout(300)  # 5 minute timeout for full training
    def test_2_exported_mnist_execution(self, sample_mnist_workflow):
        """Test execution of exported MNIST training script with checkpoint saving.
        
        This is test 2: Runs 3 epochs of training and saves checkpoint at the end.
        The checkpoint is used by test 3 for inference validation.
        """
        
        # Use real MNIST workflow if available
        if sample_mnist_workflow is None:
            pytest.fail("MNIST_Test workflow not available")
            
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
                        
                        # Run with epochs and checkpoint saving
                        execution_result = subprocess.run(
                            [sys.executable, str(runner_file), 
                             "--epochs", "3",
                             "--save-checkpoint", 
                             "--out-dir", "checkpoints",
                             "--override", f"{self.NETWORK_NODE_ID}:checkpoint_enabled=True,{self.NETWORK_NODE_ID}:checkpoint_trigger_type=end"],
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
                            # Only print first and last 500 chars for brevity
                            if len(stdout) > 1000:
                                print(f"MNIST training output (truncated):\n{stdout[:500]}\n...\n{stdout[-500:]}")
                            else:
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
                            print(f"\n📊 MNIST_Test Summary:")
                            print(f"   Runtime: {test_duration:.1f} seconds")
                            print(f"   Final Accuracy: {final_accuracy:.1%}" if final_accuracy else "   Final Accuracy: Not available")
                            print(f"   Final Loss: {final_loss:.3f}" if final_loss else "   Final Loss: Not available")
                            
                            # Check for checkpoint save - REQUIRED
                            checkpoint_dir = export_path / "checkpoints"
                            if checkpoint_dir.exists():
                                # Check for node-specific checkpoint directory
                                node_checkpoint_dir = checkpoint_dir / f"node_{self.NETWORK_NODE_ID}"
                                if node_checkpoint_dir.exists():
                                    model_file = node_checkpoint_dir / "model.pt"
                                    metadata_file = node_checkpoint_dir / "metadata.json"
                                    
                                    if model_file.exists() and metadata_file.exists():
                                        print(f"✓ Found checkpoint files in {node_checkpoint_dir.relative_to(export_path)}")
                                        print(f"   - model.pt: {model_file.stat().st_size} bytes")
                                        print(f"   - metadata.json: {metadata_file.stat().st_size} bytes")
                                        
                                        # Store export path for test 3
                                        checkpoint_info = {
                                            "export_path": str(export_path),
                                            "checkpoint_dir": str(checkpoint_dir),
                                            "node_checkpoint_dir": str(node_checkpoint_dir)
                                        }
                                        CHECKPOINT_INFO_FILE.write_text(json.dumps(checkpoint_info))
                                        print(f"✓ Saved checkpoint info to {CHECKPOINT_INFO_FILE}")
                                        # Don't cleanup this export - test 3 needs it
                                        return
                                    else:
                                        missing = []
                                        if not model_file.exists():
                                            missing.append("model.pt")
                                        if not metadata_file.exists():
                                            missing.append("metadata.json")
                                        raise AssertionError(f"Missing checkpoint files in {node_checkpoint_dir}: {', '.join(missing)}")
                                else:
                                    raise AssertionError(f"Node checkpoint directory not found: expected checkpoints/node_{self.NETWORK_NODE_ID}/")
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
                pytest.fail(f"Missing template: {e}")
            else:
                raise
                
        finally:
            # Only cleanup if we didn't save checkpoint for test 3
            if export_path and not CHECKPOINT_INFO_FILE.exists():
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
    @pytest.mark.timeout(180)  # 3 minutes for inference
    def test_3_checkpoint_loading_and_inference(self):
        """Test loading checkpoint and running inference mode with accuracy validation.
        
        This is test 3: Loads checkpoint from test 2 and verifies accuracy > 90% without learning.
        """
        
        # This test depends on test 2 creating a checkpoint
        if not CHECKPOINT_INFO_FILE.exists():
            pytest.fail("No checkpoint was saved by test_2_exported_mnist_execution - cannot test checkpoint loading")
            
        # Load checkpoint info
        checkpoint_info = json.loads(CHECKPOINT_INFO_FILE.read_text())
        export_path = Path(checkpoint_info["export_path"])
        
        if not export_path.exists():
            pytest.fail(f"Export path no longer exists: {export_path}")
        runner_file = export_path / "runner.py"
        
        if not runner_file.exists():
            pytest.fail("Runner file missing from checkpoint test")
            
        # Verify checkpoint directory structure
        checkpoint_dir = export_path / "checkpoints"
        if not checkpoint_dir.exists():
            pytest.fail("Checkpoints directory missing")
            
        node_checkpoint_dir = checkpoint_dir / f"node_{self.NETWORK_NODE_ID}"
        if not node_checkpoint_dir.exists():
            pytest.fail(f"Node checkpoint directory missing: expected checkpoints/node_{self.NETWORK_NODE_ID}/")
            
        model_file = node_checkpoint_dir / "model.pt"
        if not model_file.exists():
            pytest.fail(f"Model checkpoint file missing: {model_file}")
            
        print(f"Found checkpoint directory: {checkpoint_dir.relative_to(export_path)}")
        print(f"Loading from: {node_checkpoint_dir.relative_to(export_path)}")
        
        # Build command
        cmd = [sys.executable, str(runner_file), 
               "--inference",
               "--load-checkpoint", "checkpoints",
               "--epochs", "1"]
        print(f"Running inference command: {' '.join(cmd)}")
        print(f"Working directory: {export_path}")
        
        try:
            # Run in inference mode with checkpoint loading
            execution_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,  # 3 minutes should be enough
                cwd=export_path
            )
            
            if execution_result.returncode == 0:
                stdout = execution_result.stdout
                
                # Verify we're in inference mode
                assert "inference mode enabled" in stdout.lower(), \
                    "Should see inference mode message"
                
                # Verify checkpoint was loaded 
                assert "loading checkpoints from: checkpoints" in stdout.lower() or \
                       "loaded checkpoint" in stdout.lower(), \
                    "Should see checkpoint loading message"
                
                # Parse accuracy from inference
                accuracy, _ = self._parse_training_metrics(stdout)
                
                if accuracy is not None:
                    assert accuracy > 0.90, f"Loaded model should maintain >90% accuracy, got {accuracy:.1%}"
                    print(f"✓ Checkpoint loaded successfully")
                    print(f"   Inference accuracy: {accuracy:.1%}")
                else:
                    # If can't parse accuracy, at least check for epoch output
                    assert "epoch 1" in stdout.lower(), \
                        "Should see epoch 1 output in inference mode"
                    print("✓ Checkpoint loaded and inference ran (accuracy not parsed)")
                    print("   Note: Consider improving accuracy parsing for inference mode")
                    
            else:
                pytest.fail(f"Inference execution failed: {execution_result.stderr}")
                
        except Exception as e:
            raise
            
        finally:
            # Now we can clean up the checkpoint export and info file
            if export_path.exists():
                cleanup_export_dir(export_path)
            if CHECKPOINT_INFO_FILE.exists():
                CHECKPOINT_INFO_FILE.unlink()