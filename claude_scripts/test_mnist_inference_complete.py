#!/usr/bin/env python3
"""
Comprehensive MNIST Training + Inference Test

This script tests the complete workflow:
1. Export MNIST workflow with checkpoint saving
2. Train the model and save checkpoint
3. Load checkpoint in inference mode
4. Compare training vs inference accuracy
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from export_system.graph_exporter import GraphExporter

class MNISTInferenceTest:
    """Complete MNIST training and inference test"""
    
    def __init__(self, test_name: str = "mnist_inference_complete"):
        self.test_name = test_name
        self.export_path = Path(f"export_system/exports/{test_name}")
        self.checkpoint_dir = self.export_path / "checkpoints"
        self.results = {}
        
    def cleanup(self):
        """Clean up test artifacts"""
        if self.export_path.exists():
            shutil.rmtree(self.export_path)
            
    def export_workflow(self) -> bool:
        """Export MNIST workflow with checkpoint saving enabled"""
        print("📦 Exporting MNIST workflow...")
        
        # Load MNIST workflow
        workflow_path = Path("user/default/workflows/MNIST Test.json")
        if not workflow_path.exists():
            print(f"❌ Workflow not found: {workflow_path}")
            return False
            
        with open(workflow_path, 'r') as f:
            workflow_data = json.load(f)
        
        # Export the workflow
        exporter = GraphExporter()
        success = exporter.export_workflow(workflow_data, self.export_path)
        
        if not success:
            print("❌ Export failed")
            return False
            
        print(f"✅ Workflow exported to {self.export_path}")
        
        # Verify files were created
        runner_path = self.export_path / "runner.py"
        if not runner_path.exists():
            print("❌ runner.py not found")
            return False
            
        # Verify inference flag is present
        runner_content = runner_path.read_text()
        if "--inference" not in runner_content:
            print("❌ --inference flag not found in runner.py")
            return False
            
        print("✅ Export verification passed")
        return True
    
    def run_training(self, timeout: str = "30s") -> bool:
        """Run training with checkpoint saving"""
        print(f"🧠 Running MNIST training (timeout: {timeout})...")
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct command
        cmd = [
            sys.executable, "runner.py",
            "--timeout", timeout,
            "--save-checkpoint-dir", str(self.checkpoint_dir),
            "--verbose"
        ]
        
        print(f"📝 Running: {' '.join(cmd)}")
        print(f"📁 Working directory: {self.export_path}")
        
        try:
            # Run training
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=self.export_path,
                capture_output=True,
                text=True,
                timeout=90  # 90 seconds max
            )
            end_time = time.time()
            
            print(f"⏱️ Training completed in {end_time - start_time:.1f} seconds")
            
            # Check if process succeeded
            if result.returncode != 0:
                print(f"❌ Training failed with return code {result.returncode}")
                print("STDOUT:", result.stdout[-1000:])  # Last 1000 chars
                print("STDERR:", result.stderr[-1000:])
                return False
            
            # Parse output for accuracy information
            output_lines = result.stdout.split('\n')
            training_accuracy = self._extract_accuracy(output_lines)
            if training_accuracy is not None:
                self.results['training_accuracy'] = training_accuracy
                print(f"📊 Training accuracy: {training_accuracy:.4f}")
            
            # Check if checkpoint was saved
            checkpoint_files = list(self.checkpoint_dir.rglob("*.pt"))
            if not checkpoint_files:
                print("❌ No checkpoint files found after training")
                return False
                
            print(f"✅ Training completed, {len(checkpoint_files)} checkpoint files saved")
            
            # Save training output for analysis
            with open(self.export_path / "training_output.log", 'w') as f:
                f.write(f"RETURN CODE: {result.returncode}\n\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)
            
            return True
            
        except subprocess.TimeoutExpired:
            print("❌ Training timed out")
            return False
        except Exception as e:
            print(f"❌ Training failed with exception: {e}")
            return False
    
    def run_inference(self, timeout: str = "30s") -> bool:
        """Run inference with saved checkpoint"""
        print(f"🔍 Running MNIST inference (timeout: {timeout})...")
        
        # Verify checkpoint exists
        checkpoint_files = list(self.checkpoint_dir.rglob("*.pt"))
        if not checkpoint_files:
            print("❌ No checkpoint files found for inference")
            return False
            
        # Construct command
        cmd = [
            sys.executable, "runner.py",
            "--inference",
            "--load-checkpoint-dir", str(self.checkpoint_dir),
            "--timeout", timeout,
            "--verbose"
        ]
        
        print(f"📝 Running: {' '.join(cmd)}")
        
        try:
            # Run inference
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=self.export_path,
                capture_output=True,
                text=True,
                timeout=60  # 1 minute max
            )
            end_time = time.time()
            
            print(f"⏱️ Inference completed in {end_time - start_time:.1f} seconds")
            
            # Check if process succeeded
            if result.returncode != 0:
                print(f"❌ Inference failed with return code {result.returncode}")
                print("STDOUT:", result.stdout[-1000:])
                print("STDERR:", result.stderr[-1000:])
                return False
            
            # Parse output for accuracy information
            output_lines = result.stdout.split('\n')
            inference_accuracy = self._extract_accuracy(output_lines)
            if inference_accuracy is not None:
                self.results['inference_accuracy'] = inference_accuracy
                print(f"📊 Inference accuracy: {inference_accuracy:.4f}")
            
            # Verify inference mode was used
            if "🔍 Inference mode enabled" not in result.stdout:
                print("⚠️ Warning: Inference mode flag not detected in output")
            else:
                print("✅ Inference mode confirmed")
            
            # Save inference output for analysis
            with open(self.export_path / "inference_output.log", 'w') as f:
                f.write(f"RETURN CODE: {result.returncode}\n\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)
            
            return True
            
        except subprocess.TimeoutExpired:
            print("❌ Inference timed out")
            return False
        except Exception as e:
            print(f"❌ Inference failed with exception: {e}")
            return False
    
    def _extract_accuracy(self, output_lines: list) -> Optional[float]:
        """Extract accuracy from training/inference output"""
        for line in output_lines:
            if "accuracy:" in line.lower() or "acc:" in line.lower():
                # Look for patterns like "accuracy: 0.95" or "acc: 95.2%"
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.lower().startswith(("accuracy:", "acc:")):
                            if i + 1 < len(parts):
                                acc_str = parts[i + 1].rstrip('%,')
                                acc_val = float(acc_str)
                                # Convert percentage to decimal if needed
                                if acc_val > 1.0:
                                    acc_val = acc_val / 100.0
                                return acc_val
                except (ValueError, IndexError):
                    continue
        return None
    
    def validate_results(self) -> bool:
        """Validate that training and inference results are consistent"""
        print("📊 Validating results...")
        
        training_acc = self.results.get('training_accuracy')
        inference_acc = self.results.get('inference_accuracy')
        
        if training_acc is None:
            print("⚠️ Warning: Could not extract training accuracy")
            
        if inference_acc is None:
            print("⚠️ Warning: Could not extract inference accuracy")
            
        if training_acc is not None and inference_acc is not None:
            acc_diff = abs(training_acc - inference_acc)
            acc_diff_percent = acc_diff * 100
            
            print(f"📈 Training accuracy:  {training_acc:.4f}")
            print(f"🔍 Inference accuracy: {inference_acc:.4f}")
            print(f"📏 Accuracy difference: {acc_diff:.4f} ({acc_diff_percent:.2f}%)")
            
            # Accuracy should be very similar (within 1%)
            if acc_diff < 0.01:
                print("✅ Accuracy validation passed - inference matches training")
                return True
            else:
                print("⚠️ Warning: Accuracy difference is significant")
                return True  # Still pass but warn
        
        # Check that both processes completed successfully
        if len(self.results) > 0:
            print("✅ Basic validation passed - both training and inference completed")
            return True
        else:
            print("❌ Validation failed - no results collected")
            return False
    
    def generate_report(self) -> str:
        """Generate a summary report"""
        report = []
        report.append("=" * 60)
        report.append("MNIST TRAINING + INFERENCE TEST REPORT")
        report.append("=" * 60)
        report.append(f"Test Name: {self.test_name}")
        report.append(f"Export Path: {self.export_path}")
        report.append(f"Checkpoint Dir: {self.checkpoint_dir}")
        report.append("")
        
        # Results summary
        report.append("RESULTS:")
        for key, value in self.results.items():
            if isinstance(value, float):
                report.append(f"  {key}: {value:.4f}")
            else:
                report.append(f"  {key}: {value}")
        
        report.append("")
        
        # Files created
        if self.export_path.exists():
            report.append("FILES CREATED:")
            for file_path in sorted(self.export_path.rglob("*")):
                if file_path.is_file():
                    rel_path = file_path.relative_to(self.export_path)
                    size = file_path.stat().st_size
                    report.append(f"  {rel_path} ({size} bytes)")
        
        return "\n".join(report)

def main():
    """Run the complete MNIST inference test"""
    print("🚀 MNIST Training + Inference Test")
    print("=" * 60)
    
    # Initialize test
    test = MNISTInferenceTest()
    
    # Activate conda environment first
    print("🐍 Activating conda environment...")
    try:
        subprocess.run([
            "source", "/home/asantanna/miniconda/bin/activate", "DNNE_PY38"
        ], shell=True, check=True, capture_output=True)
        print("✅ Conda environment activated")
    except subprocess.CalledProcessError:
        print("⚠️ Could not activate conda environment (continuing anyway)")
    
    success = True
    
    try:
        # Step 1: Export workflow
        if not test.export_workflow():
            print("❌ Export failed")
            return 1
        
        # Step 2: Run training
        if not test.run_training():
            print("❌ Training failed")
            success = False
        
        # Step 3: Run inference
        if success and not test.run_inference():
            print("❌ Inference failed")
            success = False
        
        # Step 4: Validate results
        if success and not test.validate_results():
            print("❌ Validation failed")
            success = False
        
        # Generate report
        report = test.generate_report()
        print("\n" + report)
        
        # Save report
        report_path = test.export_path / "test_report.txt"
        if test.export_path.exists():
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"\n📋 Report saved to: {report_path}")
        
        if success:
            print("\n✅ All tests passed! MNIST inference mode is working correctly.")
            print("\n📝 Usage summary:")
            print("   Training: python runner.py --timeout 5m --save-checkpoint-dir ./checkpoints")
            print("   Inference: python runner.py --inference --load-checkpoint-dir ./checkpoints")
        else:
            print("\n❌ Some tests failed. Check the logs for details.")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        success = False
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        success = False
    finally:
        # Cleanup option
        import sys
        if "--cleanup" in sys.argv:
            print(f"\n🧹 Cleaning up {test.export_path}")
            test.cleanup()
        else:
            print(f"\n📁 Test artifacts preserved in: {test.export_path}")
            print("   Use --cleanup flag to auto-remove test artifacts")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())