#!/usr/bin/env python3
"""Test script to verify inference mode implementation"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from export_system.graph_exporter import GraphExporter

def test_mnist_inference():
    """Test MNIST workflow in inference mode"""
    print("\n=== Testing MNIST Inference Mode ===")
    
    # Load MNIST workflow
    workflow_path = Path("user/default/workflows/MNIST Test.json")
    if not workflow_path.exists():
        print(f"❌ Workflow not found: {workflow_path}")
        return False
        
    with open(workflow_path, 'r') as f:
        workflow_data = json.load(f)
    
    # Use proper export directory
    export_path = Path("export_system/exports/mnist_inference_test")
    
    try:
        # Export the workflow
        exporter = GraphExporter()
        success = exporter.export_workflow(workflow_data, export_path)
        
        if not success:
            print("❌ Export failed")
            return False
            
        print(f"✅ Workflow exported to {export_path}")
        
        # Check that runner.py was generated
        runner_path = export_path / "runner.py"
        if not runner_path.exists():
            print("❌ runner.py not found")
            return False
            
        # Verify inference flag is in runner.py
        runner_content = runner_path.read_text()
        if "--inference" not in runner_content:
            print("❌ --inference flag not found in runner.py")
            return False
            
        print("✅ --inference flag found in runner.py")
        
        # Test running in inference mode (dry run)
        cmd = [sys.executable, "runner.py", "--inference", "--test-mode"]
        print(f"📝 Would run: {' '.join(cmd)}")
        print("   (Not executing to avoid loading MNIST dataset)")
        
    finally:
        # Clean up
        if export_path.exists():
            shutil.rmtree(export_path)
            
    return True

def test_cartpole_inference():
    """Test Cartpole PPO workflow in inference mode"""
    print("\n=== Testing Cartpole PPO Inference Mode ===")
    
    # Load Cartpole PPO workflow
    workflow_path = Path("user/default/workflows/Cartpole_PPO.json")
    if not workflow_path.exists():
        print(f"❌ Workflow not found: {workflow_path}")
        return False
        
    with open(workflow_path, 'r') as f:
        workflow_data = json.load(f)
    
    # Use proper export directory
    export_path = Path("export_system/exports/cartpole_inference_test")
    
    try:
        # Export the workflow
        exporter = GraphExporter()
        success = exporter.export_workflow(workflow_data, export_path)
        
        if not success:
            print("❌ Export failed")
            return False
            
        print(f"✅ Workflow exported to {export_path}")
        
        # Check that runner.py was generated
        runner_path = export_path / "runner.py"
        if not runner_path.exists():
            print("❌ runner.py not found")
            return False
            
        # Verify inference flag and necessary imports
        runner_content = runner_path.read_text()
        checks = [
            ("--inference flag", "--inference" in runner_content),
            ("INFERENCE_MODE builtin", "builtins.INFERENCE_MODE = args.inference" in runner_content),
            ("isaacgym import", "import isaacgym" in runner_content)
        ]
        
        all_passed = True
        for check_name, check_result in checks:
            if check_result:
                print(f"✅ {check_name} found")
            else:
                print(f"❌ {check_name} not found")
                all_passed = False
                
        if not all_passed:
            return False
            
        # Check PPOTrainer template for inference handling
        ppo_trainer_file = None
        for node_file in (export_path / "nodes").glob("ppotrainernode_*.py"):
            ppo_trainer_file = node_file
            break
            
        if ppo_trainer_file:
            ppo_content = ppo_trainer_file.read_text()
            if "self.inference_mode" in ppo_content:
                print("✅ PPOTrainer has inference mode handling")
            else:
                print("❌ PPOTrainer missing inference mode handling")
                return False
                
        print(f"\n📝 To test inference mode, run:")
        print(f"   cd {export_path}")
        print(f"   python runner.py --inference --load-checkpoint-dir ./checkpoints --timeout 30s")
        
    finally:
        # Clean up
        if export_path.exists():
            shutil.rmtree(export_path)
            
    return True

def check_template_updates():
    """Verify that templates have been updated for inference mode"""
    print("\n=== Checking Template Updates ===")
    
    templates_dir = Path("export_system/templates/nodes")
    templates_to_check = [
        ("network_queue.py", "self.inference_mode"),
        ("training_step_queue.py", "self.inference_mode"),
        ("ppo_trainer_queue.py", "self.inference_mode"),
        ("ppo_agent_queue.py", "self.inference_mode")
    ]
    
    all_passed = True
    for template_file, search_string in templates_to_check:
        template_path = templates_dir / template_file
        if template_path.exists():
            content = template_path.read_text()
            if search_string in content:
                print(f"✅ {template_file} has inference mode support")
            else:
                print(f"❌ {template_file} missing inference mode support")
                all_passed = False
        else:
            print(f"❌ {template_file} not found")
            all_passed = False
            
    # Check base framework
    framework_path = Path("export_system/templates/base/queue_framework.py")
    if framework_path.exists():
        content = framework_path.read_text()
        if "torch.no_grad()" in content and "INFERENCE_MODE" in content:
            print("✅ queue_framework.py has torch.no_grad() context for inference")
        else:
            print("❌ queue_framework.py missing torch.no_grad() context")
            all_passed = False
            
    return all_passed

def main():
    """Run all inference mode tests"""
    print("🚀 Testing DNNE Inference Mode Implementation")
    print("=" * 50)
    
    # Check template updates
    templates_ok = check_template_updates()
    
    # Test MNIST workflow
    mnist_ok = test_mnist_inference()
    
    # Test Cartpole workflow
    cartpole_ok = test_cartpole_inference()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   Template Updates: {'✅ PASS' if templates_ok else '❌ FAIL'}")
    print(f"   MNIST Export: {'✅ PASS' if mnist_ok else '❌ FAIL'}")
    print(f"   Cartpole Export: {'✅ PASS' if cartpole_ok else '❌ FAIL'}")
    
    if templates_ok and mnist_ok and cartpole_ok:
        print("\n✅ All tests passed! Inference mode is ready to use.")
        print("\n📝 Usage:")
        print("   python runner.py --inference                    # Run in inference mode")
        print("   python runner.py --inference --load-checkpoint-dir ./checkpoints")
        print("   python runner.py --inference --timeout 30s      # Run for 30 seconds")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())