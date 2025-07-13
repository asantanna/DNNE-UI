#!/usr/bin/env python3
"""
Test export of updated Cartpole_PPO workflow with environment-specific parameters
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export_system.graph_exporter import GraphExporter
from export_system.node_exporters import register_all_exporters

def test_cartpole_ppo_export():
    """Test export of Cartpole PPO workflow with updated templates"""
    print("Testing Updated Cartpole_PPO Workflow Export...")
    print("-" * 50)
    
    # Create exporter
    exporter = GraphExporter()
    register_all_exporters(exporter)
    
    print(f"✓ Loaded export system with {len(exporter.node_registry)} node types")
    
    # Load Cartpole_PPO workflow
    workflow_path = Path("user/default/workflows/Cartpole_PPO.json")
    if not workflow_path.exists():
        print(f"✗ Workflow not found: {workflow_path}")
        return False
    
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    print(f"✓ Loaded workflow with {len(workflow.get('nodes', []))} nodes")
    
    # List nodes for verification
    nodes = workflow.get('nodes', [])
    node_types = [node.get('type') for node in nodes]
    print(f"  Node types: {', '.join(node_types)}")
    
    # Export the workflow with updated templates
    try:
        output_path = Path("export_system/exports/Cartpole_PPO_Updated")
        print(f"Exporting to: {output_path}")
        
        result = exporter.export_workflow(workflow, output_path)
        print("✓ Export successful!")
        
        # Check if runner.py was created
        runner_path = output_path / "runner.py"
        if runner_path.exists():
            print("✓ runner.py generated successfully")
            
            # Read the generated runner to check for updated features
            with open(runner_path, 'r') as f:
                runner_content = f.read()
            
            print(f"✓ Generated runner.py ({len(runner_content)} characters)")
            
            # Check for Isaac Gym nodes
            isaac_env_found = "IsaacGymEnvNode" in runner_content
            isaac_step_found = "IsaacGymStepNode" in runner_content
            
            print(f"  IsaacGymEnvNode: {'✓' if isaac_env_found else '✗'}")
            print(f"  IsaacGymStepNode: {'✓' if isaac_step_found else '✗'}")
            
            return isaac_env_found and isaac_step_found
        else:
            print("✗ runner.py not found in generated output")
            return False
            
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_generated_features(export_path):
    """Check if the generated code includes our new features"""
    print("\nChecking Generated Code Features...")
    print("-" * 40)
    
    # Check IsaacGymEnv node for environment-specific parameters
    isaac_env_files = list(export_path.glob("**/isaacgymenvnode_*.py"))
    if isaac_env_files:
        env_file = isaac_env_files[0]
        print(f"✓ Found IsaacGymEnv file: {env_file.name}")
        
        with open(env_file, 'r') as f:
            env_content = f.read()
        
        # Check for environment-specific parameter setting
        cartpole_dt_set = "# Set Cartpole-specific simulation parameters" in env_content
        dt_in_cartpole = "self.sim_params.dt = 0.0166" in env_content and "_create_cartpole_environments" in env_content
        
        print(f"  Environment-specific parameters: {'✓' if cartpole_dt_set else '✗'}")
        print(f"  Cartpole dt setting: {'✓' if dt_in_cartpole else '✗'}")
    else:
        print("✗ No IsaacGymEnv files found")
        cartpole_dt_set = dt_in_cartpole = False
    
    # Check IsaacGymStep node for dual-mode execution
    isaac_step_files = list(export_path.glob("**/isaacgymstepnode_*.py"))
    if isaac_step_files:
        step_file = isaac_step_files[0]
        print(f"✓ Found IsaacGymStep file: {step_file.name}")
        
        with open(step_file, 'r') as f:
            step_content = f.read()
        
        # Check for dual-mode execution features
        dual_mode_run = "async def run(self):" in step_content and "inference_mode" in step_content
        training_mode = "_run_training_mode" in step_content
        inference_mode = "_run_inference_mode" in step_content
        real_time_timing = "target_dt" in step_content and "sleep_time" in step_content
        
        print(f"  Dual-mode run method: {'✓' if dual_mode_run else '✗'}")
        print(f"  Training mode method: {'✓' if training_mode else '✗'}")
        print(f"  Inference mode method: {'✓' if inference_mode else '✗'}")
        print(f"  Real-time timing: {'✓' if real_time_timing else '✗'}")
    else:
        print("✗ No IsaacGymStep files found")
        dual_mode_run = training_mode = inference_mode = real_time_timing = False
    
    # Overall assessment
    features_working = all([
        cartpole_dt_set, dt_in_cartpole,
        dual_mode_run, training_mode, inference_mode, real_time_timing
    ])
    
    print(f"\n{'✓ All features implemented correctly!' if features_working else '✗ Some features missing or incorrect'}")
    return features_working

def main():
    print("DNNE Updated Cartpole Export Test")
    print("=" * 60)
    
    success = True
    
    # Test 1: Export workflow
    if test_cartpole_ppo_export():
        export_path = Path("export_system/exports/Cartpole_PPO_Updated")
        
        # Test 2: Check generated features
        if not check_generated_features(export_path):
            success = False
    else:
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✓ Environment-specific simulation parameters implemented")
        print("✓ Dual-mode execution (training/inference) implemented")
        print("✓ Real-time timing for inference mode implemented")
        print("\nThe updated Cartpole PPO system is ready for testing!")
    else:
        print("❌ Some tests failed.")
        print("Please check the template implementation and try again.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)