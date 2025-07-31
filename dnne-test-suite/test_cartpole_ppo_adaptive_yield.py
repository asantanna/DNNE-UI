#!/usr/bin/env python3
"""
Test Cartpole PPO with adaptive yielding enabled.
This tests the fixes for:
1. Import path issue in rl_games_dnne
2. Framework module availability during training
3. Adaptive yielding integration
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from export_system.graph_exporter import GraphExporter

def test_cartpole_ppo_export():
    """Test export and execution of Cartpole PPO workflow"""
    
    print("=" * 80)
    print("Testing Cartpole PPO with Adaptive Yielding")
    print("=" * 80)
    
    # Path to the workflow
    workflow_path = current_dir / "user/default/workflows/Cartpole_PPO.json"
    if not workflow_path.exists():
        print(f"❌ Workflow not found: {workflow_path}")
        return False
        
    # Export directory
    export_dir = current_dir / "export_system/exports/cartpole_ppo_test"
    
    # Clean up previous export
    if export_dir.exists():
        import shutil
        shutil.rmtree(export_dir)
    
    print(f"📁 Workflow: {workflow_path}")
    print(f"📁 Export to: {export_dir}")
    
    # Create exporter
    exporter = GraphExporter()
    
    try:
        # Load workflow data
        with open(workflow_path, 'r') as f:
            workflow_data = json.load(f)
            
        # Export the workflow
        print("\n🔄 Exporting workflow...")
        exporter.export_workflow(workflow_data, str(export_dir))
        print("✅ Export completed successfully")
        
        # Check that framework files exist
        framework_dir = export_dir / "framework"
        globals_file = framework_dir / "globals.py"
        
        if not globals_file.exists():
            print(f"❌ Framework globals.py not found at: {globals_file}")
            return False
        print(f"✅ Framework globals.py exists at: {globals_file}")
        
        # Check runner.py
        runner_file = export_dir / "runner.py"
        if not runner_file.exists():
            print(f"❌ runner.py not found at: {runner_file}")
            return False
            
        # Read runner.py to check for sys.path fix
        runner_content = runner_file.read_text()
        if "sys.path.insert(0, str(original_dir))" in runner_content:
            print("✅ runner.py contains sys.path fix for framework imports")
        else:
            print("⚠️  runner.py may not have sys.path fix (checking generated node files)")
            
        # Run the exported workflow with adaptive yielding
        print("\n🚀 Running exported workflow with adaptive yielding...")
        env = os.environ.copy()
        env["DNNE_ADAPTIVE_YIELD"] = "1"
        
        # Run for just a few steps to test
        cmd = [
            sys.executable,
            str(runner_file),
            "--headless",
            "--verbose"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        print(f"DNNE_ADAPTIVE_YIELD={env.get('DNNE_ADAPTIVE_YIELD', '0')}")
        
        # Start the process - run from the current directory
        process = subprocess.Popen(
            cmd,
            cwd=str(current_dir),  # Run from DNNE-UI directory
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Collect output for a few seconds
        import time
        start_time = time.time()
        output_lines = []
        error_lines = []
        
        while time.time() - start_time < 10:  # Run for 10 seconds
            # Check if process has output
            line = process.stdout.readline()
            if line:
                print(line.rstrip())
                output_lines.append(line)
                
                # Check for successful imports
                if "Adaptive yielding enabled" in line:
                    print("✅ Adaptive yielding successfully initialized")
                if "Added .* to sys.path for framework imports" in line:
                    print("✅ sys.path fix applied successfully")
                if "PPO training with IsaacGymEnvs" in line:
                    print("✅ PPO training started successfully")
                    
            # Check for errors
            error = process.stderr.readline() 
            if error:
                print(f"ERROR: {error.rstrip()}")
                error_lines.append(error)
                
                # Check for import errors
                if "cannot import framework.globals" in error:
                    print("❌ Framework import error - sys.path fix not working")
                    process.terminate()
                    return False
                    
        # Terminate the process
        process.terminate()
        process.wait()
        
        # Check results
        all_output = "".join(output_lines)
        all_errors = "".join(error_lines)
        
        success = True
        if "ModuleNotFoundError" in all_errors or "ImportError" in all_errors:
            print("\n❌ Import errors detected")
            success = False
        elif "Adaptive yielding enabled" in all_output:
            print("\n✅ Adaptive yielding working correctly")
        else:
            print("\n⚠️  Adaptive yielding status unclear")
            
        return success
        
    except Exception as e:
        print(f"\n❌ Export/execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Make sure we're in the conda environment
    if "DNNE_PY38" not in os.environ.get("CONDA_DEFAULT_ENV", ""):
        print("⚠️  Warning: Not in DNNE_PY38 conda environment")
        print("Run: source /home/asantanna/miniconda/bin/activate DNNE_PY38")
        
    success = test_cartpole_ppo_export()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)