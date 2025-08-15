#!/usr/bin/env python3
"""
Standalone debug version of the failing MNIST_Test.
Removes pytest dependencies to make it easier to debug.
"""

import sys
import subprocess
import time
import re
from pathlib import Path
import shutil

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def validate_export_output(export_path: Path) -> bool:
    """Validate that export output has the expected structure."""
    if not export_path.exists():
        return False
    
    # Check for required files
    required_files = ["runner.py", "__init__.py"]
    for file_name in required_files:
        if not (export_path / file_name).exists():
            return False
    
    # Check for framework and nodes directories
    required_dirs = ["framework", "nodes"]
    for dir_name in required_dirs:
        if not (export_path / dir_name).is_dir():
            return False
    
    return True

def cleanup_export_dir(export_path: Path):
    """Clean up temporary export directory."""
    if export_path.exists() and export_path.is_dir():
        try:
            shutil.rmtree(export_path)
        except OSError:
            print(f"Warning: Could not fully clean up {export_path}")

def export_workflow_for_test(workflow_name: str, test_name: str = None) -> Path:
    """Export a workflow using the standardized programmatic export utility for testing."""
    
    # Generate test directory name
    if test_name:
        target_dir = f"test_{test_name}"
    else:
        target_dir = f"test_{workflow_name.lower().replace(' ', '_')}"
    
    # Get the path to the programmatic export utility
    export_script = project_root / "dnne_test_suite" / "utilities" / "programmatic_export.py"
    
    # Run the export utility
    try:
        result = subprocess.run(
            [sys.executable, str(export_script), workflow_name, "--target-dir", target_dir],
            capture_output=True,
            text=True,
            timeout=60,  # 1 minute timeout
            cwd=project_root
        )
        
        if result.returncode != 0:
            print(f"Export STDOUT: {result.stdout}")
            print(f"Export STDERR: {result.stderr}")
            raise RuntimeError(f"Export failed with return code {result.returncode}")
        
        export_path = project_root / "export_system" / "exports" / target_dir
        
        # Validate the export was successful
        if not validate_export_output(export_path):
            raise RuntimeError(f"Export validation failed for {export_path}")
        
        return export_path
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Export timed out for workflow: {workflow_name}")
    except Exception as e:
        raise RuntimeError(f"Export failed for workflow {workflow_name}: {e}")

def main():
    """Run the standalone MNIST_Test."""
    print("🧪 Running standalone MNIST export and execution test")
    
    workflow_name = "MNIST_Test"
    export_path = None
    
    try:
        print(f"📁 Exporting workflow: {workflow_name}")
        
        # Use standardized export utility
        export_path = export_workflow_for_test(workflow_name, "debug_standalone")
        print(f"✅ Export completed: {export_path}")
        
        # Show what files were generated
        print("📂 Generated files:")
        for file_path in export_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(export_path)
                print(f"   {rel_path}")
        
        # Export utility already validates output, but double-check
        if validate_export_output(export_path):
            runner_file = export_path / "runner.py"
            print(f"📄 Runner file: {runner_file}")
            
            if runner_file.exists():
                print("🚀 Executing generated script...")
                
                # Track test execution time
                test_start_time = time.time()
                
                # Run with timeout to prevent hanging
                execution_result = subprocess.run(
                    [sys.executable, str(runner_file), "--test-mode"],
                    capture_output=True,
                    text=True,
                    timeout=60,  # 1 minute timeout for debug
                    cwd=export_path  # Run in export directory
                )
                
                test_end_time = time.time()
                test_duration = test_end_time - test_start_time
                
                print(f"⏱️  Execution completed in {test_duration:.1f} seconds")
                print(f"📊 Return code: {execution_result.returncode}")
                
                # Show output regardless of success/failure
                if execution_result.stdout:
                    print("\n📝 STDOUT:")
                    print(execution_result.stdout)
                
                if execution_result.stderr:
                    print("\n❌ STDERR:")
                    print(execution_result.stderr)
                
                # Check execution results
                if execution_result.returncode == 0:
                    print("✅ Execution successful!")
                else:
                    print(f"❌ Execution failed with return code: {execution_result.returncode}")
                    
            else:
                print(f"❌ Runner file not found: {runner_file}")
        else:
            print(f"❌ Export validation failed for: {export_path}")
            
    except Exception as e:
        print(f"💥 Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if export_path:
            print(f"🧹 Cleaning up: {export_path}")
            cleanup_export_dir(export_path)

if __name__ == "__main__":
    main()