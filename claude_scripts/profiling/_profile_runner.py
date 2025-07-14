#!/usr/bin/env python3
"""
Profile Runner - Executes profiling for IsaacGymEnvs and DNNE

Uses python -m cProfile to profile each system externally.
Saves profile data and basic metrics to /tmp/.
"""

import subprocess
import time
import json
import os
from pathlib import Path

class ProfileRunner:
    """Runs profiling for both systems using external cProfile"""
    
    def __init__(self, num_iterations=10, num_envs=512, timeout=300):
        self.num_iterations = num_iterations
        self.num_envs = num_envs
        self.timeout = timeout
    
    def extract_max_epochs_from_workflow(self):
        """Extract max_epochs value from DNNE workflow JSON"""
        workflow_file = Path("/mnt/e/ALS-Projects/DNNE/DNNE-UI/user/default/workflows/Cartpole_PPO.json")
        if not workflow_file.exists():
            return None
            
        try:
            with open(workflow_file, 'r') as f:
                workflow_data = json.load(f)
                
            # Find PPOTrainerNode in the nodes array
            nodes = workflow_data.get('nodes', [])
            for node in nodes:
                if node.get('type') == 'PPOTrainerNode':
                    # Extract widgets_values - first value is max_epochs
                    widgets_values = node.get('widgets_values', [])
                    if widgets_values and len(widgets_values) > 0:
                        return int(widgets_values[0])
                        
        except Exception as e:
            print(f"  ⚠️ Could not extract max_epochs from workflow: {e}")
            return None
        return None
    
    def profile_isaacgymenvs(self):
        """Profile IsaacGymEnvs using subprocess with cProfile"""
        print("\n🔬 Running IsaacGymEnvs profiling...")
        
        # Extract max_epochs from workflow to match DNNE
        workflow_max_epochs = self.extract_max_epochs_from_workflow()
        if workflow_max_epochs:
            print(f"  📊 Found max_epochs={workflow_max_epochs} in DNNE workflow")
        
        # Use workflow max_epochs if available, otherwise use num_iterations
        iterations_to_run = workflow_max_epochs if workflow_max_epochs else self.num_iterations
        
        # Change to IsaacGymEnvs directory
        original_dir = os.getcwd()
        isaac_dir = '/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs'
        
        if not Path(isaac_dir).exists():
            print(f"❌ IsaacGymEnvs directory not found: {isaac_dir}")
            return None
        
        os.chdir(isaac_dir)
        
        try:
            # Build the command to run IsaacGymEnvs directly with cProfile
            prof_file = '/tmp/isaacgymenvs_training.prof'
            
            cmd = [
                'python', '-m', 'cProfile',
                '-o', prof_file,
                'isaacgymenvs/train.py',
                'task=Cartpole',
                f'task.env.numEnvs={self.num_envs}',
                f'train.params.config.max_epochs={iterations_to_run}',
                'train.params.config.horizon_length=16',
                'train.params.config.minibatch_size=8192',
                'headless=True',
                'test=False',
                'dnne_profiling=True'  # Enable profiling for C++ timing
            ]
            
            print(f"  Running {iterations_to_run} iterations with {self.num_envs} environments...")
            if workflow_max_epochs and iterations_to_run != self.num_iterations:
                print(f"  (Matched to DNNE workflow instead of default {self.num_iterations})")
            print(f"  Command: {' '.join(cmd[:5])}...")
            
            # Track time
            start_time = time.time()
            
            # Run with profiling
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=self.timeout
            )
            
            total_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"  ✅ Completed in {total_time:.2f}s")
                
                # Create basic metrics
                metrics = {
                    'system': 'IsaacGymEnvs',
                    'num_iterations': iterations_to_run,
                    'num_envs': self.num_envs,
                    'total_time': total_time,
                    'step_count': iterations_to_run * self.num_envs * 16,  # epochs * envs * horizon_length
                    'steps_per_sec': (iterations_to_run * self.num_envs * 16) / total_time if total_time > 0 else 0,
                    'prof_file': prof_file,
                    'status': 'success'
                }
                
                # Save metrics
                metrics_file = '/tmp/isaacgymenvs_metrics.json'
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                # Try to trigger timing data save by importing and checking
                try:
                    import gc
                    import sys
                    # Add IsaacGymEnvs to path
                    if isaac_dir not in sys.path:
                        sys.path.insert(0, isaac_dir)
                    
                    # Look for VecTask instances with timing data
                    for obj in gc.get_objects():
                        if hasattr(obj, 'save_timing_data') and hasattr(obj, 'timing_data'):
                            try:
                                obj.save_timing_data()
                                print("  📊 Saved C++ timing data")
                                break
                            except:
                                pass
                except:
                    pass
                
                print(f"  📊 Steps/sec: {metrics['steps_per_sec']:.1f}")
                return metrics
                
            else:
                print(f"  ❌ Failed with return code {result.returncode}")
                if result.stderr:
                    print(f"  Error: {result.stderr[:500]}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"  ❌ Timed out after {self.timeout}s")
            return None
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            os.chdir(original_dir)
    
    def profile_dnne(self):
        """Profile DNNE using subprocess with cProfile"""
        print("\n🔬 Running DNNE profiling...")
        
        export_dir = Path("/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO")
        if not export_dir.exists():
            print(f"  ❌ DNNE export not found at: {export_dir}")
            print("  Please export the Cartpole_PPO workflow first")
            return None
        
        # Build the command to run DNNE directly with cProfile
        prof_file = '/tmp/dnne_training.prof'
        runner_script = export_dir / 'runner.py'
        
        # Extract max_epochs from workflow to show expected iterations
        workflow_max_epochs = self.extract_max_epochs_from_workflow()
        expected_iterations = workflow_max_epochs if workflow_max_epochs else self.num_iterations
        
        # Calculate appropriate timeout for DNNE
        # Give it plenty of time since it should stop on its own
        dnne_timeout = min(expected_iterations * 6, self.timeout)
        
        cmd = [
            'python', '-m', 'cProfile',
            '-o', prof_file,
            str(runner_script),
            '--timeout', f'{dnne_timeout}s',  # This is just a safety timeout
            '--dnne-profiling'  # Enable profiling for C++ timing
        ]
        
        print(f"  Running to match {expected_iterations} epochs with {self.num_envs} environments...")
        print(f"  Timeout: {dnne_timeout}s (safety timeout - DNNE should stop at {expected_iterations} epochs)")
        print(f"  Command: {' '.join(cmd[:5])}...")
        
        # Track time
        start_time = time.time()
        
        # Run with profiling
        try:
            # Save debug output
            debug_file = open('/tmp/dnne_profile_debug.log', 'w')
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(export_dir),
                timeout=self.timeout
            )
            
            total_time = time.time() - start_time
            
            # Write debug info
            debug_file.write(f"Return code: {result.returncode}\n")
            debug_file.write(f"STDOUT:\n{result.stdout}\n")
            debug_file.write(f"STDERR:\n{result.stderr}\n")
            debug_file.close()
            
            # Check for success (DNNE may exit with timeout which is OK)
            if result.returncode == 0 or "Completed" in result.stdout or Path(prof_file).exists():
                print(f"  ✅ Completed in {total_time:.2f}s")
                
                # Extract step count from output
                # Look for environment node computations (e.g., "7: 86 computations")
                step_count = expected_iterations * self.num_envs * 16  # Default estimate
                env_computations = 0
                
                for line in result.stdout.splitlines():
                    if ": " in line and "computations" in line:
                        try:
                            parts = line.strip().split(":")
                            if len(parts) >= 2:
                                comp_part = parts[1].strip().split()[0]
                                node_id = parts[0].strip()
                                # Node 7 is typically the environment node
                                if node_id == "7":
                                    env_computations = int(comp_part)
                                    break
                        except:
                            pass
                
                # If we found environment computations, calculate total steps
                if env_computations > 0:
                    step_count = env_computations * self.num_envs
                    # Verify it's close to expected (5 epochs = 80 env steps = 40,960 total)
                    expected_steps = expected_iterations * self.num_envs * 16
                    if abs(step_count - expected_steps) > self.num_envs * 10:
                        print(f"  ⚠️  Warning: Expected ~{expected_steps//self.num_envs} env steps, got {env_computations}")
                
                # Create basic metrics
                metrics = {
                    'system': 'DNNE',
                    'num_iterations': self.num_iterations,
                    'num_envs': self.num_envs,
                    'total_time': total_time,
                    'step_count': step_count,
                    'steps_per_sec': step_count / total_time if total_time > 0 else 0,
                    'prof_file': prof_file,
                    'status': 'success'
                }
                
                # Save metrics
                metrics_file = '/tmp/dnne_metrics.json'
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                print(f"  📊 Steps/sec: {metrics['steps_per_sec']:.1f}")
                return metrics
                
            else:
                print(f"  ❌ Failed with return code {result.returncode}")
                print(f"  Check /tmp/dnne_profile_debug.log for details")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"  ❌ Timed out after {self.timeout}s")
            return None
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None