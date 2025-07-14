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
    
    def profile_isaacgymenvs(self):
        """Profile IsaacGymEnvs using subprocess with cProfile"""
        print("\n🔬 Running IsaacGymEnvs profiling...")
        
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
                f'train.params.config.max_epochs={self.num_iterations}',
                'train.params.config.horizon_length=16',
                'train.params.config.minibatch_size=8192',
                'headless=True',
                'test=False',
                'dnne_profiling=True'  # Enable profiling for C++ timing
            ]
            
            print(f"  Running {self.num_iterations} iterations with {self.num_envs} environments...")
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
                    'num_iterations': self.num_iterations,
                    'num_envs': self.num_envs,
                    'total_time': total_time,
                    'step_count': self.num_iterations * 16,  # horizon_length
                    'steps_per_sec': (self.num_iterations * 16) / total_time if total_time > 0 else 0,
                    'prof_file': prof_file,
                    'status': 'success'
                }
                
                # Save metrics
                metrics_file = '/tmp/isaacgymenvs_metrics.json'
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
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
        
        # Calculate appropriate timeout for DNNE
        # DNNE is typically 2x slower, so use iterations * 6 seconds as estimate
        dnne_timeout = min(self.num_iterations * 6, self.timeout)
        
        cmd = [
            'python', '-m', 'cProfile',
            '-o', prof_file,
            str(runner_script),
            '--timeout', f'{dnne_timeout}s',
            '--dnne-profiling'  # Enable profiling for C++ timing
        ]
        
        print(f"  Running {self.num_iterations} iterations with {self.num_envs} environments...")
        print(f"  Timeout: {dnne_timeout}s")
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
                
                # Extract step count from output if available
                step_count = self.num_iterations * 16  # Default estimate
                if "Step count:" in result.stdout:
                    try:
                        for line in result.stdout.splitlines():
                            if "Step count:" in line:
                                step_count = int(line.split(":")[-1].strip())
                                break
                    except:
                        pass
                
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