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
    
    def __init__(self, num_envs=512, timeout=300, override_epochs=None, visual=False,
                 ppo_cycle_debug=False, stop_after_cycle=None, fixed_seed=None, capture_values=False,
                 enable_cpp_profiling=False):
        self.num_envs = num_envs
        self.timeout = timeout
        self.override_epochs = override_epochs
        self.visual = visual
        self.ppo_cycle_debug = ppo_cycle_debug
        self.stop_after_cycle = stop_after_cycle
        self.fixed_seed = fixed_seed
        self.capture_values = capture_values
        self.enable_cpp_profiling = enable_cpp_profiling
    
    def extract_max_epochs_from_workflow(self):
        """Extract max_epochs value from DNNE workflow JSON"""
        workflow_file = Path("/mnt/e/ALS-Projects/DNNE/DNNE-UI/user/default/workflows/Cartpole_PPO.json")
        if not workflow_file.exists():
            raise FileNotFoundError(f"Workflow file not found: {workflow_file}")
            
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
                        
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in workflow file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error extracting max_epochs from workflow: {e}")
            
        # If we got here, max_epochs was not found
        raise ValueError("max_epochs not found in workflow JSON. PPOTrainerNode must have max_epochs configured.")
    
    def profile_isaacgymenvs(self):
        """Profile IsaacGymEnvs using subprocess with cProfile"""
        print("\n🔬 Running IsaacGymEnvs profiling...")
        
        # Handle epochs override
        if self.override_epochs:
            iterations_to_run = self.override_epochs
            print(f"  📊 Using override epochs: {iterations_to_run}")
        else:
            # Extract max_epochs from workflow to match DNNE
            workflow_max_epochs = self.extract_max_epochs_from_workflow()
            if workflow_max_epochs:
                print(f"  📊 Found max_epochs={workflow_max_epochs} in DNNE workflow")
            
            # Use workflow max_epochs - no fallback
            if workflow_max_epochs:
                iterations_to_run = workflow_max_epochs
            else:
                raise ValueError("max_epochs not found in workflow JSON. This is required for PPO training.")
        
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
            
            # Set up environment variables for debugging
            env = os.environ.copy()
            
            # Check for user-set env vars and fail if found
            if 'USE_RL_GAMES_DEBUG' in os.environ:
                raise RuntimeError("USE_RL_GAMES_DEBUG is already set. Profiler must control this variable.")
            
            # Set the env var for child processes to use rl_games_debug
            env['USE_RL_GAMES_DEBUG'] = '1'
            print("  [DNNE_DEBUG] Setting USE_RL_GAMES_DEBUG=1 for IsaacGymEnvs")
            
            if self.ppo_cycle_debug:
                env['PPO_CYCLE_DEBUG'] = '1'
                if self.stop_after_cycle:
                    env['PPO_STOP_AFTER_CYCLE'] = '1'
            if self.fixed_seed is not None:
                env['FIXED_SEED'] = str(self.fixed_seed)
            
            cmd = [
                'python', '-m', 'cProfile',
                '-o', prof_file,
                'isaacgymenvs/train.py',
                'task=Cartpole',
                f'task.env.numEnvs={self.num_envs}',
                f'train.params.config.max_epochs={iterations_to_run}',
                'train.params.config.horizon_length=16',
                'train.params.config.minibatch_size=8192',
                f'headless={not self.visual}',  # Visual mode disables headless
                'test=False',
                f'dnne_cpp_profiling={self.enable_cpp_profiling}'  # Use enable_cpp_profiling flag
            ]
            
            # Add fixed seed to command if specified
            if self.fixed_seed is not None:
                cmd.append(f'seed={self.fixed_seed}')
            
            print(f"  Running {iterations_to_run} epochs with {self.num_envs} environments...")
            # Removed comparison with num_iterations since we no longer have it
            print(f"  Command: {' '.join(cmd[:5])}...")
            
            # Track time
            start_time = time.time()
            
            # Run with profiling
            try:
                # Save debug output
                debug_file = open('/tmp/isaacgymenvs_profile_debug.log', 'w')
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True,
                    timeout=self.timeout,
                    env=env  # Pass the environment with PPO debug flags
                )
                
                total_time = time.time() - start_time
                
                # Write debug info
                debug_file.write(f"Return code: {result.returncode}\n")
                debug_file.write(f"STDOUT:\n{result.stdout}\n")
                debug_file.write(f"STDERR:\n{result.stderr}\n")
                debug_file.close()
                
                if result.returncode == 0:
                    print(f"  ✅ Completed in {total_time:.2f}s")
                    
                    # Save PPO cycle debug output if enabled
                    if self.ppo_cycle_debug:
                        self._save_ppo_cycle_debug(result.stdout, 'isaacgym')
                    
                    # Extract episode returns from training output
                    episode_return_metrics = self._extract_episode_returns_from_output(result.stdout, result.stderr)
                
                    # Create basic metrics
                    metrics = {
                        'system': 'IsaacGymEnvs',
                        'num_epochs': iterations_to_run,
                        'num_envs': self.num_envs,
                        'total_time': total_time,
                        'step_count': iterations_to_run * self.num_envs * 16,  # epochs * envs * horizon_length
                        'steps_per_sec': (iterations_to_run * self.num_envs * 16) / total_time if total_time > 0 else 0,
                        'prof_file': prof_file,
                        'status': 'success',
                        'learning_metrics': episode_return_metrics
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
            # No directory renaming - we use environment variables now
    
    def _extract_episode_returns_from_output(self, stdout, stderr):
        """Extract episode return information from IsaacGymEnvs training output"""
        import re
        
        # Initialize metrics
        episode_metrics = {
            "total_episodes": 0,
            "completed_episodes": 0,
            "average_episode_return": 0.0,
            "last_n_episodes": 100,
            "episode_returns": [],
            "source": "isaacgymenvs_output_parsing",
            "data_available": False
        }
        
        # Look for episode return patterns in the output
        output_text = stdout + "\n" + stderr
        
        # Try to find episode reward/return information
        # IsaacGymEnvs outputs rewards in checkpoint filenames and training logs
        found_rewards = []
        episode_counts = []
        
        # Pattern 1: Checkpoint filenames with rewards (e.g., "ep_25_rew_162.62842.pth" or "rew__279.9_.pth")
        checkpoint_patterns = [
            r'ep_(\d+)_rew_([0-9.-]+)\.pth',  # Pattern: ep_25_rew_162.62842.pth
            r'rew__([0-9.-]+)_\.pth',          # Pattern: rew__279.9_.pth
            r'rew_([0-9.-]+)\.pth'             # Pattern: rew_279.9.pth
        ]
        
        for pattern in checkpoint_patterns:
            if 'ep_' in pattern:
                # Extract both episode number and reward
                matches = re.findall(pattern, output_text)
                for match in matches:
                    try:
                        ep_num = int(match[0])
                        reward = float(match[1])
                        if -1000 < reward < 1000:
                            found_rewards.append(reward)
                            episode_counts.append(ep_num)
                    except (ValueError, IndexError):
                        continue
            else:
                # Just extract reward
                matches = re.findall(pattern, output_text)
                for match in matches:
                    try:
                        reward = float(match)
                        if -1000 < reward < 1000:
                            found_rewards.append(reward)
                    except ValueError:
                        continue
        
        # Pattern 2: Look for any other reward reporting patterns
        general_patterns = [
            r'mean_reward[:\s]+([0-9.-]+)',
            r'episode_reward[:\s]+([0-9.-]+)',
            r'ep_rew_mean[:\s]+([0-9.-]+)',
        ]
        
        for pattern in general_patterns:
            matches = re.findall(pattern, output_text, re.IGNORECASE)
            for match in matches:
                try:
                    reward = float(match)
                    if -1000 < reward < 1000 and reward not in found_rewards:
                        found_rewards.append(reward)
                except ValueError:
                    continue
        
        if found_rewards:
            # Use actual extracted reward values
            episode_metrics["episode_returns"] = found_rewards[-100:]  # Last 100 values
            episode_metrics["completed_episodes"] = len(episode_metrics["episode_returns"])
            episode_metrics["average_episode_return"] = found_rewards[-1] if found_rewards else 0.0  # Use most recent
            
            # Estimate total episodes based on training progress
            # IsaacGymEnvs checkpoints show progression, so use the highest episode count if available
            if episode_counts:
                episode_metrics["total_episodes"] = max(episode_counts) * self.num_envs  # Scale by num environments
            else:
                # Estimate based on steps and typical episode length
                # Cartpole episodes typically last 50-200 steps when learning
                estimated_episodes = int(self.num_envs * 40 * 16 / 100)  # Conservative estimate
                episode_metrics["total_episodes"] = estimated_episodes
                
            episode_metrics["data_available"] = True
            print(f"  📊 Extracted {len(found_rewards)} episode checkpoint rewards from IsaacGymEnvs")
        else:
            # No real data available - report this clearly
            episode_metrics["data_available"] = False
            print(f"  📊 No episode returns found in IsaacGymEnvs output - learning metrics unavailable")
        
        return episode_metrics
    
    def _extract_dnne_episode_returns(self, stdout, stderr):
        """Extract episode return information from DNNE training output"""
        import re
        
        # Initialize metrics
        episode_metrics = {
            "total_episodes": 0,
            "completed_episodes": 0,
            "average_episode_return": 0.0,
            "last_n_episodes": 100,
            "episode_returns": [],
            "source": "dnne_output_parsing",
            "data_available": False
        }
        
        # Look for episode return patterns in DNNE output
        output_text = stdout + "\n" + stderr
        
        # DNNE logs episode returns in the format:
        # "avg episode return = 234.5" and "Episode 1: episode return = 234.5"
        episode_patterns = [
            r'avg episode return[:\s=]+([0-9.-]+)',
            r'average episode return[:\s=]+([0-9.-]+)',
            r'Episode \d+: episode return[:\s=]+([0-9.-]+)',  # Individual episodes
            r'episode return[:\s=]+([0-9.-]+)',
        ]
        
        found_returns = []
        for pattern in episode_patterns:
            matches = re.findall(pattern, output_text, re.IGNORECASE)
            for match in matches:
                try:
                    episode_return = float(match)
                    # Filter reasonable return values (Cartpole typically -500 to +500)
                    if -1000 < episode_return < 1000:
                        found_returns.append(episode_return)
                except ValueError:
                    continue
        
        if found_returns:
            # Use actual extracted episode return values
            episode_metrics["episode_returns"] = found_returns[-100:]  # Last 100 values
            episode_metrics["completed_episodes"] = len(episode_metrics["episode_returns"])
            episode_metrics["average_episode_return"] = found_returns[-1]  # Most recent average
            episode_metrics["total_episodes"] = len(found_returns)
            episode_metrics["data_available"] = True
            print(f"  📊 Extracted {len(found_returns)} episode return measurements from DNNE output")
        else:
            # No real data available - report this clearly
            episode_metrics["data_available"] = False
            print(f"  📊 No episode returns found in DNNE output - learning metrics unavailable")
        
        return episode_metrics
    
    def _save_ppo_cycle_debug(self, output, system_name):
        """Save PPO cycle debug output for analysis"""
        import re
        
        debug_data = {
            'system': system_name,
            'ppo_cycle_logs': [],
            'actions': [],
            'values': [],
            'rewards': [],
            'observations': [],
            'initial_state': {},
            'batch_info': {},
            'gradient_info': {},
            'loss_components': {},
            'policy_params': {}
        }
        
        # Extract all DNNE_DEBUG lines
        for line in output.splitlines():
            if '[DNNE_DEBUG]' in line:
                debug_data['ppo_cycle_logs'].append(line)
                
                # Extract PPO_CYCLE step data
                if 'PPO_CYCLE: Step' in line:
                    match = re.search(r'Step (\d+): action=([-\d.]+), value=([-\d.]+), reward=([-\d.]+)', line)
                    if match:
                        step, action, value, reward = match.groups()
                        debug_data['actions'].append(float(action))
                        debug_data['values'].append(float(value))
                        debug_data['rewards'].append(float(reward))
                
                # Extract initial state information
                elif 'PPO_INITIAL:' in line:
                    if 'First observation:' in line:
                        match = re.search(r'First observation: \[([-\d., ]+)\]', line)
                        if match:
                            debug_data['initial_state']['first_obs'] = [float(x) for x in match.group(1).split(', ')]
                    elif 'Observation shape:' in line:
                        match = re.search(r'Observation shape: torch.Size\(\[([\d, ]+)\]\)', line)
                        if match:
                            debug_data['initial_state']['obs_shape'] = [int(x) for x in match.group(1).split(', ')]
                    elif 'Obs normalization - mean:' in line:
                        match = re.search(r'mean: \[([-\d., ]+)\]', line)
                        if match:
                            debug_data['initial_state']['obs_norm_mean'] = [float(x) for x in match.group(1).split(', ')]
                    elif 'Obs normalization - var:' in line:
                        match = re.search(r'var: \[([-\d., ]+)\]', line)
                        if match:
                            debug_data['initial_state']['obs_norm_var'] = [float(x) for x in match.group(1).split(', ')]
                    elif 'Actor first layer weights:' in line:
                        match = re.search(r'weights: \[([-\d., ]+)\]', line)
                        if match:
                            debug_data['initial_state']['actor_weights'] = [float(x) for x in match.group(1).split(', ')]
                    elif 'Mu layer weights:' in line:
                        match = re.search(r'weights: \[([-\d., ]+)\]', line)
                        if match:
                            debug_data['initial_state']['mu_weights'] = [float(x) for x in match.group(1).split(', ')]
                    elif 'Sigma vals:' in line:
                        match = re.search(r'Sigma vals: \[([-\d., ]+)\]', line)
                        if match:
                            debug_data['initial_state']['sigma_vals'] = [float(x) for x in match.group(1).split(', ')]
                
                # Extract batch information
                elif 'PPO_BATCH:' in line:
                    if 'Advantages' in line and 'mean:' in line:
                        match = re.search(r'mean: ([-\d.]+), std: ([-\d.]+)', line)
                        if match:
                            debug_data['batch_info']['advantages_mean'] = float(match.group(1))
                            debug_data['batch_info']['advantages_std'] = float(match.group(2))
                    elif 'Returns' in line and 'mean:' in line:
                        match = re.search(r'mean: ([-\d.]+), std: ([-\d.]+)', line)
                        if match:
                            debug_data['batch_info']['returns_mean'] = float(match.group(1))
                            debug_data['batch_info']['returns_std'] = float(match.group(2))
                    elif 'Values' in line and 'mean:' in line:
                        match = re.search(r'mean: ([-\d.]+), std: ([-\d.]+)', line)
                        if match:
                            debug_data['batch_info']['values_mean'] = float(match.group(1))
                            debug_data['batch_info']['values_std'] = float(match.group(2))
                
                # Extract gradient information
                elif 'PPO_GRAD:' in line:
                    if 'Actor loss:' in line:
                        match = re.search(r'Actor loss: ([-\d.]+)', line)
                        if match:
                            debug_data['loss_components']['actor_loss'] = float(match.group(1))
                    elif 'Critic loss:' in line:
                        match = re.search(r'Critic loss: ([-\d.]+)', line)
                        if match:
                            debug_data['loss_components']['critic_loss'] = float(match.group(1))
                    elif 'Entropy:' in line:
                        match = re.search(r'Entropy: ([-\d.]+)', line)
                        if match:
                            debug_data['loss_components']['entropy'] = float(match.group(1))
                    elif 'Total loss:' in line:
                        match = re.search(r'Total loss: ([-\d.]+)', line)
                        if match:
                            debug_data['loss_components']['total_loss'] = float(match.group(1))
                    elif 'KL divergence:' in line:
                        match = re.search(r'KL divergence: ([-\d.]+)', line)
                        if match:
                            debug_data['gradient_info']['kl_divergence'] = float(match.group(1))
                    elif 'Mu shape:' in line and 'mean:' in line:
                        match = re.search(r'mean: ([-\d.]+), std: ([-\d.]+)', line)
                        if match:
                            debug_data['policy_params']['mu_mean'] = float(match.group(1))
                            debug_data['policy_params']['mu_std'] = float(match.group(2))
                    elif 'Sigma shape:' in line and 'mean:' in line:
                        match = re.search(r'mean: ([-\d.]+), std: ([-\d.]+)', line)
                        if match:
                            debug_data['policy_params']['sigma_mean'] = float(match.group(1))
                            debug_data['policy_params']['sigma_std'] = float(match.group(2))
        
        # Save to JSON file
        output_file = f'/tmp/{system_name}_ppo_cycle_debug.json'
        with open(output_file, 'w') as f:
            json.dump(debug_data, f, indent=2)
        
        print(f"  📊 Saved PPO cycle debug data to {output_file}")
        if debug_data['actions']:
            print(f"     Captured {len(debug_data['actions'])} environment steps")
        if debug_data['initial_state']:
            print(f"     Captured initial state information")
        if debug_data['loss_components']:
            print(f"     Captured loss components and gradients")
    
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
        
        # Always extract workflow value to know the default
        workflow_max_epochs = self.extract_max_epochs_from_workflow()
        
        # Handle epochs override
        if self.override_epochs:
            expected_iterations = self.override_epochs
            print(f"  📊 Using override epochs: {expected_iterations}")
        else:
            expected_iterations = workflow_max_epochs
            print(f"  📊 Found max_epochs={workflow_max_epochs} in DNNE workflow")
        
        # Calculate appropriate timeout for DNNE
        # Give it plenty of time since it should stop on its own
        dnne_timeout = min(expected_iterations * 6, self.timeout)
        
        # Set up environment variables for PPO cycle debugging
        env = os.environ.copy()
        env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For deterministic behavior
        if self.ppo_cycle_debug:
            env['PPO_CYCLE_DEBUG'] = '1'
            print(f"  📊 PPO_CYCLE_DEBUG enabled")
            if self.stop_after_cycle:
                env['PPO_STOP_AFTER_CYCLE'] = str(self.stop_after_cycle)
                # Override epochs to 1 for stop after cycle
                expected_iterations = 1
                print(f"  📊 Stop after {self.stop_after_cycle} cycle(s) enabled - overriding to 1 epoch")
        if self.fixed_seed is not None:
            env['FIXED_SEED'] = str(self.fixed_seed)
            print(f"  📊 Fixed seed: {self.fixed_seed}")
        
        cmd = [
            'python', '-m', 'cProfile',
            '-o', prof_file,
            str(runner_script),
            '--timeout', f'{dnne_timeout}s',  # This is just a safety timeout
            '--dnne-profiling'  # Enable profiling for C++ timing
            # Note: --verbose removed as episode returns are now printed without it
        ]
        
        # Add visual mode if enabled
        if self.visual:
            cmd.append('--visual')
        
        # Add epochs override if specified
        if self.override_epochs:
            cmd.extend(['--epochs', str(self.override_epochs)])
        
        # Add fixed seed if specified
        if self.fixed_seed is not None:
            cmd.extend(['--fixed-seed', str(self.fixed_seed)])
        
        print(f"  Running {expected_iterations} epochs with {self.num_envs} environments...")
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
                timeout=self.timeout,
                env=env  # Pass the environment with PPO debug flags
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
                
                # Save PPO cycle debug output if enabled
                if self.ppo_cycle_debug:
                    # PPO_CYCLE logs should now be in stdout since we use print()
                    self._save_ppo_cycle_debug(result.stdout, 'dnne')
                
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
                                # Node 9 is the IsaacGymStepNode that does the actual stepping
                                if node_id == "9":
                                    env_computations = int(comp_part)
                                    break
                        except:
                            pass
                
                # If we found environment computations, calculate total steps
                if env_computations > 0:
                    step_count = env_computations * self.num_envs
                    # Verify it's close to expected
                    expected_steps = expected_iterations * self.num_envs * 16
                    if abs(step_count - expected_steps) > self.num_envs * 10:
                        print(f"  ⚠️  Warning: Expected ~{expected_steps//self.num_envs} env steps, got {env_computations}")
                
                # Extract episode returns from DNNE output
                dnne_episode_metrics = self._extract_dnne_episode_returns(result.stdout, result.stderr)
                
                # Create basic metrics
                metrics = {
                    'system': 'DNNE',
                    'num_epochs': expected_iterations,
                    'num_envs': self.num_envs,
                    'total_time': total_time,
                    'step_count': step_count,
                    'steps_per_sec': step_count / total_time if total_time > 0 else 0,
                    'prof_file': prof_file,
                    'status': 'success',
                    'learning_metrics': dnne_episode_metrics
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