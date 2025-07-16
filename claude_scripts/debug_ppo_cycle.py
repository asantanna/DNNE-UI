#!/usr/bin/env python3
"""
Debug PPO implementation following docs-dnne/for_claude/debug_strategy_for_ppo.md

Runs both DNNE and IsaacGymEnvs with:
- Fixed seed for deterministic execution
- Debug logging for each phase of the algorithm
- Stops after N PPO cycles for comparison
"""

import subprocess
import sys
import os
import time
import argparse
import json

# Conceptual blocks of PPO algorithm
PPO_PHASES = {
    "1_global_init": "Global initialization",
    "2_ppo_init": "PPO algorithm initialization", 
    "3_episode_init": "Per episode initialization",
    "3.1_reset_env": "Reset environment",
    "3.2_prepare_ppo": "Prepare PPO for episode",
    "4_learning_cycle": "Learning cycle",
    "4.1_get_obs": "Get environment observation",
    "4.2_ppo_forward": "PPO forward pass (policy)",
    "4.3_step_env": "Send action to env and step",
    "4.4_ppo_update": "PPO update (training)"
}

class PPODebugger:
    def __init__(self, seed=42, debug_cycles=1):
        self.seed = seed
        self.debug_cycles = debug_cycles
        self.dnne_logs = []
        self.isaacgym_logs = []
        
    def run_dnne_debug(self):
        """Run DNNE with debug logging"""
        print("\n" + "="*80)
        print(f"Running DNNE with seed {self.seed}, stopping after {self.debug_cycles} PPO cycles")
        print("="*80)
        
        # First export the workflow
        os.chdir("/mnt/e/ALS-Projects/DNNE/DNNE-UI")
        export_cmd = [sys.executable, "claude_scripts/programmatic_export.py", "Cartpole_PPO"]
        subprocess.run(export_cmd, check=True, capture_output=True)
        
        # Add debug logging to key nodes
        self._add_dnne_debug_logging()
        
        # Run with debug flags
        env = os.environ.copy()
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["PPO_DEBUG"] = "1"
        env["DEBUG_CYCLE_STOP"] = str(self.debug_cycles)
        
        cmd = [
            sys.executable,
            "export_system/exports/Cartpole_PPO/runner.py",
            f"--fixed-seed={self.seed}",
            "--epochs=1",
            "--headless",
            "--verbose"
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 universal_newlines=True, bufsize=1, env=env)
        
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            if line:
                output_lines.append(line.rstrip())
                if "[PPO Debug]" in line or "Episode" in line:
                    print(f"DNNE: {line.rstrip()}")
                    self.dnne_logs.append(line.rstrip())
        
        process.wait()
        return output_lines
    
    def run_isaacgym_debug(self):
        """Run IsaacGymEnvs with debug logging"""
        print("\n" + "="*80)
        print(f"Running IsaacGymEnvs with seed {self.seed}, stopping after {self.debug_cycles} PPO cycles")
        print("="*80)
        
        # Create debug runner for IsaacGymEnvs
        debug_runner = '''
import sys
sys.path.append('/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs')
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import numpy as np
import random

# Set seed
seed = {seed}
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Import after seed
import hydra
from omegaconf import DictConfig, OmegaConf
from isaacgymenvs import isaacgymenvs

# Add debug hooks
original_reset = None
original_step = None
ppo_cycle_count = 0
debug_cycle_stop = {debug_cycles}

def debug_reset(self, env_ids):
    print("[PPO Debug] [3.1_reset_env] Resetting environments")
    result = original_reset(env_ids)
    return result

def debug_step(self, actions):
    global ppo_cycle_count
    print("[PPO Debug] [4.3_step_env] Stepping environment")
    obs, rewards, dones, info = original_step(actions)
    
    # Check if PPO update would happen
    if hasattr(self, 'vec_env') and self.vec_env.obs_dict.get('states', None) is not None:
        buffer_size = len(self.vec_env.obs_dict['states'])
        if buffer_size >= 16:  # horizon length
            ppo_cycle_count += 1
            print(f"[PPO Debug] [4.4_ppo_update] PPO update cycle {{ppo_cycle_count}}")
            if ppo_cycle_count >= debug_cycle_stop:
                print(f"[PPO Debug] Stopping after {{ppo_cycle_count}} PPO cycles")
                sys.exit(0)
    
    return obs, rewards, dones, info

# Run with minimal config
cfg = OmegaConf.create({{
    'task': {{'name': 'Cartpole'}},
    'train': {{
        'params': {{
            'algo': {{'name': 'a2c_continuous'}},
            'model': {{'name': 'continuous_a2c_logstd'}},
            'config': {{
                'name': 'Cartpole',
                'num_actors': 512,
                'horizon_length': 16,
                'minibatch_size': 8192,
                'mini_epochs': 8,
                'gamma': 0.99,
                'tau': 0.95,
                'e_clip': 0.2,
                'learning_rate': 3e-4,
                'lr_schedule': 'adaptive',
                'kl_threshold': 0.008,
                'truncate_grads': True,
                'grad_norm': 1.0,
                'clip_value': True,
                'seq_len': 4,
                'bounds_loss_coef': 0.0001,
                'max_epochs': 1
            }}
        }}
    }},
    'task_name': 'Cartpole',
    'experiment': 'debug',
    'num_envs': 512,
    'seed': seed,
    'torch_deterministic': True,
    'physics_engine': 'physx',
    'sim_device': 'cuda:0',
    'rl_device': 'cuda:0',
    'graphics_device_id': 0,
    'headless': True
}})

print("[PPO Debug] [1_global_init] Starting IsaacGymEnvs")
print("[PPO Debug] [2_ppo_init] Initializing PPO algorithm")

# Run training
isaacgymenvs.train(cfg)
'''
        
        # Write debug runner
        runner_path = "/tmp/isaacgym_debug_runner.py"
        with open(runner_path, 'w') as f:
            f.write(debug_runner.format(seed=self.seed, debug_cycles=self.debug_cycles))
        
        env = os.environ.copy()
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        
        cmd = [sys.executable, runner_path]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 universal_newlines=True, bufsize=1, env=env)
        
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            if line:
                output_lines.append(line.rstrip())
                if "[PPO Debug]" in line:
                    print(f"IsaacGymEnvs: {line.rstrip()}")
                    self.isaacgym_logs.append(line.rstrip())
        
        process.wait()
        return output_lines
    
    def _add_dnne_debug_logging(self):
        """Add debug logging to DNNE nodes"""
        # Add to PPOAgentNode
        agent_file = "export_system/exports/Cartpole_PPO/nodes/ppoagentnode_3.py"
        with open(agent_file, 'r') as f:
            content = f.read()
        
        if "PPO_DEBUG" not in content:
            # Add debug check at start of compute
            debug_code = '''
        # Debug logging for PPO phases
        if os.environ.get("PPO_DEBUG"):
            self.logger.info("[PPO Debug] [4.2_ppo_forward] PPO forward pass starting")
'''
            content = content.replace('async def compute(self, observations) -> Dict[str, Any]:\n        """', 
                                    'async def compute(self, observations) -> Dict[str, Any]:\n        """' + debug_code)
            
            with open(agent_file, 'w') as f:
                f.write(content)
        
        # Add to PPOTrainerNode
        trainer_file = "export_system/exports/Cartpole_PPO/nodes/ppotrainernode_6.py"
        with open(trainer_file, 'r') as f:
            content = f.read()
            
        if "PPO_DEBUG" not in content:
            # Add cycle counter and debug logging
            content = content.replace('self.step_count = 0', 
                                    'self.step_count = 0\n        self.ppo_cycle_count = 0')
            
            # Add debug logging in compute
            debug_code = '''
                # Debug logging
                if os.environ.get("PPO_DEBUG"):
                    self.logger.info(f"[PPO Debug] [4.4_ppo_update] PPO update cycle {self.ppo_cycle_count + 1}")
                
                # Check debug stop
                if os.environ.get("DEBUG_CYCLE_STOP"):
                    stop_after = int(os.environ.get("DEBUG_CYCLE_STOP"))
                    if self.ppo_cycle_count >= stop_after - 1:
                        self.logger.info(f"[PPO Debug] Stopping after {self.ppo_cycle_count + 1} PPO cycles")
                        import sys
                        sys.exit(0)
                
                self.ppo_cycle_count += 1
'''
            content = content.replace('# Perform PPO training using rl_games components\n                total_loss = self.rlgames_ppo_update(',
                                    debug_code + '\n                # Perform PPO training using rl_games components\n                total_loss = self.rlgames_ppo_update(')
            
            with open(trainer_file, 'w') as f:
                f.write(content)
    
    def compare_logs(self):
        """Compare the debug logs from both systems"""
        print("\n" + "="*80)
        print("COMPARISON OF PPO EXECUTION")
        print("="*80)
        
        print(f"\nDNNE captured {len(self.dnne_logs)} debug events")
        print(f"IsaacGymEnvs captured {len(self.isaacgym_logs)} debug events")
        
        # Look for specific patterns
        dnne_phases = self._extract_phases(self.dnne_logs)
        isaac_phases = self._extract_phases(self.isaacgym_logs)
        
        print("\nPhase execution order:")
        print("\nDNNE:")
        for phase in dnne_phases[:20]:  # First 20 phases
            print(f"  {phase}")
            
        print("\nIsaacGymEnvs:")
        for phase in isaac_phases[:20]:  # First 20 phases
            print(f"  {phase}")
        
        # Look for divergence
        min_len = min(len(dnne_phases), len(isaac_phases))
        divergence_point = None
        for i in range(min_len):
            if dnne_phases[i] != isaac_phases[i]:
                divergence_point = i
                break
        
        if divergence_point:
            print(f"\n⚠️  Divergence found at step {divergence_point}:")
            print(f"  DNNE: {dnne_phases[divergence_point]}")
            print(f"  IsaacGymEnvs: {isaac_phases[divergence_point]}")
        else:
            print("\n✅ No divergence found in captured phases")
    
    def _extract_phases(self, logs):
        """Extract phase identifiers from logs"""
        phases = []
        for log in logs:
            for phase_id, phase_name in PPO_PHASES.items():
                if f"[{phase_id}_" in log:
                    phases.append(phase_id)
                    break
        return phases

def main():
    parser = argparse.ArgumentParser(description="Debug PPO implementation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug-cycles", type=int, default=1, help="Number of PPO cycles to run")
    args = parser.parse_args()
    
    # Activate conda environment
    activate_cmd = "source /home/asantanna/miniconda/bin/activate DNNE_PY38"
    subprocess.run(activate_cmd, shell=True, executable="/bin/bash")
    
    debugger = PPODebugger(seed=args.seed, debug_cycles=args.debug_cycles)
    
    # Run both systems
    dnne_output = debugger.run_dnne_debug()
    isaac_output = debugger.run_isaacgym_debug()
    
    # Compare results
    debugger.compare_logs()
    
    # Save detailed results
    results = {
        "seed": args.seed,
        "debug_cycles": args.debug_cycles,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dnne_logs": debugger.dnne_logs,
        "isaacgym_logs": debugger.isaacgym_logs,
        "dnne_output_lines": len(dnne_output),
        "isaacgym_output_lines": len(isaac_output)
    }
    
    output_file = "/tmp/ppo_debug_comparison.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()