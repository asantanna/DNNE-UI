#!/usr/bin/env python3
"""
Debug single PPO cycle comparison between DNNE and IsaacGymEnvs
Following docs-dnne/for_claude/debug_strategy_for_ppo.md
"""

import subprocess
import sys
import os
import json
import numpy as np
import re

def add_dnne_debug_logging():
    """Add debug logging to DNNE nodes for PPO cycle tracking"""
    
    # Export fresh
    subprocess.run([sys.executable, "claude_scripts/programmatic_export.py", "Cartpole_PPO"], 
                   check=True, capture_output=True)
    
    # Add to Isaac Gym Step Node to track steps
    step_file = "export_system/exports/Cartpole_PPO/nodes/isaacgymstepnode_9.py"
    with open(step_file, 'r') as f:
        content = f.read()
    
    if "PPO_CYCLE_DEBUG" not in content:
        # Add step counter
        content = content.replace('self.throttle_steps = 1',
                                'self.throttle_steps = 1\n        self.ppo_step_count = 0')
        
        # Add debug in compute
        debug_code = '''
        # PPO cycle debug tracking
        if os.environ.get("PPO_CYCLE_DEBUG"):
            self.ppo_step_count += 1
            self.logger.info(f"[PPO_CYCLE] Step {self.ppo_step_count}: Env stepped")
            if self.ppo_step_count >= 16:  # Horizon length
                self.logger.info("[PPO_CYCLE] Reached horizon length - PPO update should trigger")
'''
        content = content.replace('# Apply smart throttling',
                                debug_code + '\n        # Apply smart throttling')
        
        with open(step_file, 'w') as f:
            f.write(content)
    
    # Add to PPO Agent for forward pass tracking
    agent_file = "export_system/exports/Cartpole_PPO/nodes/ppoagentnode_3.py"
    with open(agent_file, 'r') as f:
        content = f.read()
    
    if "PPO_CYCLE_DEBUG" not in content:
        # Add debug at start of compute
        debug_code = '''
        # PPO cycle debug
        if os.environ.get("PPO_CYCLE_DEBUG"):
            self.logger.info(f"[PPO_CYCLE] Forward pass - obs shape: {observations.shape if isinstance(observations, torch.Tensor) else 'not tensor'}")
'''
        content = content.replace('try:\n            # Ensure observations',
                                'try:' + debug_code + '\n            # Ensure observations')
        
        # Add debug after action sampling
        debug_code2 = '''
                if os.environ.get("PPO_CYCLE_DEBUG"):
                    self.logger.info(f"[PPO_CYCLE] Action sampled: mean={action_mean[0].tolist()}, std={action_std.tolist()}, action={action[0].tolist()}, value={value[0].item()}")
'''
        content = content.replace('# Compute log probability\n                log_prob = policy_dist.log_prob(action).sum(dim=-1)',
                                '# Compute log probability\n                log_prob = policy_dist.log_prob(action).sum(dim=-1)' + debug_code2)
        
        with open(agent_file, 'w') as f:
            f.write(content)
    
    # Add to PPO Trainer for update tracking
    trainer_file = "export_system/exports/Cartpole_PPO/nodes/ppotrainernode_6.py"
    with open(trainer_file, 'r') as f:
        content = f.read()
    
    if "PPO_CYCLE_DEBUG" not in content:
        # Add debug when buffer is full
        debug_code = '''
                if os.environ.get("PPO_CYCLE_DEBUG"):
                    self.logger.info(f"[PPO_CYCLE] Buffer full ({len(self.buffer_states)} steps) - starting PPO update")
                    self.logger.info(f"[PPO_CYCLE] First 5 rewards: {rewards[:5].tolist()}")
                    self.logger.info(f"[PPO_CYCLE] First 5 values: {values[:5].tolist()}")
                    
                    # Stop after first cycle
                    if os.environ.get("PPO_STOP_AFTER_CYCLE") == "1":
                        self.logger.info("[PPO_CYCLE] Stopping after first PPO cycle as requested")
                        import sys
                        sys.exit(0)
'''
        content = content.replace('# Convert buffer to tensors\n                states = torch.stack(self.buffer_states)',
                                debug_code + '\n                # Convert buffer to tensors\n                states = torch.stack(self.buffer_states)')
        
        with open(trainer_file, 'w') as f:
            f.write(content)

def run_dnne_single_cycle(seed=42):
    """Run DNNE for one PPO cycle with debug output"""
    print("\n" + "="*80)
    print(f"Running DNNE single PPO cycle (seed={seed})")
    print("="*80)
    
    # Add debug logging
    add_dnne_debug_logging()
    
    # Run with debug flags
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PPO_CYCLE_DEBUG"] = "1"
    env["PPO_STOP_AFTER_CYCLE"] = "1"
    
    cmd = [
        sys.executable,
        "export_system/exports/Cartpole_PPO/runner.py",
        f"--fixed-seed={seed}",
        "--epochs=1",
        "--headless",
        "--verbose"
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, bufsize=1, env=env)
    
    dnne_logs = []
    for line in iter(process.stdout.readline, ''):
        if line:
            if "[PPO_CYCLE]" in line:
                print(f"DNNE: {line.rstrip()}")
                dnne_logs.append(line.rstrip())
    
    process.wait()
    return dnne_logs

def run_isaacgym_single_cycle(seed=42):
    """Run IsaacGymEnvs for one PPO cycle with debug output"""
    print("\n" + "="*80)
    print(f"Running IsaacGymEnvs single PPO cycle (seed={seed})")
    print("="*80)
    
    # Create minimal debug runner
    debug_runner = '''
import sys
sys.path.append('/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs')
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import numpy as np
import random

# Set seed BEFORE any imports
seed = {seed}
print(f"[PPO_CYCLE] Setting seed: {{seed}}")
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Now import IsaacGymEnvs
from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd

# Minimal Cartpole config
import omegaconf
cfg = omegaconf.OmegaConf.create({{
    "task": {{
        "name": "Cartpole",
        "physics_engine": "physx",
        "env": {{
            "numEnvs": 512,
            "envSpacing": 4.0,
            "resetDist": 3.0,
            "maxEffort": 400.0,
            "clipObservations": 5.0,
            "clipActions": 1.0
        }},
        "sim": {{
            "dt": 0.0166,
            "substeps": 2,
            "up_axis": "z",
            "use_gpu_pipeline": True,
            "gravity": [0.0, 0.0, -9.81],
            "physx": {{
                "num_threads": 4,
                "solver_type": 1,
                "use_gpu": True,
                "num_position_iterations": 4,
                "num_velocity_iterations": 0,
                "contact_offset": 0.02,
                "rest_offset": 0.001,
                "bounce_threshold_velocity": 0.2,
                "max_depenetration_velocity": 100.0,
                "default_buffer_size_multiplier": 5.0,
                "max_gpu_contact_pairs": 8388608,
                "num_subscenes": 4,
                "contact_collection": 0
            }}
        }}
    }},
    "physics_engine": "physx",
    "sim_device": "cuda:0",
    "rl_device": "cuda:0",
    "graphics_device_id": 0,
    "headless": True,
    "seed": seed
}})

# Create environment
print("[PPO_CYCLE] Creating Cartpole environment")
env = isaacgym_task_map["Cartpole"](cfg=omegaconf_to_dict(cfg.task), 
                                    sim_device="cuda:0",
                                    graphics_device_id=0,
                                    headless=True)

# Create simple PPO agent
print("[PPO_CYCLE] Creating PPO agent")
obs_shape = (4,)  # Cartpole observation space
act_shape = (1,)  # Cartpole action space

# Simple actor-critic network
class SimpleAC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ELU(),
            torch.nn.Linear(32, 32),
            torch.nn.ELU(),
            torch.nn.Linear(32, 1)
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ELU(),
            torch.nn.Linear(32, 32),
            torch.nn.ELU(),
            torch.nn.Linear(32, 1)
        )
        self.log_std = torch.nn.Parameter(torch.zeros(1))
        
    def forward(self, obs):
        return self.actor(obs), self.critic(obs), self.log_std.expand(obs.shape[0], 1)

model = SimpleAC().cuda()

# Initialize observation normalization
obs_rms = RunningMeanStd(obs_shape).cuda()

# Reset environment
obs = env.reset()
print(f"[PPO_CYCLE] Initial observation shape: {{obs.shape}}")

# Run exactly 16 steps (horizon length)
for step in range(16):
    # Update observation normalization
    obs_rms.update(obs)
    norm_obs = obs_rms.normalize(obs)
    
    # Forward pass
    with torch.no_grad():
        action_mean, value, log_std = model(norm_obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
    
    if step < 5:  # Log first 5 steps
        print(f"[PPO_CYCLE] Step {{step+1}}: action_mean={{action_mean[0].item():.4f}}, std={{std[0].item():.4f}}, action={{action[0].item():.4f}}, value={{value[0].item():.4f}}")
    
    # Step environment
    obs, reward, done, info = env.step(action)
    
    if step == 15:
        print(f"[PPO_CYCLE] Completed 16 steps - PPO update would trigger now")
        print("[PPO_CYCLE] Stopping after first cycle")

print("[PPO_CYCLE] IsaacGymEnvs single cycle complete")
'''
    
    # Write and run debug script
    runner_path = "/tmp/isaacgym_single_cycle.py"
    with open(runner_path, 'w') as f:
        f.write(debug_runner.format(seed=seed))
    
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    cmd = [sys.executable, runner_path]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, bufsize=1, env=env)
    
    isaac_logs = []
    for line in iter(process.stdout.readline, ''):
        if line:
            if "[PPO_CYCLE]" in line:
                print(f"IsaacGymEnvs: {line.rstrip()}")
                isaac_logs.append(line.rstrip())
    
    process.wait()
    return isaac_logs

def compare_cycles(dnne_logs, isaac_logs):
    """Compare the single PPO cycle outputs"""
    print("\n" + "="*80)
    print("SINGLE PPO CYCLE COMPARISON")
    print("="*80)
    
    # Extract values from logs
    dnne_values = extract_values(dnne_logs)
    isaac_values = extract_values(isaac_logs)
    
    print("\n1. Forward Pass Comparison (first few steps):")
    print("   DNNE actions:", dnne_values.get("actions", [])[:3])
    print("   Isaac actions:", isaac_values.get("actions", [])[:3])
    
    print("\n2. Value Predictions:")
    print("   DNNE values:", dnne_values.get("values", [])[:3])
    print("   Isaac values:", isaac_values.get("values", [])[:3])
    
    print("\n3. Cycle Completion:")
    print(f"   DNNE reached {dnne_values.get('steps', 0)} steps")
    print(f"   Isaac reached {isaac_values.get('steps', 0)} steps")
    
    # Identify key differences
    differences = []
    if dnne_values.get("actions") and isaac_values.get("actions"):
        if abs(dnne_values["actions"][0] - isaac_values["actions"][0]) > 0.1:
            differences.append("Initial actions differ significantly")
    
    if differences:
        print(f"\n⚠️  Key differences found: {differences}")
    else:
        print("\n✅ Outputs appear similar")

def extract_values(logs):
    """Extract numerical values from log lines"""
    values = {"actions": [], "values": [], "steps": 0}
    
    for log in logs:
        # Extract actions
        if "action=" in log and "action_mean" not in log:
            match = re.search(r'action=\[([-\d.]+)\]', log)
            if match:
                values["actions"].append(float(match.group(1)))
        
        # Extract values
        if "value=" in log:
            match = re.search(r'value=([-\d.]+)', log)
            if match:
                values["values"].append(float(match.group(1)))
        
        # Count steps
        if "Step" in log:
            match = re.search(r'Step (\d+):', log)
            if match:
                values["steps"] = max(values["steps"], int(match.group(1)))
    
    return values

def main():
    # Activate conda environment
    activate_cmd = "source /home/asantanna/miniconda/bin/activate DNNE_PY38"
    subprocess.run(activate_cmd, shell=True, executable="/bin/bash")
    
    os.chdir("/mnt/e/ALS-Projects/DNNE/DNNE-UI")
    
    seed = 42
    
    # Run both systems for single cycle
    dnne_logs = run_dnne_single_cycle(seed)
    isaac_logs = run_isaacgym_single_cycle(seed)
    
    # Compare results
    compare_cycles(dnne_logs, isaac_logs)
    
    # Save logs
    results = {
        "seed": seed,
        "dnne_logs": dnne_logs,
        "isaac_logs": isaac_logs
    }
    
    with open("/tmp/single_ppo_cycle_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n💾 Logs saved to /tmp/single_ppo_cycle_comparison.json")

if __name__ == "__main__":
    main()