#!/usr/bin/env python3
"""
IsaacGymEnvs single PPO cycle runner - exits after 16 steps
"""

import sys
sys.path.append('/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs')
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# CRITICAL: Import isaacgym BEFORE torch
import isaacgym

# NOW import torch after isaacgym
import torch
import numpy as np
import random

# Set seed after imports
seed = 42
print(f"[PPO_CYCLE] Setting seed: {seed}")
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
cfg = omegaconf.OmegaConf.create({
    "task": {
        "name": "Cartpole",
        "physics_engine": "physx",
        "env": {
            "numEnvs": 512,
            "envSpacing": 4.0,
            "resetDist": 3.0,
            "maxEffort": 400.0,
            "clipObservations": 5.0,
            "clipActions": 1.0
        },
        "sim": {
            "dt": 0.0166,
            "substeps": 2,
            "up_axis": "z",
            "use_gpu_pipeline": True,
            "gravity": [0.0, 0.0, -9.81],
            "physx": {
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
            }
        }
    },
    "physics_engine": "physx",
    "sim_device": "cuda:0",
    "rl_device": "cuda:0",
    "graphics_device_id": 0,
    "headless": True,
    "seed": seed
})

# Create environment
print("[PPO_CYCLE] Creating Cartpole environment")
env = isaacgym_task_map["Cartpole"](
    cfg=omegaconf_to_dict(cfg.task), 
    rl_device="cuda:0",
    sim_device="cuda:0",
    graphics_device_id=0,
    headless=True,
    virtual_screen_capture=False,
    force_render=False
)

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
obs_dict = env.reset()
obs = obs_dict["obs"] if isinstance(obs_dict, dict) else obs_dict
print(f"[PPO_CYCLE] Initial observation shape: {obs.shape}")

# Collect data for exactly 16 steps (horizon length)
observations = []
actions = []
values = []
rewards = []

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
    
    # Log first 5 steps
    if step < 5:
        print(f"[PPO_CYCLE] Step {step+1}: action={action[0].item():.4f}, value={value[0].item():.4f}")
    
    # Store data
    observations.append(obs)
    actions.append(action)
    values.append(value)
    
    # Step environment
    obs_dict, reward, done, info = env.step(action)
    obs = obs_dict["obs"] if isinstance(obs_dict, dict) else obs_dict
    rewards.append(reward)

# Log collected data summary
print(f"[PPO_CYCLE] Buffer full (16 steps) - starting PPO update")
print(f"[PPO_CYCLE] First 5 rewards: {[r[0].item() for r in rewards[:5]]}")
print(f"[PPO_CYCLE] First 5 values: {[v[0].item() for v in values[:5]]}")
print(f"[PPO_CYCLE] First 5 actions: {[a[0].item() for a in actions[:5]]}")
print("[PPO_CYCLE] First PPO cycle complete - exiting now")

# EXIT after first cycle
sys.exit(0)