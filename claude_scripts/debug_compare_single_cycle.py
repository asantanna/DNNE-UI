#!/usr/bin/env python3
"""
Compare single PPO cycle between DNNE and IsaacGymEnvs with fixed seed
"""

import subprocess
import sys
import os
import time
import json

def run_dnne_single_cycle(seed=42):
    """Run DNNE for single PPO cycle and capture debug output"""
    print("\n" + "="*80)
    print(f"Running DNNE with fixed seed {seed}")
    print("="*80)
    
    # Export fresh workflow
    dnne_path = "/mnt/e/ALS-Projects/DNNE/DNNE-UI"
    os.chdir(dnne_path)
    
    export_cmd = [sys.executable, "claude_scripts/programmatic_export.py", "Cartpole_PPO"]
    subprocess.run(export_cmd, check=True)
    
    # Run with fixed seed
    export_dir = "export_system/exports/Cartpole_PPO"
    os.chdir(export_dir)
    
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    cmd = [
        sys.executable,
        "runner.py",
        f"--fixed-seed={seed}",
        "--epochs=1",
        "--timeout=3s",
        "--headless",
        "--verbose"
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, bufsize=1, env=env)
    
    dnne_debug = {
        "initial_obs": None,
        "initial_weights": [],
        "first_action": None,
        "first_value": None,
        "first_reward": None,
        "gae_info": None
    }
    
    for line in iter(process.stdout.readline, ''):
        if line:
            # Capture key debug values
            if "[PPO Agent Debug] Initial model weights:" in line:
                dnne_debug["capturing_weights"] = True
            elif dnne_debug.get("capturing_weights") and "shared.0.weight:" in line:
                dnne_debug["initial_weights"].append(line.strip())
                dnne_debug["capturing_weights"] = False
            elif "[PPO Agent Debug] Raw observations (first 5):" in line and dnne_debug["initial_obs"] is None:
                dnne_debug["initial_obs"] = line.strip()
            elif "[PPO Agent Debug] Sampled action:" in line and dnne_debug["first_action"] is None:
                dnne_debug["first_action"] = line.strip()
            elif "[PPO Agent Debug] Value output:" in line and dnne_debug["first_value"] is None:
                dnne_debug["first_value"] = line.strip()
            elif "[PPO Trainer Debug] Reward:" in line and dnne_debug["first_reward"] is None:
                dnne_debug["first_reward"] = line.strip()
            elif "[PPO Trainer Debug] Computing GAE" in line:
                dnne_debug["gae_info"] = line.strip()
                
            print(f"DNNE: {line.rstrip()}")
    
    process.wait()
    os.chdir(dnne_path)
    
    return dnne_debug

def run_isaacgymenvs_single_cycle(seed=42):
    """Run IsaacGymEnvs for single cycle and capture output"""
    print("\n" + "="*80)
    print(f"Running IsaacGymEnvs with fixed seed {seed}")
    print("="*80)
    
    isaacgymenvs_path = "/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs"
    os.chdir(isaacgymenvs_path)
    
    # Create a custom script to capture debug info
    debug_script = """
import sys
sys.path.append('/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs')

import torch
import numpy as np
import random

# Set fixed seed
seed = {SEED}
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Import after seed setting
from isaacgymenvs import isaacgymenvs
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base="1.1", config_path="./cfg", config_name="config")
def launch_rlg_hydra(cfg: DictConfig):
    # Override config for single step
    cfg.seed = seed
    cfg.task_name = "Cartpole"
    cfg.num_envs = 512
    cfg.test = False
    cfg.torch_deterministic = True
    cfg.max_iterations = 1
    cfg.headless = True
    cfg.task.env.numEnvs = 512
    
    # Print config to verify
    print(f"[IsaacGymEnvs Debug] Seed: {cfg.seed}")
    print(f"[IsaacGymEnvs Debug] Num envs: {cfg.num_envs}")
    
    # Create trainer
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.algo_observer import AlgoObserver
    from rl_games.algos_torch import torch_ext
    from rl_games.algos_torch.running_mean_std import RunningMeanStd
    
    # Monkey patch to capture debug info
    original_init = RunningMeanStd.__init__
    def debug_init(self, shape, device='cuda'):
        original_init(self, shape, device)
        print(f"[IsaacGymEnvs Debug] RunningMeanStd initialized: shape={shape}")
    RunningMeanStd.__init__ = debug_init
    
    # Run training
    isaacgymenvs.train(cfg)

if __name__ == "__main__":
    launch_rlg_hydra()
""".replace("{SEED}", str(seed))
    
    with open("debug_single_cycle.py", "w") as f:
        f.write(debug_script)
    
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    cmd = [sys.executable, "debug_single_cycle.py"]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, bufsize=1, env=env)
    
    igenv_debug = {
        "initial_obs": None,
        "running_mean_std": [],
        "first_step": None
    }
    
    for line in iter(process.stdout.readline, ''):
        if line:
            # Capture key values
            if "RunningMeanStd initialized:" in line:
                igenv_debug["running_mean_std"].append(line.strip())
            elif "fps step:" in line and igenv_debug["first_step"] is None:
                igenv_debug["first_step"] = line.strip()
                
            print(f"IsaacGymEnvs: {line.rstrip()}")
    
    process.wait()
    
    return igenv_debug

def main():
    seed = 42
    
    # Run both systems
    dnne_results = run_dnne_single_cycle(seed)
    igenv_results = run_isaacgymenvs_single_cycle(seed)
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print("\nDNNE Debug Info:")
    for key, value in dnne_results.items():
        if value and not key.startswith("capturing"):
            print(f"  {key}: {value}")
    
    print("\nIsaacGymEnvs Debug Info:")
    for key, value in igenv_results.items():
        if value:
            print(f"  {key}: {value}")
    
    # Save detailed results
    results = {
        "seed": seed,
        "dnne": dnne_results,
        "isaacgymenvs": igenv_results
    }
    
    with open("/mnt/e/ALS-Projects/DNNE/DNNE-UI/claude_scripts/single_cycle_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: single_cycle_comparison.json")

if __name__ == "__main__":
    main()