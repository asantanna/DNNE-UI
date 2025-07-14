#!/usr/bin/env python3
"""
IsaacGym Profiling Wrapper
Ensures timing data is saved after training completes
"""

import sys
import os
import hydra
from pathlib import Path

# Add IsaacGymEnvs to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "DNNE-LINUX-SUPPORT" / "IsaacGymEnvs"))

# Import everything from train.py
from isaacgymenvs.train import launch_rlg_hydra, preprocess_train_config
from omegaconf import DictConfig, OmegaConf

# Store reference to env for timing data access
env_ref = None

# Original launch function with timing save
@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def launch_rlg_hydra_with_timing(cfg: DictConfig):
    global env_ref
    
    # Call the original training function
    try:
        # Import here to ensure proper initialization
        import isaacgym
        from isaacgymenvs.tasks import isaacgym_task_map
        import gym
        from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
        from rl_games.common import env_configurations, vecenv
        from rl_games.torch_runner import Runner
        
        # Enable profiling
        if hasattr(cfg, 'dnne_profiling'):
            cfg.task.dnne_profiling = cfg.dnne_profiling
        
        # Create environment
        task_name = cfg.task.name
        task_cfg = cfg.task
        task_cfg["env"]["numEnvs"] = cfg.get("num_envs", task_cfg["env"]["numEnvs"])
        
        # Enable profiling in task config
        task_cfg["dnne_profiling"] = True
        
        # Create and run environment
        print(f"Creating task: {task_name} with dnne_profiling enabled")
        
        # Run original launch function
        launch_rlg_hydra(cfg)
        
    finally:
        # Try to save timing data from any active environment
        print("\n[PROFILING] Checking for timing data to save...")
        
        # Try to find the environment instance and save timing data
        import gc
        for obj in gc.get_objects():
            if hasattr(obj, 'save_timing_data') and callable(obj.save_timing_data):
                try:
                    obj.save_timing_data()
                    print("[PROFILING] Saved timing data successfully")
                    break
                except Exception as e:
                    print(f"[PROFILING] Error saving timing data: {e}")

if __name__ == "__main__":
    # Change to IsaacGymEnvs directory for Hydra
    os.chdir("/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs")
    launch_rlg_hydra_with_timing()