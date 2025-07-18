#!/usr/bin/env python3
"""
IsaacGymEnvs debug runner with detailed logging for PPO comparison
"""

import sys
import os
import torch
import numpy as np
import random

# Add IsaacGymEnvs to path
sys.path.append('/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs')

# Set environment variable for deterministic CUBLAS
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-envs", type=int, default=512, help="Number of environments")
    parser.add_argument("--horizon-length", type=int, default=16, help="Horizon length for PPO")
    parser.add_argument("--max-iterations", type=int, default=1, help="Max training iterations")
    args = parser.parse_args()
    
    # Set fixed seed BEFORE any imports that might use random numbers
    print(f"🔒 Setting fixed seed: {args.seed}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Now import IsaacGymEnvs components
    import hydra
    from omegaconf import DictConfig, OmegaConf
    from isaacgymenvs import isaacgymenvs
    from rl_games.algos_torch import torch_ext
    from rl_games.algos_torch.running_mean_std import RunningMeanStd
    from rl_games.algos_torch import a2c_continuous
    from rl_games.algos_torch import models
    
    # Monkey patch to add debug logging
    original_rms_init = RunningMeanStd.__init__
    def debug_rms_init(self, shape, device='cuda'):
        original_rms_init(self, shape, device)
        print(f"[IsaacGymEnvs Debug] RunningMeanStd initialized: shape={shape}, device={device}")
        print(f"[IsaacGymEnvs Debug] Initial mean: {self.mean.tolist() if hasattr(self.mean, 'tolist') else self.mean}")
        print(f"[IsaacGymEnvs Debug] Initial var: {self.var.tolist() if hasattr(self.var, 'tolist') else self.var}")
    RunningMeanStd.__init__ = debug_rms_init
    
    # Monkey patch the actor-critic model to capture debug info
    original_forward = None
    
    def debug_forward(self, obs_dict):
        """Wrapped forward pass with debug logging"""
        obs = obs_dict['obs']
        
        # Only log first forward pass
        if not hasattr(self, '_debug_logged'):
            self._debug_logged = True
            print(f"[IsaacGymEnvs Debug] First forward pass:")
            print(f"[IsaacGymEnvs Debug] Raw observations shape: {obs.shape}")
            print(f"[IsaacGymEnvs Debug] Raw observations (first env, first 5): {obs[0][:5].tolist() if obs.shape[0] > 0 else 'empty'}")
            
            # Log network weights
            if hasattr(self, 'actor_mlp'):
                for i, layer in enumerate(self.actor_mlp.modules()):
                    if hasattr(layer, 'weight'):
                        print(f"[IsaacGymEnvs Debug] actor_mlp.{i}.weight shape: {layer.weight.shape}, first 5: {layer.weight.data.flatten()[:5].tolist()}")
                        break  # Just log first layer
        
        # Call original forward
        result = original_forward(obs_dict)
        
        # Log outputs on first pass
        if hasattr(self, '_debug_logged') and self._debug_logged:
            self._debug_logged = False  # Only log once
            
            if 'mus' in result:
                print(f"[IsaacGymEnvs Debug] Action mean (first env): {result['mus'][0].tolist()}")
            if 'sigmas' in result:
                print(f"[IsaacGymEnvs Debug] Action std (first env): {result['sigmas'][0].tolist()}")
            if 'values' in result:
                print(f"[IsaacGymEnvs Debug] Value (first env): {result['values'][0].item()}")
            if 'actions' in result:
                print(f"[IsaacGymEnvs Debug] Sampled action (first env): {result['actions'][0].tolist()}")
            if 'neglogpacs' in result:
                print(f"[IsaacGymEnvs Debug] Neg log prob (first env): {result['neglogpacs'][0].item()}")
        
        return result
    
    # Monkey patch PPO algorithm to capture training info
    original_train_actor_critic = None
    
    def debug_train_actor_critic(self, input_dict):
        """Wrapped train_actor_critic with debug logging"""
        # Log training batch info on first call
        if not hasattr(self, '_train_debug_logged'):
            self._train_debug_logged = True
            print(f"[IsaacGymEnvs Debug] First PPO update:")
            print(f"[IsaacGymEnvs Debug] Batch size: {input_dict['obs'].shape[0]}")
            print(f"[IsaacGymEnvs Debug] Rewards (first 5): {input_dict['rewards'][:5].tolist()}")
            print(f"[IsaacGymEnvs Debug] Values (first 5): {input_dict['values'][:5].tolist()}")
            print(f"[IsaacGymEnvs Debug] Advantages (first 5): {input_dict['advantages'][:5].tolist()}")
            print(f"[IsaacGymEnvs Debug] Returns (first 5): {input_dict['returns'][:5].tolist()}")
        
        return original_train_actor_critic(input_dict)
    
    @hydra.main(version_base="1.1", config_path="/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs/isaacgymenvs/cfg", config_name="config")
    def launch_rlg_hydra(cfg: DictConfig):
        # Override config for debugging
        cfg.seed = args.seed
        cfg.task_name = "Cartpole"
        cfg.task.name = "Cartpole"
        cfg.num_envs = args.num_envs
        cfg.task.env.numEnvs = args.num_envs
        cfg.test = False
        cfg.torch_deterministic = True
        cfg.max_iterations = args.max_iterations
        cfg.headless = True
        cfg.graphics_device_id = -1
        
        # Override PPO specific settings to match DNNE
        cfg.train.params.config.horizon_length = args.horizon_length
        cfg.train.params.config.minibatch_size = 8192
        cfg.train.params.config.mini_epochs = 8
        cfg.train.params.config.num_actors = args.num_envs
        cfg.train.params.config.normalize_input = True
        cfg.train.params.config.normalize_value = True
        cfg.train.params.config.gamma = 0.99
        cfg.train.params.config.tau = 0.95
        cfg.train.params.config.e_clip = 0.2
        cfg.train.params.config.learning_rate = 3e-4
        cfg.train.params.config.grad_norm = 1.0
        
        print(f"[IsaacGymEnvs Debug] Configuration:")
        print(f"  Seed: {cfg.seed}")
        print(f"  Num envs: {cfg.num_envs}")
        print(f"  Horizon length: {cfg.train.params.config.horizon_length}")
        print(f"  Normalize input: {cfg.train.params.config.normalize_input}")
        print(f"  Normalize value: {cfg.train.params.config.normalize_value}")
        
        # Apply monkey patches before creating trainer
        global original_forward, original_train_actor_critic
        
        # Import model classes
        from rl_games.algos_torch.models import ModelA2CContinuousLogStd
        original_forward = ModelA2CContinuousLogStd.forward
        ModelA2CContinuousLogStd.forward = debug_forward
        
        # Import PPO algorithm
        from rl_games.algos_torch.a2c_continuous import A2CAgent
        original_train_actor_critic = A2CAgent.train_actor_critic
        A2CAgent.train_actor_critic = debug_train_actor_critic
        
        # Run training
        isaacgymenvs.train(cfg)
    
    # Launch Hydra app
    launch_rlg_hydra()

if __name__ == "__main__":
    main()