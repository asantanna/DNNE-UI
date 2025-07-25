import asyncio
import time
from framework import QueueNode, SensorNode

#!/usr/bin/env python3
"""
PPO Agent Node - Calls IsaacGymEnvs train.py
This node consolidates PPO and environment configuration to run RL training
"""

import os
import sys
import asyncio
import time
from pathlib import Path
from framework import QueueNode
from framework.globals import Global

class PPOAgentNode_8(QueueNode):
    """
    PPO Agent Node that runs IsaacGymEnvs training
    Consolidates environment and PPO configuration from virtual nodes
    """
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=[])
        self.setup_outputs(["metrics"])
        
        # Node configuration
        self.network_mlp_layers = [256, 128, 64]
        self.network_activation = "elu"
        self.separate_value_network = False
        self.checkpoint_interval = 100
        self.keep_checkpoints = 5
        self.load_checkpoint = ""
        self.log_interval = 10
        self.save_interval = 1000
        self.experiment_name = "PPO_DNNE"
        self.mixed_precision = False
        self.multi_gpu = False
        
        # Environment configuration from virtual node
        self.env_config = {
            'task': 'Cartpole',
            'num_envs': 64,
            'seed': 42,
            'headless': True,
            'graphics_device_id': 0,
            'sim_device': 'cuda:0',
            'physics_engine': 'physx',
            'multi_gpu': False,
            'enable_cameras': False,
            'force_render': False,
            'use_gpu_pipeline': True,
            'num_threads': 0,
            'solver_type': 1,
            'num_subscenes': 0,
        }
        
        # PPO configuration from virtual node
        self.ppo_config = {
            'minibatch_size': 8,
            'horizon_length': 16,
            'learning_rate': 0.0003,
            'schedule_type': 'constant',
            'gamma': 0.99,
            'tau': 0.95,
            'e_clip': 0.2,
            'clip_value': True,
            'mini_epochs': 4,
            'critic_coef': 0.5,
            'entropy_coef': 0.01,
            'bounds_loss_coef': 0.0,
            'max_agent_steps': 10000000000,
            'normalize_advantage': True,
            'normalize_input': True,
            'value_bootstrap': True,
            'clip_actions': False,
        }
        
        # IsaacGymEnvs path
        self.isaac_gym_envs_path = "/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs"
        
        # Track if training has been completed
        self.training_completed = False
    
    async def run(self):
        """Override run to execute training only once"""
        self.running = True
        self.logger.info(f"Starting PPO training node {self.node_id}")
        
        try:
            # Run training once
            outputs = await self.compute()
            self.compute_count += 1
            self.last_compute_time = time.time()
            
            # Send outputs to any connected nodes
            for output_name, output_data in outputs.items():
                await self.send_output(output_name, output_data)
            
            self.logger.info(f"PPO training completed for node {self.node_id}")
            self.training_completed = True
            
            # Keep the node alive but idle
            while self.running:
                await asyncio.sleep(1.0)
                
        except asyncio.CancelledError:
            self.logger.info(f"Node {self.node_id} cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in node {self.node_id}: {e}")
            raise
    
    async def compute(self):
        """
        Run IsaacGymEnvs train.py with consolidated configuration
        """
        
        # Create configuration for train.py
        train_config = self._create_train_config()
        
        # Run IsaacGymEnvs train.py
        metrics = await self._run_training(train_config)
        
        return {"metrics": metrics}
    
    def _create_train_config(self):
        """
        Create configuration arguments for IsaacGymEnvs train.py
        """
        # Check for visual mode override from command line
        import builtins
        visual_mode = getattr(builtins, 'VISUAL_MODE', False)
        headless_mode = getattr(builtins, 'HEADLESS_MODE', False)
        
        # Determine headless setting: visual mode overrides everything
        if visual_mode:
            headless = False
            print("🖼️  Visual mode enabled - launching with GUI")
        elif headless_mode:
            headless = True
            print("🖥️  Headless mode enforced")
        else:
            headless = self.env_config['headless']
        
        # Base configuration
        config_args = [
            f"task={self.env_config['task']}",
            f"num_envs={self.env_config['num_envs']}",
            f"headless={headless}",
            f"sim_device={self.env_config['sim_device']}",
            f"rl_device={self.env_config['sim_device']}",  # Use same device for RL
            f"physics_engine={self.env_config['physics_engine']}",
            f"pipeline=gpu",  # GPU pipeline for Isaac Gym
        ]
        
        # PPO algorithm configuration
        ppo_args = [
            f"train.params.config.minibatch_size={self.ppo_config['minibatch_size']}",
            f"train.params.config.horizon_length={self.ppo_config['horizon_length']}",
            f"train.params.config.learning_rate={self.ppo_config['learning_rate']}",
            f"train.params.config.lr_schedule={self.ppo_config['schedule_type']}",  # lr_schedule not schedule_type
            f"train.params.config.gamma={self.ppo_config['gamma']}",
            f"train.params.config.tau={self.ppo_config['tau']}",
            f"train.params.config.e_clip={self.ppo_config['e_clip']}",
            f"train.params.config.clip_value={self.ppo_config['clip_value']}",
            f"train.params.config.mini_epochs={self.ppo_config['mini_epochs']}",
            f"train.params.config.critic_coef={self.ppo_config['critic_coef']}",
            f"train.params.config.entropy_coef={self.ppo_config['entropy_coef']}",
            f"train.params.config.bounds_loss_coef={self.ppo_config['bounds_loss_coef']}",
            # Note: max_agent_steps doesn't exist, use max_epochs instead
            f"train.params.config.max_epochs=1000",  # Default to 1000 epochs
        ]
        
        # Network configuration
        # Format list as [256,128,64] for Hydra
        mlp_units_str = "[" + ",".join(map(str, self.network_mlp_layers)) + "]"
        network_args = [
            f"train.params.network.mlp.units={mlp_units_str}",
            f"train.params.network.mlp.activation={self.network_activation}",
            f"train.params.network.separate={self.separate_value_network}",
        ]
        
        # Training configuration
        training_args = [
            f"train.params.config.save_frequency={self.checkpoint_interval}",
            f"experiment={self.experiment_name}",
            f"train.params.config.mixed_precision={self.mixed_precision}",
            f"multi_gpu={self.multi_gpu}",
        ]
        
        # Load checkpoint if specified
        if self.load_checkpoint:
            config_args.append(f"checkpoint={self.load_checkpoint}")
        
        # Combine all arguments
        all_args = config_args + ppo_args + network_args + training_args
        
        return all_args
    
    async def _run_training(self, train_config):
        """
        Run IsaacGymEnvs train.py with configuration
        This uses subprocess to maintain isolation and proper environment setup
        """
        # Change to IsaacGymEnvs directory
        isaac_gym_envs_path = Path(self.isaac_gym_envs_path)
        if not isaac_gym_envs_path.exists():
            raise RuntimeError(f"IsaacGymEnvs path not found: {isaac_gym_envs_path}")
        
        # Save current directory
        original_dir = os.getcwd()
        
        try:
            # Change to IsaacGymEnvs/isaacgymenvs directory where train.py lives
            os.chdir(isaac_gym_envs_path / "isaacgymenvs")
            
            # Debug: Print current directory to verify we're in the right place
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in directory: {os.listdir('.')[:10]}")
            
            # Set up Hydra configuration path
            os.environ['HYDRA_FULL_ERROR'] = '1'
            
            # Enable DNNE adaptive yielding for cooperative execution
            os.environ['DNNE_ADAPTIVE_YIELD'] = '1'
            
            # Convert args to sys.argv format for hydra
            sys.argv = ["train.py"] + train_config
            
            # Use runpy to execute train.py as __main__
            # This should make Hydra resolve paths correctly
            import runpy
            
            # Run training with periodic yielding
            print(f"Starting PPO training with IsaacGymEnvs...")
            print(f"Configuration: {' '.join(train_config)}")
            
            # Check if event loop is accessible before running
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                print(f"[DNNE_DEBUG] Event loop accessible before runpy: {loop}")
            except RuntimeError as e:
                print(f"[DNNE_DEBUG] No event loop before runpy: {e}")
            
            # Run training directly - it will yield cooperatively
            result = runpy.run_path("train.py", run_name="__main__")
            
            # Extract metrics from result
            metrics = {
                "training_complete": True,
                "final_reward": result.get("final_reward", 0.0),
                "total_steps": result.get("total_steps", 0),
                "training_time": result.get("training_time", 0.0),
            }
            
            return metrics
            
        finally:
            # Restore original directory
            os.chdir(original_dir)
            # Remove from sys.path
            if str(isaac_gym_envs_path) in sys.path:
                sys.path.remove(str(isaac_gym_envs_path))
