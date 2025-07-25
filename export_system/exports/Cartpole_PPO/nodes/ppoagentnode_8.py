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
import threading
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
        # Base configuration
        config_args = [
            f"task={self.env_config['task']}",
            f"num_envs={self.env_config['num_envs']}",
            f"headless={self.env_config['headless']}",
            f"device={self.env_config['sim_device']}",
            f"physics_engine={self.env_config['physics_engine']}",
        ]
        
        # PPO algorithm configuration
        ppo_args = [
            f"train.params.config.minibatch_size={self.ppo_config['minibatch_size']}",
            f"train.params.config.horizon_length={self.ppo_config['horizon_length']}",
            f"train.params.config.learning_rate={self.ppo_config['learning_rate']}",
            f"train.params.config.schedule_type={self.ppo_config['schedule_type']}",
            f"train.params.config.num_actors={self.env_config['num_envs']}",
            f"train.params.config.gamma={self.ppo_config['gamma']}",
            f"train.params.config.tau={self.ppo_config['tau']}",
            f"train.params.config.e_clip={self.ppo_config['e_clip']}",
            f"train.params.config.clip_value={self.ppo_config['clip_value']}",
            f"train.params.config.mini_epochs={self.ppo_config['mini_epochs']}",
            f"train.params.config.critic_coef={self.ppo_config['critic_coef']}",
            f"train.params.config.entropy_coef={self.ppo_config['entropy_coef']}",
            f"train.params.config.bounds_loss_coef={self.ppo_config['bounds_loss_coef']}",
            f"train.params.config.max_agent_steps={self.ppo_config['max_agent_steps']}",
        ]
        
        # Network configuration
        network_args = [
            f"train.params.network.mlp.units={self.network_mlp_layers}",
            f"train.params.network.mlp.activation={self.network_activation}",
            f"train.params.network.separate_value={self.separate_value_network}",
        ]
        
        # Training configuration
        training_args = [
            f"checkpoint_interval={self.checkpoint_interval}",
            f"keep_checkpoints={self.keep_checkpoints}",
            f"log_interval={self.log_interval}",
            f"save_interval={self.save_interval}",
            f"experiment_name={self.experiment_name}",
            f"mixed_precision={self.mixed_precision}",
            f"multi_gpu={self.multi_gpu}",
        ]
        
        # Load checkpoint if specified
        if self.load_checkpoint:
            training_args.append(f"checkpoint={self.load_checkpoint}")
        
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
            # Change to IsaacGymEnvs directory
            os.chdir(isaac_gym_envs_path)
            
            # Set up Hydra configuration path
            os.environ['HYDRA_FULL_ERROR'] = '1'
            
            # Import and run train.py directly
            # This approach allows async yielding to work properly
            sys.path.insert(0, str(isaac_gym_envs_path))
            
            # Import IsaacGymEnvs train module
            from isaacgymenvs.train import launch_rlg_hydra
            
            # Convert args to sys.argv format for hydra
            sys.argv = ["train.py"] + train_config
            
            # Run training with periodic yielding
            print(f"Starting PPO training with IsaacGymEnvs...")
            print(f"Configuration: {' '.join(train_config)}")
            
            # Note: launch_rlg_hydra will run the training loop
            # We need to wrap it to allow yielding
            result = await self._run_with_yielding(launch_rlg_hydra)
            
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
    
    async def _run_with_yielding(self, train_func):
        """
        Run the training function with periodic yielding
        This allows other async tasks to run during long training sessions
        """
        import asyncio
        import threading
        import time
        
        result = {}
        exception = None
        
        def run_training():
            nonlocal result, exception
            try:
                # Run the training function
                # Note: This is a blocking call, but we're in a thread
                result = train_func()
            except Exception as e:
                exception = e
        
        # Start training in a separate thread
        training_thread = threading.Thread(target=run_training)
        training_thread.start()
        
        # Periodically yield while training runs
        while training_thread.is_alive():
            await Global.async_adaptive_yield()
            await asyncio.sleep(1.0)  # Check every second
        
        # Training complete, check for exceptions
        if exception:
            raise exception
        
        return result
