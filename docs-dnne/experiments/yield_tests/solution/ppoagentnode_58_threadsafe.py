import asyncio
import time
from framework import QueueNode, SensorNode

#!/usr/bin/env python3
"""
PPO Agent Node - Thread-safe version with proper yielding
This version patches sync_adaptive_yield to work from threads
"""

import os
import sys
import asyncio
import time
from pathlib import Path
from framework import QueueNode
from framework.globals import Global
from framework.globals_threadsafe import ThreadSafeYielder, thread_safe_sync_adaptive_yield
import json

class PPOAgentNode_58(QueueNode):
    """
    PPO Agent Node that runs IsaacGymEnvs training with thread-safe yielding
    """
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=[])
        self.setup_outputs(["metrics"])
        
        # Node configuration (same as original)
        self.network_mlp_layers = [32, 32]
        self.network_activation = "elu"
        self.separate_value_network = False
        self.checkpoint_interval = 25
        self.keep_checkpoints = 5
        self.load_checkpoint = ""
        self.log_interval = 10
        self.save_interval = 25
        self.experiment_name = "Cartpole_PPO"
        self.mixed_precision = False
        self.multi_gpu = False
        
        # Environment configuration from virtual node
        self.env_config = {
            'task': 'Cartpole',
            'num_envs': 512,
            'seed': 42,
            'headless': False,
            'graphics_device_id': 0,
            'sim_device': 'cuda:0',
            'physics_engine': 'physx',
            'multi_gpu': False,
            'enable_cameras': False,
            'force_render': True,
            'use_gpu_pipeline': True,
            'num_threads': 0,
            'solver_type': 1,
            'num_subscenes': 0,
        }
        
        # Validate task selection
        if self.env_config['task'] == 'none':
            raise ValueError(
                "Invalid environment task: 'none'. "
                "Please select a valid task from the dropdown in the IsaacGymEnvs node before exporting."
            )
        
        # PPO configuration from virtual node
        self.ppo_config = {
            'minibatch_size': 8192,
            'horizon_length': 16,
            'learning_rate': 0.00030000000000000014,
            'schedule_type': 'adaptive',
            'gamma': 0.9900000000000002,
            'tau': 0.9500000000000002,
            'e_clip': 0.20000000000000004,
            'clip_value': True,
            'mini_epochs': 8,
            'critic_coef': 4,
            'entropy_coef': 0,
            'bounds_loss_coef': 0.00010000000000000002,
            'max_epochs': 100,
            'normalize_advantage': True,
            'normalize_input': True,
            'normalize_value': True,
        }
        
        # Check for node-specific overrides
        max_iterations_override = Global.get_node_config(self.node_id, 'max_iterations', None)
        if max_iterations_override is not None:
            self.ppo_config['max_epochs'] = max_iterations_override
            self.logger.info(f"Using max_iterations override from node config: {max_iterations_override}")
        
        # IsaacGymEnvs path
        self.isaac_gym_envs_path = "/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs"
        
        # Track training state
        self.training_completed = False
        self.training_state = None
    
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
        Run IsaacGymEnvs train.py with thread-safe yielding
        """
        print(f"[PPO_AGENT_THREADSAFE] Starting training with thread-safe yielding")
        
        # Set up thread-safe yielder
        loop = asyncio.get_running_loop()
        yielder = ThreadSafeYielder.get_instance()
        await yielder.start(loop)
        
        # Patch Global.sync_adaptive_yield to use thread-safe version
        original_sync_yield = Global.sync_adaptive_yield
        Global.sync_adaptive_yield = classmethod(lambda cls: thread_safe_sync_adaptive_yield())
        
        try:
            # Create configuration for train.py
            train_config = self._create_train_config()
            
            # Run training with async wrapper
            metrics = await self._run_training_async(train_config)
            
            return {"metrics": metrics}
            
        finally:
            # Restore original sync_adaptive_yield
            Global.sync_adaptive_yield = original_sync_yield
    
    def _create_train_config(self):
        """
        Create configuration arguments for IsaacGymEnvs train.py
        (Same as original implementation)
        """
        # Check for visual mode override from command line
        visual_mode = Global.visual_mode
        headless_mode = Global.headless_mode
        
        # Determine headless and force_render settings
        force_render = self.env_config['force_render']
        
        if visual_mode:
            # Visual mode: enable GUI
            headless = False
        elif headless_mode:
            # Headless mode: disable GUI and force_render
            headless = True
            force_render = False  # Override force_render in headless mode
        else:
            # Use settings from config
            headless = self.env_config['headless']
        
        # Base configuration
        config_args = [
            f"task={self.env_config['task']}",
            f"num_envs={self.env_config['num_envs']}",
            f"headless={headless}",
            f"force_render={force_render}",
            f"sim_device={self.env_config['sim_device']}",
            f"rl_device={self.env_config['sim_device']}",
            f"physics_engine={self.env_config['physics_engine']}",
            f"pipeline=gpu",
        ]
        
        # PPO algorithm configuration
        ppo_args = [
            f"train.params.config.minibatch_size={self.ppo_config['minibatch_size']}",
            f"train.params.config.horizon_length={self.ppo_config['horizon_length']}",
            f"train.params.config.learning_rate={self.ppo_config['learning_rate']}",
            f"train.params.config.lr_schedule={self.ppo_config['schedule_type']}",
            f"train.params.config.gamma={self.ppo_config['gamma']}",
            f"train.params.config.tau={self.ppo_config['tau']}",
            f"train.params.config.e_clip={self.ppo_config['e_clip']}",
            f"train.params.config.clip_value={self.ppo_config['clip_value']}",
            f"train.params.config.mini_epochs={self.ppo_config['mini_epochs']}",
            f"train.params.config.critic_coef={self.ppo_config['critic_coef']}",
            f"train.params.config.entropy_coef={self.ppo_config['entropy_coef']}",
            f"train.params.config.bounds_loss_coef={self.ppo_config['bounds_loss_coef']}",
            f"train.params.config.normalize_advantage={self.ppo_config['normalize_advantage']}",
            f"train.params.config.normalize_input={self.ppo_config['normalize_input']}",
            f"train.params.config.normalize_value={self.ppo_config['normalize_value']}",
            f"train.params.config.max_epochs={self.ppo_config['max_epochs']}",
        ]
        
        # Network configuration
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
    
    async def _run_training_async(self, train_config):
        """
        Run IsaacGymEnvs training with thread-safe yielding.
        This version runs training in a thread pool with proper yield support.
        """
        print(f"[PPO_AGENT_THREADSAFE] Setting up async training with thread-safe yield")
        
        # Change to IsaacGymEnvs directory
        isaac_gym_envs_path = Path(self.isaac_gym_envs_path)
        if not isaac_gym_envs_path.exists():
            raise RuntimeError(f"IsaacGymEnvs path not found: {isaac_gym_envs_path}")
        
        # Save current directory
        original_dir = os.getcwd()
        
        # Get event loop for running sync code
        loop = asyncio.get_running_loop()
        
        # Create a wrapper that runs training
        def run_training_with_patched_yield():
            """Run training with patched sync_adaptive_yield"""
            try:
                # Add original directory to Python path
                if str(original_dir) not in sys.path:
                    sys.path.insert(0, str(original_dir))
                
                # Change to IsaacGymEnvs/isaacgymenvs directory
                os.chdir(isaac_gym_envs_path / "isaacgymenvs")
                
                # Set up environment
                os.environ['HYDRA_FULL_ERROR'] = '1'
                os.environ['DNNE_ADAPTIVE_YIELD'] = '1'
                
                # Convert args to sys.argv format
                sys.argv = ["train.py"] + train_config
                
                # Import and run train.py
                import runpy
                print(f"[PPO_AGENT_THREADSAFE] Starting training with runpy")
                result = runpy.run_path("train.py", run_name="__main__")
                
                # Extract metrics
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
                if str(original_dir) in sys.path:
                    sys.path.remove(str(original_dir))
        
        try:
            # Run training in executor with thread-safe yielding
            print(f"[PPO_AGENT_THREADSAFE] Running training in executor with thread-safe yield")
            
            # Use thread pool executor to run sync training code
            metrics = await loop.run_in_executor(None, run_training_with_patched_yield)
            
            print(f"[PPO_AGENT_THREADSAFE] Training completed with metrics: {metrics}")
            return metrics
            
        except Exception as e:
            print(f"[PPO_AGENT_THREADSAFE] Error during training: {e}")
            raise