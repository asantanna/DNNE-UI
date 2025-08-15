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

# Define local DNNE_print to avoid import order issues
#DBG_TAG# def DNNE_print_local(level, component, message):
#DBG_TAG#      """Local debug print function"""
#DBG_TAG#      print(f"[DNNE_DEBUG] {level}/{component}: {message}", flush=True)

class PPOAgentNode_58(QueueNode):
    """
    PPO Agent Node that runs IsaacGymEnvs training
    Consolidates environment and PPO configuration from virtual nodes
    """
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=[])
        self.setup_outputs(["metrics"])
        
        # Node configuration
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
        
        # Track if training has been completed
        self.training_completed = False
    
    async def run(self):
        """Override run to execute training only once"""
        self.running = True
        self.logger.info(f"Starting PPO training node {self.node_id}")
        #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 1: run() method started")
        
        try:
            # Run training once
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 2: About to call compute()")
            outputs = await self.compute()
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 3: compute() returned")
            self.compute_count += 1
            self.last_compute_time = time.time()
            
            # Send outputs to any connected nodes
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", f"🚀 CHECKPOINT 4: Sending {len(outputs)} outputs")
            for output_name, output_data in outputs.items():
                await self.send_output(output_name, output_data)
            
            self.logger.info(f"PPO training completed for node {self.node_id}")
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 5: Training marked as completed")
            self.training_completed = True
            
            # Keep the node alive but idle
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 6: Entering idle loop")
            while self.running:
                await asyncio.sleep(1.0)
                
        except asyncio.CancelledError:
            self.logger.info(f"Node {self.node_id} cancelled")
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT ERROR: AsyncIO cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in node {self.node_id}: {e}")
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", f"🚀 CHECKPOINT ERROR: Exception {type(e).__name__}: {e}")
            raise
    
    async def compute(self):
        """
        Run IsaacGymEnvs train.py with consolidated configuration
        """
        print("D", "PPO_AGENT", "🚀 CHECKPOINT 7: compute() method started") #DBG_TAG# 
        
        # Create configuration for train.py
        #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 8: Creating train config")
        train_config = self._create_train_config()
        #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", f"🚀 CHECKPOINT 9: Train config created with {len(train_config)} args")
        
        # Run IsaacGymEnvs train.py
        #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 10: About to call _run_training()")
        metrics = await self._run_training(train_config)
        #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 11: _run_training() returned")
        
        return {"metrics": metrics}
    
    def _create_train_config(self):
        """
        Create configuration arguments for IsaacGymEnvs train.py
        """
        # Check for visual mode override from command line
        visual_mode = Global.visual_mode
        headless_mode = Global.headless_mode
        
        # Determine headless and force_render settings
        force_render = self.env_config['force_render']
        
        if visual_mode:
            # Visual mode: enable GUI
            headless = False
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", "🖼️  Visual mode enabled - launching with GUI")
        elif headless_mode:
            # Headless mode: disable GUI and force_render
            headless = True
            force_render = False  # Override force_render in headless mode
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", "🖥️  Headless mode enforced - disabling force_render")
        else:
            # Use settings from config
            headless = self.env_config['headless']
        
        # Base configuration
        config_args = [
            f"task={self.env_config['task']}",
            f"num_envs={self.env_config['num_envs']}",
            f"headless={headless}",
            f"force_render={force_render}",  # Use the potentially overridden value
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
            f"train.params.config.normalize_advantage={self.ppo_config['normalize_advantage']}",
            f"train.params.config.normalize_input={self.ppo_config['normalize_input']}",
            f"train.params.config.normalize_value={self.ppo_config['normalize_value']}",
            f"train.params.config.max_epochs={self.ppo_config['max_epochs']}",
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
        
        #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", f"🚀 Full train config ({len(all_args)} args):")
        #DBG_TAG# for i, arg in enumerate(all_args):
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", f"  [{i}] {arg}")
        
        return all_args
    
    async def _run_training(self, train_config):
        """
        Run IsaacGymEnvs train.py with configuration
        This uses runpy to execute train.py in the same process, preserving the event loop context
        """
        #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 12: _run_training() started")
        
        # Change to IsaacGymEnvs directory
        isaac_gym_envs_path = Path(self.isaac_gym_envs_path)
        if not isaac_gym_envs_path.exists():
            raise RuntimeError(f"IsaacGymEnvs path not found: {isaac_gym_envs_path}")
        
        # Save current directory
        original_dir = os.getcwd()
        #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", f"🚀 CHECKPOINT 13: Original dir: {original_dir}")
        
        try:
            # Add original directory to Python path so rl_games_dnne can find framework module
            if str(original_dir) not in sys.path:
                sys.path.insert(0, str(original_dir))
                #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", f"Added {original_dir} to sys.path for framework imports")
            
            # Change to IsaacGymEnvs/isaacgymenvs directory where train.py lives
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", f"🚀 CHECKPOINT 14: Changing to {isaac_gym_envs_path / 'isaacgymenvs'}")
            os.chdir(isaac_gym_envs_path / "isaacgymenvs")
            
            # Debug: Print current directory to verify we're in the right place
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", f"Current directory: {os.getcwd()}")
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", f"Files in directory: {os.listdir('.')[:10]}")
            
            # Set up Hydra configuration path
            os.environ['HYDRA_FULL_ERROR'] = '1'
            
            # Enable DNNE adaptive yielding for cooperative execution
            os.environ['DNNE_ADAPTIVE_YIELD'] = '1'
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", f"🚀 CHECKPOINT 14.5: DNNE_ADAPTIVE_YIELD={os.environ.get('DNNE_ADAPTIVE_YIELD', 'NOT SET')}")
            
            # Convert args to sys.argv format for hydra
            sys.argv = ["train.py"] + train_config
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", f"🚀 CHECKPOINT 15: sys.argv set to {len(sys.argv)} args")
            
            # Use runpy to execute train.py as __main__
            # This should make Hydra resolve paths correctly
            import runpy
            
            # Run training with periodic yielding
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", f"Starting PPO training with IsaacGymEnvs...")
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", f"Configuration: {' '.join(train_config)}")
            
            # Run training directly - it will yield cooperatively
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPONT 16: About to call runpy.run_path()")
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", f"🚀 CHECKPOINT 16.1: train.py exists: {os.path.exists('train.py')}")
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 16.2: runpy.run_path('train.py', run_name='__main__') starting NOW...")
            loop = asyncio.get_running_loop()
            print(f"[PPO_AGENT] Passing event loop to train.py - Loop ID: {id(loop)}")
            print(f"[PPO_AGENT] Loop object: {loop}")
            result = runpy.run_path("train.py", init_globals={"dnne_loop": loop}, run_name="__main__")
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 17: runpy.run_path() RETURNED!!! (This probably won't print)")
            
            # Extract metrics from result
            metrics = {
                "training_complete": True,
                "final_reward": result.get("final_reward", 0.0),
                "total_steps": result.get("total_steps", 0),
                "training_time": result.get("training_time", 0.0),
            }
            
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", f"🚀 CHECKPOINT 18: Metrics extracted: {metrics}")
            return metrics
            
        finally:
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 19: In finally block")
            # Restore original directory
            os.chdir(original_dir)
            #DBG_TAG# DNNE_print_local("D", "PPO_AGENT", f"🚀 CHECKPOINT 20: Restored directory to {original_dir}")
            # Remove added paths from sys.path
            if str(original_dir) in sys.path:
                sys.path.remove(str(original_dir))
            if str(isaac_gym_envs_path) in sys.path:
                sys.path.remove(str(isaac_gym_envs_path))
