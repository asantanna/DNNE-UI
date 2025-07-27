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
def DNNE_print_local(level, component, message):
    """Local debug print function"""
    print(f"[DNNE_DEBUG] {level}/{component}: {message}", flush=True)

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """
    PPO Agent Node that runs IsaacGymEnvs training
    Consolidates environment and PPO configuration from virtual nodes
    """
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=[])
        self.setup_outputs(["metrics"])
        
        # Node configuration
        self.network_mlp_layers = {NETWORK_MLP_LAYERS}
        self.network_activation = "{NETWORK_ACTIVATION}"
        self.separate_value_network = {SEPARATE_VALUE_NETWORK}
        self.checkpoint_interval = {CHECKPOINT_INTERVAL}
        self.keep_checkpoints = {KEEP_CHECKPOINTS}
        self.load_checkpoint = "{LOAD_CHECKPOINT}"
        self.log_interval = {LOG_INTERVAL}
        self.save_interval = {SAVE_INTERVAL}
        self.experiment_name = "{EXPERIMENT_NAME}"
        self.mixed_precision = {MIXED_PRECISION}
        self.multi_gpu = {MULTI_GPU}
        
        # Environment configuration from virtual node
        self.env_config = {{
            'task': '{ENV_TASK}',
            'num_envs': {ENV_NUM_ENVS},
            'seed': {ENV_SEED},
            'headless': {ENV_HEADLESS},
            'graphics_device_id': {ENV_GRAPHICS_DEVICE},
            'sim_device': '{ENV_SIM_DEVICE}',
            'physics_engine': '{ENV_PHYSICS_ENGINE}',
            'multi_gpu': {ENV_MULTI_GPU},
            'enable_cameras': {ENV_ENABLE_CAMERAS},
            'force_render': {ENV_FORCE_RENDER},
            'use_gpu_pipeline': {ENV_USE_GPU_PIPELINE},
            'num_threads': {ENV_NUM_THREADS},
            'solver_type': {ENV_SOLVER_TYPE},
            'num_subscenes': {ENV_NUM_SUBSCENES},
        }}
        
        # Validate task selection
        if self.env_config['task'] == 'none':
            raise ValueError(
                "Invalid environment task: 'none'. "
                "Please select a valid task from the dropdown in the IsaacGymEnvs node before exporting."
            )
        
        # PPO configuration from virtual node
        self.ppo_config = {{
            'minibatch_size': {PPO_MINIBATCH_SIZE},
            'horizon_length': {PPO_HORIZON_LENGTH},
            'learning_rate': {PPO_LEARNING_RATE},
            'schedule_type': '{PPO_SCHEDULE_TYPE}',
            'gamma': {PPO_GAMMA},
            'tau': {PPO_TAU},
            'e_clip': {PPO_E_CLIP},
            'clip_value': {PPO_CLIP_VALUE},
            'mini_epochs': {PPO_MINI_EPOCHS},
            'critic_coef': {PPO_CRITIC_COEF},
            'entropy_coef': {PPO_ENTROPY_COEF},
            'bounds_loss_coef': {PPO_BOUNDS_LOSS_COEF},
            'max_epochs': {PPO_MAX_EPOCHS},
            'normalize_advantage': {PPO_NORMALIZE_ADVANTAGE},
            'normalize_input': {PPO_NORMALIZE_INPUT},
            'normalize_value': {PPO_NORMALIZE_VALUE},
        }}
        
        # Check for node-specific overrides
        max_iterations_override = Global.get_node_config(self.node_id, 'max_iterations', None)
        if max_iterations_override is not None:
            self.ppo_config['max_epochs'] = max_iterations_override
            self.logger.info(f"Using max_iterations override from node config: {max_iterations_override}")
        
        # IsaacGymEnvs path
        self.isaac_gym_envs_path = "{ISAAC_GYM_ENVS_PATH}"
        
        # Track if training has been completed
        self.training_completed = False
    
    async def run(self):
        """Override run to execute training only once"""
        self.running = True
        self.logger.info(f"Starting PPO training node {self.node_id}")
        DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 1: run() method started")
        
        try:
            # Run training once
            DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 2: About to call compute()")
            outputs = await self.compute()
            DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 3: compute() returned")
            self.compute_count += 1
            self.last_compute_time = time.time()
            
            # Send outputs to any connected nodes
            DNNE_print_local("D", "PPO_AGENT", f"🚀 CHECKPOINT 4: Sending {len(outputs)} outputs")
            for output_name, output_data in outputs.items():
                await self.send_output(output_name, output_data)
            
            self.logger.info(f"PPO training completed for node {self.node_id}")
            DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 5: Training marked as completed")
            self.training_completed = True
            
            # Keep the node alive but idle
            DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 6: Entering idle loop")
            while self.running:
                await asyncio.sleep(1.0)
                
        except asyncio.CancelledError:
            self.logger.info(f"Node {self.node_id} cancelled")
            DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT ERROR: AsyncIO cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in node {self.node_id}: {e}")
            DNNE_print_local("D", "PPO_AGENT", f"🚀 CHECKPOINT ERROR: Exception {type(e).__name__}: {e}")
            raise
    
    async def compute(self):
        """
        Run IsaacGymEnvs train.py with consolidated configuration
        """
        DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 7: compute() method started")
        
        # Create configuration for train.py
        DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 8: Creating train config")
        train_config = self._create_train_config()
        DNNE_print_local("D", "PPO_AGENT", f"🚀 CHECKPOINT 9: Train config created with {len(train_config)} args")
        
        # Run IsaacGymEnvs train.py
        DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 10: About to call _run_training()")
        metrics = await self._run_training(train_config)
        DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 11: _run_training() returned")
        
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
            DNNE_print_local("D", "PPO_AGENT", "🖼️  Visual mode enabled - launching with GUI")
        elif headless_mode:
            # Headless mode: disable GUI and force_render
            headless = True
            force_render = False  # Override force_render in headless mode
            DNNE_print_local("D", "PPO_AGENT", "🖥️  Headless mode enforced - disabling force_render")
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
        
        DNNE_print_local("D", "PPO_AGENT", f"🚀 Full train config ({len(all_args)} args):")
        for i, arg in enumerate(all_args):
            DNNE_print_local("D", "PPO_AGENT", f"  [{i}] {arg}")
        
        return all_args
    
    async def _run_training(self, train_config):
        """
        Run IsaacGymEnvs train.py with configuration
        This uses subprocess to maintain isolation and proper environment setup
        """
        DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 12: _run_training() started")
        
        # Change to IsaacGymEnvs directory
        isaac_gym_envs_path = Path(self.isaac_gym_envs_path)
        if not isaac_gym_envs_path.exists():
            raise RuntimeError(f"IsaacGymEnvs path not found: {isaac_gym_envs_path}")
        
        # Save current directory
        original_dir = os.getcwd()
        DNNE_print_local("D", "PPO_AGENT", f"🚀 CHECKPOINT 13: Original dir: {original_dir}")
        
        try:
            # Add original directory to Python path so rl_games_dnne can find framework module
            if str(original_dir) not in sys.path:
                sys.path.insert(0, str(original_dir))
                DNNE_print_local("D", "PPO_AGENT", f"Added {original_dir} to sys.path for framework imports")
            
            # Change to IsaacGymEnvs/isaacgymenvs directory where train.py lives
            DNNE_print_local("D", "PPO_AGENT", f"🚀 CHECKPOINT 14: Changing to {isaac_gym_envs_path / 'isaacgymenvs'}")
            os.chdir(isaac_gym_envs_path / "isaacgymenvs")
            
            # Debug: Print current directory to verify we're in the right place
            DNNE_print_local("D", "PPO_AGENT", f"Current directory: {os.getcwd()}")
            DNNE_print_local("D", "PPO_AGENT", f"Files in directory: {os.listdir('.')[:10]}")
            
            # Set up Hydra configuration path
            os.environ['HYDRA_FULL_ERROR'] = '1'
            
            # Enable DNNE adaptive yielding for cooperative execution
            os.environ['DNNE_ADAPTIVE_YIELD'] = '1'
            DNNE_print_local("D", "PPO_AGENT", f"🚀 CHECKPOINT 14.5: DNNE_ADAPTIVE_YIELD={os.environ.get('DNNE_ADAPTIVE_YIELD', 'NOT SET')}")
            
            # Convert args to sys.argv format for hydra
            sys.argv = ["train.py"] + train_config
            DNNE_print_local("D", "PPO_AGENT", f"🚀 CHECKPOINT 15: sys.argv set to {len(sys.argv)} args")
            
            # Use runpy to execute train.py as __main__
            # This should make Hydra resolve paths correctly
            import runpy
            
            # Run training with periodic yielding
            DNNE_print_local("D", "PPO_AGENT", f"Starting PPO training with IsaacGymEnvs...")
            DNNE_print_local("D", "PPO_AGENT", f"Configuration: {' '.join(train_config)}")
            
            # Check if event loop is accessible before running
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                DNNE_print_local("D", "PPO_AGENT", f"Event loop accessible before runpy: {loop}")
            except RuntimeError as e:
                DNNE_print_local("D", "PPO_AGENT", f"No event loop before runpy: {e}")
            
            # Run training directly - it will yield cooperatively
            DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 16: About to call runpy.run_path()")
            DNNE_print_local("D", "PPO_AGENT", f"🚀 CHECKPOINT 16.1: train.py exists: {os.path.exists('train.py')}")
            DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 16.2: runpy.run_path('train.py', run_name='__main__') starting NOW...")
            result = runpy.run_path("train.py", run_name="__main__")
            DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 17: runpy.run_path() RETURNED!!! (This probably won't print)")
            
            # Extract metrics from result
            metrics = {
                "training_complete": True,
                "final_reward": result.get("final_reward", 0.0),
                "total_steps": result.get("total_steps", 0),
                "training_time": result.get("training_time", 0.0),
            }
            
            DNNE_print_local("D", "PPO_AGENT", f"🚀 CHECKPOINT 18: Metrics extracted: {metrics}")
            return metrics
            
        finally:
            DNNE_print_local("D", "PPO_AGENT", "🚀 CHECKPOINT 19: In finally block")
            # Restore original directory
            os.chdir(original_dir)
            DNNE_print_local("D", "PPO_AGENT", f"🚀 CHECKPOINT 20: Restored directory to {original_dir}")
            # Remove added paths from sys.path
            if str(original_dir) in sys.path:
                sys.path.remove(str(original_dir))
            if str(isaac_gym_envs_path) in sys.path:
                sys.path.remove(str(isaac_gym_envs_path))
    