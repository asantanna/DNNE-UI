"""Node implementation for IsaacGymStepNode (ID: 9)"""
import asyncio
import time
from typing import Dict, Any
import torch
import numpy as np
# Isaac Gym imports are handled at runtime in the template
from framework.base import QueueNode, SensorNode

# Template variables - replaced during export

class IsaacGymStepNode_9(QueueNode):
    """Isaac Gym step node with dual-mode execution using clean environment classes"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["sim_handle", "actions", "trigger"])
        self.setup_outputs(["observations", "rewards", "done", "info", "next_observations"])
        
        # Step tracking
        self.step_count = 0
        self.control_step_count = 0
        self.control_freq_inv = 16  # Number of physics steps per control step (match IsaacGymEnvs)
        self.physics_step_count = 0  # Track physics steps for render frequency
        self.last_render_time = 0.0
        self.render_interval = 1.0 / 60.0  # 60 FPS backup for viewer updates
        
        # Episode return tracking for learning metrics
        self.episode_returns = []  # Store cumulative returns for completed episodes
        self.current_episode_returns = None  # Cumulative returns for current episodes
        self.episode_count = 0
        self.last_n_episodes = 100  # Track last N episodes for averaging
        
    async def run(self):
        """Dual-mode execution: training vs inference timing"""
        self.running = True
        self.logger.info(f"Starting node {self.node_id}")
        
        # Check if we're in inference mode
        import builtins
        inference_mode = getattr(builtins, 'INFERENCE_MODE', False)
        
        try:
            # Wait for sim_handle first
            sim_handle = await self.input_queues["sim_handle"].get()
            self.logger.info(f"Received simulation handle: {sim_handle.environment.get_environment_name()}")
            
            if inference_mode:
                # Inference mode: Auto-trigger with real-time timing
                self.logger.info("🎮 Inference mode: Auto-triggering with real-time timing")
                await self._run_inference_mode(sim_handle)
            else:
                # Training mode: Trigger-based execution at maximum speed
                self.logger.info("🏃 Training mode: Trigger-based execution for maximum speed")
                await self._run_training_mode(sim_handle)
                
        except asyncio.CancelledError:
            self.logger.info(f"Node {self.node_id} cancelled")
            raise
        finally:
            self.running = False
    
    async def _run_training_mode(self, sim_handle):
        """Training mode: Trigger-based execution for maximum speed"""
        while self.running:
            try:
                # Wait for actions and trigger inputs
                actions = await self.input_queues["actions"].get()
                trigger = await self.input_queues["trigger"].get()
                
                # Execute computation 
                outputs = await self.compute(sim_handle, actions, trigger)
                self.compute_count += 1  # Track computation count
                
                # Send outputs immediately (no timing delay)
                if outputs:
                    for output_name, value in outputs.items():
                        await self.send_output(output_name, value)
                        
            except Exception as e:
                self.logger.error(f"Training mode error: {e}")
                await asyncio.sleep(0.1)
    
    async def _run_inference_mode(self, sim_handle):
        """Inference mode: Auto-trigger with real-time timing loop"""
        import time
        import asyncio
        
        # Get environment-specific simulation timing
        sim_params = sim_handle.environment.get_simulation_params()
        target_dt = sim_params.get("dt", 0.0166)  # Default to 60Hz
        
        # Check if viewer is enabled (only throttle for visual observation)
        viewer_enabled = hasattr(sim_handle, 'viewer') and sim_handle.viewer is not None
        use_realtime_throttling = viewer_enabled
        
        if use_realtime_throttling:
            self.logger.info(f"🕐 Real-time visual mode: target_dt={target_dt}s ({1/target_dt:.1f}Hz)")
        else:
            self.logger.info(f"🚀 Headless inference mode: Maximum speed (no throttling)")
        
        # In inference mode, we don't wait for triggers - we auto-generate them
        while self.running:
            try:
                loop_start_time = time.time()
                
                # Get latest actions (non-blocking with timeout)
                actions = None
                try:
                    actions = await asyncio.wait_for(self.input_queues["actions"].get(), timeout=0.001)
                except asyncio.TimeoutError:
                    # No new actions available, use None (maintain last actions in simulation)
                    pass
                
                # Clear any pending triggers (we don't use them in inference mode)
                try:
                    while not self.input_queues["trigger"].empty():
                        await asyncio.wait_for(self.input_queues["trigger"].get(), timeout=0.001)
                except (asyncio.TimeoutError, asyncio.QueueEmpty):
                    pass
                
                # Execute simulation step (no trigger in inference mode)
                outputs = await self.compute(sim_handle, actions, None)
                self.compute_count += 1  # Track computation count
                
                # Send outputs
                if outputs:
                    for output_name, value in outputs.items():
                        await self.send_output(output_name, value)
                
                # Smart throttling: Only when viewer is enabled for visual observation
                elapsed = time.time() - loop_start_time
                if use_realtime_throttling:
                    sleep_time = max(0, target_dt - elapsed)
                    
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                    else:
                        # Log if we're running behind real-time (for viewer)
                        if self.step_count % 100 == 0:  # Log every 100 steps
                            self.logger.warning(f"Behind real-time: {elapsed:.4f}s > {target_dt:.4f}s (target)")
                else:
                    # No throttling for headless inference - run at maximum speed
                    if self.step_count % 1000 == 0:  # Log every 1000 steps
                        effective_hz = 1.0 / elapsed if elapsed > 0 else float('inf')
                        self.logger.info(f"Headless performance: {elapsed*1000:.2f}ms per step ({effective_hz:.1f}Hz)")
                        
            except Exception as e:
                self.logger.error(f"Inference mode error: {e}")
                if use_realtime_throttling:
                    await asyncio.sleep(target_dt)  # Maintain timing for viewer
                else:
                    await asyncio.sleep(0.001)  # Minimal delay for stability
    
    async def compute(self, sim_handle, actions, trigger=None) -> Dict[str, Any]:
        """Execute one simulation step using environment class methods"""
        try:
            import torch
            
            # Validate simulation handle
            if not hasattr(sim_handle, 'environment') or sim_handle.environment is None:
                raise RuntimeError("Invalid simulation handle or environment not initialized")
            
            environment = sim_handle.environment
            num_envs = environment.num_envs
            
            # Always step simulation when we have actions
            if actions is not None:
                # Step environment with actions
                observations, rewards, done, info = environment.step_simulation(actions)
            else:
                # No actions - just get current state
                observations = environment.get_observations()
                rewards = environment.compute_rewards()
                done = environment.check_termination()
                info = {"step_count": self.step_count}
            
            # Update viewer if available (control-frequency based like IsaacGymEnvs)
            if hasattr(sim_handle, 'viewer') and sim_handle.viewer is not None:
                # Increment physics step counter
                self.physics_step_count += 1
                
                # Only render every control_freq_inv steps (like IsaacGymEnvs force_render pattern)
                if self.physics_step_count % self.control_freq_inv == 0:
                    # Additional time-based limiting as backup (60 FPS max)
                    import time
                    current_time = time.time()
                    if current_time - self.last_render_time >= self.render_interval:
                        environment.update_viewer(sim_handle.viewer)
                        self.last_render_time = current_time
            
            # Track episode returns for learning metrics
            self._update_episode_returns(rewards, done)
            
            # Update step counter
            self.step_count += 1
            
            # Log progress periodically
            if self.step_count % 1000 == 0:
                avg_reward = torch.mean(rewards).item() if len(rewards) > 0 else 0.0
                avg_return = self._get_average_episode_return()
                self.logger.info(f"Step {self.step_count}: avg reward = {avg_reward:.4f}, avg episode return = {avg_return:.1f}")
            
            # Return current step results
            # The trigger parameter is just for synchronization, not for changing behavior
            return {
                "observations": observations,              # Current step observations
                "rewards": rewards,                       # Current step rewards
                "done": done,                            # Current step done flags
                "info": info,                            # Current step info
                "next_observations": observations         # For PPO, next_obs = current obs
            }
            
        except Exception as e:
            self.logger.error(f"Error in simulation step: {e}")
            raise
    
    def _update_episode_returns(self, rewards, done):
        """Update episode return tracking with current step rewards and done flags"""
        import torch
        import numpy as np
        
        # Initialize current episode returns if needed (first step)
        if self.current_episode_returns is None:
            num_envs = len(rewards) if hasattr(rewards, '__len__') else 1
            self.current_episode_returns = torch.zeros(num_envs, device=rewards.device)
        
        # Accumulate rewards for all environments
        self.current_episode_returns += rewards
        
        # Check for completed episodes (done=True)
        if done.any():
            # Get completed episode indices
            completed_episodes = torch.where(done)[0]
            
            # Store returns for completed episodes
            for env_idx in completed_episodes:
                episode_return = self.current_episode_returns[env_idx].item()
                self.episode_returns.append(episode_return)
                self.episode_count += 1
                
                # Print episode return for profiler to capture
                print(f"Episode {self.episode_count}: episode return = {episode_return:.2f}")
                
                # Print rolling average every 10 episodes (for profiler capture)
                if self.episode_count % 10 == 0:
                    avg_return = self._get_average_episode_return()
                    print(f"Average episode return (last {len(self.episode_returns)} episodes) = {avg_return:.2f}")
                    print(f"avg episode return = {avg_return:.2f}")  # Profiler-friendly format
                
                # Keep only last N episodes for efficiency
                if len(self.episode_returns) > self.last_n_episodes:
                    self.episode_returns.pop(0)
                
                # Reset episode return for this environment
                self.current_episode_returns[env_idx] = 0.0
    
    def _get_average_episode_return(self):
        """Get average episode return from last N completed episodes"""
        if len(self.episode_returns) == 0:
            return 0.0
        import numpy as np
        return np.mean(self.episode_returns)
    
    def get_learning_metrics(self):
        """Get learning performance metrics for comparison"""
        return {
            "total_episodes": self.episode_count,
            "completed_episodes": len(self.episode_returns),
            "average_episode_return": self._get_average_episode_return(),
            "last_n_episodes": self.last_n_episodes,
            "episode_returns": self.episode_returns.copy() if self.episode_returns else []
        }
