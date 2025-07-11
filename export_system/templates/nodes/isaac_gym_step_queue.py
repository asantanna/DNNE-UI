# Template variables - replaced during export
template_vars = {
    "NODE_ID": "isaac_step_1",
    "CLASS_NAME": "IsaacGymStepNode"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Isaac Gym step node with dual-mode execution using clean environment classes"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["sim_handle", "actions", "trigger"])
        self.setup_outputs(["observations", "rewards", "done", "info", "next_observations"])
        
        # Step tracking
        self.step_count = 0
        
        # State caching for RL training loop
        self.cached_observations = None
        self.cached_rewards = None
        self.cached_done = None
        self.cached_info = None
        
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
        self.logger.info(f"🕐 Real-time timing: target_dt={target_dt}s ({1/target_dt:.1f}Hz)")
        
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
                
                # Send outputs
                if outputs:
                    for output_name, value in outputs.items():
                        await self.send_output(output_name, value)
                
                # Real-time timing: sleep to maintain target dt
                elapsed = time.time() - loop_start_time
                sleep_time = max(0, target_dt - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    # Log if we're running behind real-time
                    if self.step_count % 100 == 0:  # Log every 100 steps
                        self.logger.warning(f"Behind real-time: {elapsed:.4f}s > {target_dt:.4f}s (target)")
                        
            except Exception as e:
                self.logger.error(f"Inference mode error: {e}")
                await asyncio.sleep(target_dt)  # Maintain timing even on errors
    
    async def compute(self, sim_handle, actions, trigger=None) -> Dict[str, Any]:
        """Execute one simulation step using environment class methods"""
        try:
            import torch
            
            # Validate simulation handle
            if not hasattr(sim_handle, 'environment') or sim_handle.environment is None:
                raise RuntimeError("Invalid simulation handle or environment not initialized")
            
            environment = sim_handle.environment
            num_envs = environment.num_envs
            
            # Handle trigger-based next_observations output (for RL training)
            if trigger is not None:
                # Output cached observations when triggered by PPOTrainer
                next_observations = self.cached_observations if self.cached_observations is not None else torch.zeros(num_envs, 4)
                self.logger.debug(f"Trigger received, outputting cached observations: shape={next_observations.shape}")
                return {
                    "observations": torch.zeros(num_envs, 4),  # observations (dummy)
                    "rewards": torch.zeros(num_envs),           # rewards (dummy)  
                    "done": torch.zeros(num_envs, dtype=torch.bool),  # done (dummy)
                    "info": {},                               # info (dummy)
                    "next_observations": next_observations      # next_observations (cached)
                }
            
            # Normal execution: step simulation using environment class
            if actions is not None:
                # Step environment with actions
                observations, rewards, done, info = environment.step_simulation(actions)
            else:
                # No actions - just get current state
                observations = environment.get_observations()
                rewards = environment.compute_rewards()
                done = environment.check_termination()
                info = {"step_count": self.step_count}
            
            # Update viewer if available
            if hasattr(sim_handle, 'viewer') and sim_handle.viewer is not None:
                environment.update_viewer(sim_handle.viewer)
            
            # Cache results for later trigger-based output
            self.cached_observations = observations
            self.cached_rewards = rewards
            self.cached_done = done
            self.cached_info = info
            
            # Update step counter
            self.step_count += 1
            
            # Log progress periodically
            if self.step_count % 1000 == 0:
                avg_reward = torch.mean(rewards).item() if len(rewards) > 0 else 0.0
                self.logger.info(f"Step {self.step_count}: avg reward = {avg_reward:.4f}")
            
            return {
                "observations": observations,              # Current step observations
                "rewards": rewards,                       # Current step rewards
                "done": done,                            # Current step done flags
                "info": info,                            # Current step info
                "next_observations": torch.zeros(num_envs, 4)  # next_observations (empty until triggered)
            }
            
        except Exception as e:
            self.logger.error(f"Error in simulation step: {e}")
            raise