class SimulationTracker_{NODE_ID}(QueueNode):
    """
    Simulation Tracker - Tracks RL/robotics training progress.
    Monitors episodes, rewards, and performance metrics.
    """
    
    def __init__(self):
        super().__init__()
        self.node_id = "{NODE_ID}"
        self.max_episodes = {MAX_EPISODES}
        self.success_threshold = {SUCCESS_THRESHOLD}
        
        # Episode tracking
        self.episode_count = 0
        self.timestep_count = 0
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.losses = []
        
        # Running statistics
        self.window_size = 100  # For rolling averages
        self.best_reward = float('-inf')
        self.last_improvement_episode = 0
        
        # Telemetry
        self.telemetry_enabled = DNNE_globals.telemetry_enabled
        self.telemetry_client = None
        if self.telemetry_enabled:
            self.telemetry_client = DNNE_globals.telemetry_client
        
        # Track first observation time
        self.start_time = None
        self.last_episode_time = None
    
    def setup_inputs(self):
        """Set up input queues."""
        self.add_input_queue("observation", max_size=1)
        self.add_input_queue("done", max_size=1)
        self.add_input_queue("loss", max_size=1)
        self.add_input_queue("reward", max_size=1)
        self.add_input_queue("custom_metrics", max_size=1)
    
    def setup_outputs(self):
        """Set up output queues."""
        self.add_output_queue("control_metrics")
    
    async def compute(self, observation=None, done=None, loss=None, 
                     reward=None, custom_metrics=None):
        """
        Track simulation progress and compute control metrics.
        """
        # Initialize timing on first observation
        if self.start_time is None and observation is not None:
            self.start_time = time.time()
            self.last_episode_time = self.start_time
        
        # Update timestep count
        if observation is not None:
            self.timestep_count += 1
            self.current_episode_length += 1
        
        # Accumulate reward
        if reward is not None:
            self.current_episode_reward += float(reward)
        
        # Track loss
        if loss is not None:
            self.losses.append(float(loss))
            
            # Send loss telemetry
            if self.telemetry_enabled and self.telemetry_client:
                self.telemetry_client.send_metric(
                    "loss", self.node_id, float(loss)
                )
        
        # Episode completed
        episode_done = False
        if done is not None and done:
            episode_done = True
            self.episode_count += 1
            
            # Record episode statistics
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Check for improvement
            if self.current_episode_reward > self.best_reward:
                self.best_reward = self.current_episode_reward
                self.last_improvement_episode = self.episode_count
            
            # Determine success (task-specific, could be from custom_metrics)
            success = False
            if custom_metrics and "success" in custom_metrics:
                success = custom_metrics["success"]
            elif self.current_episode_reward > 0:  # Simple heuristic
                success = True
            self.episode_successes.append(success)
            
            # Send episode telemetry
            if self.telemetry_enabled and self.telemetry_client:
                current_time = time.time()
                episode_time = current_time - self.last_episode_time
                self.last_episode_time = current_time
                
                # Episode metrics
                self.telemetry_client.send_metric(
                    "episode_reward", self.node_id, self.current_episode_reward
                )
                self.telemetry_client.send_metric(
                    "episode_length", self.node_id, self.current_episode_length
                )
                self.telemetry_client.send_metric(
                    "episode_time", self.node_id, episode_time
                )
                self.telemetry_client.send_metric(
                    "episode_count", self.node_id, self.episode_count
                )
                
                # Success tracking
                if len(self.episode_successes) > 0:
                    recent_successes = self.episode_successes[-self.window_size:]
                    success_rate = sum(recent_successes) / len(recent_successes)
                    self.telemetry_client.send_metric(
                        "success_rate", self.node_id, success_rate
                    )
            
            # Reset for next episode
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
        
        # Compute control metrics
        control_metrics = self._compute_control_metrics(episode_done)
        
        # Output control metrics
        await self.output_queues["control_metrics"].put(control_metrics)
        
        return control_metrics
    
    def _compute_control_metrics(self, episode_done):
        """
        Compute control metrics for downstream nodes.
        """
        # Calculate running averages
        avg_reward = 0.0
        avg_length = 0.0
        success_rate = 0.0
        
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-self.window_size:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            
        if self.episode_lengths:
            recent_lengths = self.episode_lengths[-self.window_size:]
            avg_length = sum(recent_lengths) / len(recent_lengths)
            
        if self.episode_successes:
            recent_successes = self.episode_successes[-self.window_size:]
            success_rate = sum(recent_successes) / len(recent_successes)
        
        # Calculate improvement rate
        improvement_rate = 0.0
        if len(self.episode_rewards) > self.window_size:
            old_avg = sum(self.episode_rewards[-2*self.window_size:-self.window_size]) / self.window_size
            new_avg = avg_reward
            if old_avg != 0:
                improvement_rate = (new_avg - old_avg) / abs(old_avg)
        
        # Determine if training is done
        training_done = False
        
        # Check max episodes
        if self.episode_count >= self.max_episodes:
            training_done = True
            if self.telemetry_enabled:
                logger.info(f"SimulationTracker_{NODE_ID}: Reached max episodes ({self.max_episodes})")
        
        # Check success threshold
        if success_rate >= self.success_threshold and self.episode_count >= self.window_size:
            training_done = True
            if self.telemetry_enabled:
                logger.info(f"SimulationTracker_{NODE_ID}: Reached success threshold ({success_rate:.2%} >= {self.success_threshold:.2%})")
        
        # Check for convergence (no improvement in last N episodes)
        convergence_window = min(500, self.max_episodes // 10)
        if self.episode_count - self.last_improvement_episode > convergence_window:
            training_done = True
            if self.telemetry_enabled:
                logger.info(f"SimulationTracker_{NODE_ID}: Converged (no improvement in {convergence_window} episodes)")
        
        # Build control metrics dictionary
        control_metrics = {
            # Core control
            "episode": self.episode_count,
            "timestep": self.timestep_count,
            "done": training_done,
            
            # Episode metrics
            "episode_done": episode_done,
            "episode_reward": self.episode_rewards[-1] if self.episode_rewards else 0.0,
            "avg_reward": avg_reward,
            
            # Performance tracking
            "success_rate": success_rate,
            "improvement_rate": improvement_rate,
            "best_reward": self.best_reward,
            
            # Additional statistics
            "avg_episode_length": avg_length,
            "episodes_since_improvement": self.episode_count - self.last_improvement_episode,
        }
        
        # Add latest loss if available
        if self.losses:
            control_metrics["latest_loss"] = self.losses[-1]
            if len(self.losses) >= self.window_size:
                control_metrics["avg_loss"] = sum(self.losses[-self.window_size:]) / self.window_size
        
        return control_metrics
    
    def get_state(self):
        """Get node state for checkpointing."""
        return {
            "episode_count": self.episode_count,
            "timestep_count": self.timestep_count,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_successes": self.episode_successes,
            "losses": self.losses,
            "best_reward": self.best_reward,
            "last_improvement_episode": self.last_improvement_episode,
        }
    
    def set_state(self, state):
        """Restore node state from checkpoint."""
        self.episode_count = state.get("episode_count", 0)
        self.timestep_count = state.get("timestep_count", 0)
        self.episode_rewards = state.get("episode_rewards", [])
        self.episode_lengths = state.get("episode_lengths", [])
        self.episode_successes = state.get("episode_successes", [])
        self.losses = state.get("losses", [])
        self.best_reward = state.get("best_reward", float('-inf'))
        self.last_improvement_episode = state.get("last_improvement_episode", 0)