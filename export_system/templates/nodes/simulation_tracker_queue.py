# Get logger from globals
from framework.globals import Global as g, dnne_logging

# Robotics subsystem logger
robotics_logger = dnne_logging.getLogger("robotics")

class SimulationTracker_{NODE_ID}(QueueNode):
    """
    Simulation Tracker - Tracks RL/robotics training progress.
    Monitors episodes, rewards, and performance metrics.
    """
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.node_id = node_id
        self.node_logger = robotics_logger  # Use robotics subsystem logger
        
        # Set up input and output queues
        # Required: observation, done, loss (core tracking inputs)
        # Optional: reward, custom_metrics (additional metrics)
        self.setup_inputs(required=["observation", "done", "loss"], 
                         optional=["reward", "custom_metrics"])
        self.setup_outputs(["control_metrics"])
        
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
        
        # Telemetry configuration
        self.telemetry_enabled = g.get_node_config(self.node_id, 'telemetry_enabled', False)
        if self.telemetry_enabled:
            self.node_logger.info(f"SimulationTracker_{NODE_ID}: Telemetry enabled")
        
        # Track first observation time
        self.start_time = None
        self.last_episode_time = None
    
    
    async def compute(self, observation, done, loss, 
                     reward=None, custom_metrics=None):
        """
        Track simulation progress and compute control metrics.
        """
        # Initialize timing on first observation
        if self.start_time is None:
            self.start_time = time.time()
            self.last_episode_time = self.start_time
        
        # Update timestep count (observation is always present)
        self.timestep_count += 1
        self.current_episode_length += 1
        
        # Accumulate reward
        if reward is not None:
            self.current_episode_reward += float(reward)
        
        # Track loss (always present)
        self.losses.append(float(loss))
        
        # Send loss telemetry
        if self.telemetry_enabled:
            from framework import telemetry
            telemetry.report_custom(self.node_id, "loss", float(loss))
        
        # Episode completed (done is always present)
        episode_done = False
        if done:
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
            if self.telemetry_enabled:
                from framework import telemetry
                current_time = time.time()
                episode_time = current_time - self.last_episode_time
                self.last_episode_time = current_time
                
                # Episode metrics
                telemetry.report_custom(self.node_id, "episode_reward", self.current_episode_reward)
                telemetry.report_custom(self.node_id, "episode_length", float(self.current_episode_length))
                telemetry.report_custom(self.node_id, "episode_time", episode_time)
                telemetry.report_custom(self.node_id, "episode_count", float(self.episode_count))
                
                # Success tracking
                if len(self.episode_successes) > 0:
                    recent_successes = self.episode_successes[-self.window_size:]
                    success_rate = sum(recent_successes) / len(recent_successes)
                    telemetry.report_custom(self.node_id, "success_rate", success_rate)
            
            # Reset for next episode
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
        
        # Compute control metrics
        control_metrics = self._compute_control_metrics(episode_done)
        
        # Output control metrics
        await self.send_output("control_metrics", control_metrics)
        
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
                robotics_logger.info(f"SimulationTracker_{NODE_ID}: Reached max episodes ({self.max_episodes})")
        
        # Check success threshold
        if success_rate >= self.success_threshold and self.episode_count >= self.window_size:
            training_done = True
            if self.telemetry_enabled:
                robotics_logger.info(f"SimulationTracker_{NODE_ID}: Reached success threshold ({success_rate:.2%} >= {self.success_threshold:.2%})")
        
        # Check for convergence (no improvement in last N episodes)
        convergence_window = min(500, self.max_episodes // 10)
        if self.episode_count - self.last_improvement_episode > convergence_window:
            training_done = True
            if self.telemetry_enabled:
                robotics_logger.info(f"SimulationTracker_{NODE_ID}: Converged (no improvement in {convergence_window} episodes)")
        
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