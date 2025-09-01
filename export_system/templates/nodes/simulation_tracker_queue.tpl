# Get logger from globals
from framework.globals import Global as g, dnne_logging
from framework.time_utils import parse_duration
import time
import statistics

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
        # Required: observation, loss (must be present every timestep)
        # Optional: done, reward, custom_metrics (episodic or additional metrics)
        self.setup_inputs(required=["observation", "loss"], 
                         optional=["done", "reward", "custom_metrics"])
        self.setup_outputs(["control_metrics"])
        
        self.max_episodes = {MAX_EPISODES}
        self.success_threshold = {SUCCESS_THRESHOLD}
        
        # Telemetry reporting configuration
        self.telemetry_mode = {TELEMETRY_MODE}  # 'time', 'steps', or 'episodes'
        self.telemetry_interval_str = {TELEMETRY_INTERVAL}
        self.telemetry_stats = {TELEMETRY_STATS}
        
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
            # Parse telemetry interval based on mode
            if self.telemetry_mode == 'time':
                self.telemetry_interval = parse_duration(self.telemetry_interval_str)
                self.node_logger.info(f"SimulationTracker_{NODE_ID}: Telemetry enabled with {self.telemetry_mode} interval: {self.telemetry_interval:.1f}s")
            else:
                try:
                    self.telemetry_interval = int(self.telemetry_interval_str)
                except ValueError:
                    self.telemetry_interval = 100  # Default fallback
                    self.node_logger.warning(f"Invalid interval '{self.telemetry_interval_str}' for {self.telemetry_mode} mode, using default: 100")
                self.node_logger.info(f"SimulationTracker_{NODE_ID}: Telemetry enabled with {self.telemetry_mode} interval: {self.telemetry_interval}")
            
            # Telemetry buffers for aggregation
            self.telemetry_loss_buffer = []
            self.telemetry_reward_buffer = []
            self.telemetry_episode_reward_buffer = []
            self.telemetry_episode_length_buffer = []
            
            # Telemetry counters
            self.telemetry_last_report_time = time.time()
            self.telemetry_last_report_step = 0
            self.telemetry_last_report_episode = 0
        
        # Track first observation time
        self.start_time = None
        self.last_episode_time = None
    
    
    async def compute(self, observation=None, loss=None, 
                     done=None, reward=None, custom_metrics=None):
        """
        Track simulation progress and compute control metrics.
        """
        # Initialize timing on first observation
        if self.start_time is None:
            self.start_time = time.time()
            self.last_episode_time = self.start_time
        
        # Update timestep count only if observation is present
        if observation is not None:
            self.timestep_count += 1
            self.current_episode_length += 1
        
        # Accumulate reward
        if reward is not None:
            reward_value = float(reward)
            self.current_episode_reward += reward_value
            
            # Buffer reward for telemetry aggregation
            if self.telemetry_enabled:
                self.telemetry_reward_buffer.append(reward_value)
        
        # Track loss (only if present)
        if loss is not None:
            loss_value = float(loss)
            self.losses.append(loss_value)
            
            # Buffer loss for telemetry aggregation
            if self.telemetry_enabled:
                self.telemetry_loss_buffer.append(loss_value)
        
        # Episode completed - ANY value on done input triggers episode end
        # (done is None if no signal received)
        episode_done = False
        if done is not None:
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
            
            # Buffer episode metrics for telemetry
            if self.telemetry_enabled:
                self.telemetry_episode_reward_buffer.append(self.current_episode_reward)
                self.telemetry_episode_length_buffer.append(self.current_episode_length)
            
            # Reset for next episode
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
        
        # Check if we should report telemetry
        if self.telemetry_enabled:
            should_report = self._should_report_telemetry(episode_done)
            if should_report:
                self._report_telemetry()
        
        # Compute control metrics
        control_metrics = self._compute_control_metrics(episode_done)
        
        # Output control metrics
        await self.send_output("control_metrics", control_metrics)
        
        return control_metrics
    
    def _should_report_telemetry(self, episode_done):
        """Determine if telemetry should be reported based on configured mode and interval."""
        if not self.telemetry_enabled:
            return False
        
        # Mode-specific checks
        if self.telemetry_mode == 'time':
            time_elapsed = time.time() - self.telemetry_last_report_time
            return time_elapsed >= self.telemetry_interval
        
        elif self.telemetry_mode == 'steps':
            steps_elapsed = self.timestep_count - self.telemetry_last_report_step
            return steps_elapsed >= self.telemetry_interval
        
        elif self.telemetry_mode == 'episodes':
            # Report on episode completion and check interval
            if episode_done:
                episodes_elapsed = self.episode_count - self.telemetry_last_report_episode
                return episodes_elapsed >= self.telemetry_interval
            return False
        
        return False
    
    def _report_telemetry(self):
        """Report aggregated telemetry statistics."""
        if not self.telemetry_enabled:
            return
        
        from framework import telemetry
        
        # Update last report markers
        current_time = time.time()
        time_window = current_time - self.telemetry_last_report_time
        self.telemetry_last_report_time = current_time
        self.telemetry_last_report_step = self.timestep_count
        self.telemetry_last_report_episode = self.episode_count
        
        # Report timing info
        telemetry.report_custom(self.node_id, "report_time_window", time_window)
        telemetry.report_custom(self.node_id, "report_timestep", float(self.timestep_count))
        telemetry.report_custom(self.node_id, "report_episode", float(self.episode_count))
        
        # Report loss statistics
        if self.telemetry_loss_buffer:
            losses = self.telemetry_loss_buffer
            telemetry.report_custom(self.node_id, "loss_samples", float(len(losses)))
            
            if self.telemetry_stats:
                # Calculate statistics
                loss_mean = statistics.mean(losses)
                loss_min = min(losses)
                loss_max = max(losses)
                
                telemetry.report_custom(self.node_id, "loss_mean", loss_mean)
                telemetry.report_custom(self.node_id, "loss_min", loss_min)
                telemetry.report_custom(self.node_id, "loss_max", loss_max)
                
                if len(losses) > 1:
                    loss_std = statistics.stdev(losses)
                    telemetry.report_custom(self.node_id, "loss_std", loss_std)
                
                # Percentiles if enough data
                if len(losses) >= 4:
                    quartiles = statistics.quantiles(losses, n=4)
                    telemetry.report_custom(self.node_id, "loss_p25", quartiles[0])
                    telemetry.report_custom(self.node_id, "loss_p50", quartiles[1])
                    telemetry.report_custom(self.node_id, "loss_p75", quartiles[2])
            else:
                # Just report the latest value
                telemetry.report_custom(self.node_id, "loss_latest", losses[-1])
            
            # Clear buffer
            self.telemetry_loss_buffer = []
        
        # Report reward statistics
        if self.telemetry_reward_buffer:
            rewards = self.telemetry_reward_buffer
            telemetry.report_custom(self.node_id, "reward_samples", float(len(rewards)))
            
            if self.telemetry_stats:
                reward_mean = statistics.mean(rewards)
                reward_min = min(rewards)
                reward_max = max(rewards)
                
                telemetry.report_custom(self.node_id, "reward_mean", reward_mean)
                telemetry.report_custom(self.node_id, "reward_min", reward_min)
                telemetry.report_custom(self.node_id, "reward_max", reward_max)
                
                if len(rewards) > 1:
                    reward_std = statistics.stdev(rewards)
                    telemetry.report_custom(self.node_id, "reward_std", reward_std)
            else:
                telemetry.report_custom(self.node_id, "reward_latest", rewards[-1])
            
            # Clear buffer
            self.telemetry_reward_buffer = []
        
        # Report episode statistics
        if self.telemetry_episode_reward_buffer:
            ep_rewards = self.telemetry_episode_reward_buffer
            ep_lengths = self.telemetry_episode_length_buffer
            
            telemetry.report_custom(self.node_id, "episodes_completed", float(len(ep_rewards)))
            
            if self.telemetry_stats and ep_rewards:
                # Episode reward stats
                telemetry.report_custom(self.node_id, "episode_reward_mean", statistics.mean(ep_rewards))
                telemetry.report_custom(self.node_id, "episode_reward_min", min(ep_rewards))
                telemetry.report_custom(self.node_id, "episode_reward_max", max(ep_rewards))
                
                # Episode length stats
                if ep_lengths:
                    telemetry.report_custom(self.node_id, "episode_length_mean", statistics.mean(ep_lengths))
                    telemetry.report_custom(self.node_id, "episode_length_min", float(min(ep_lengths)))
                    telemetry.report_custom(self.node_id, "episode_length_max", float(max(ep_lengths)))
            else:
                # Just report latest
                if ep_rewards:
                    telemetry.report_custom(self.node_id, "episode_reward_latest", ep_rewards[-1])
                if ep_lengths:
                    telemetry.report_custom(self.node_id, "episode_length_latest", float(ep_lengths[-1]))
            
            # Clear buffers
            self.telemetry_episode_reward_buffer = []
            self.telemetry_episode_length_buffer = []
        
        # Report success rate if we have data
        if self.episode_successes:
            recent_successes = self.episode_successes[-self.window_size:]
            success_rate = sum(recent_successes) / len(recent_successes)
            telemetry.report_custom(self.node_id, "success_rate", success_rate)
    
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
            # if self.telemetry_enabled:
            #     robotics_logger.info(f"SimulationTracker_{NODE_ID}: Reached max episodes ({self.max_episodes})")
        
        # Check success threshold
        if success_rate >= self.success_threshold and self.episode_count >= self.window_size:
            training_done = True
            # if self.telemetry_enabled:
            #     robotics_logger.info(f"SimulationTracker_{NODE_ID}: Reached success threshold ({success_rate:.2%} >= {self.success_threshold:.2%})")
        
        # Check for convergence (no improvement in last N episodes)
        convergence_window = min(500, self.max_episodes // 10)
        if self.episode_count - self.last_improvement_episode > convergence_window:
            training_done = True
            # if self.telemetry_enabled:
            #     robotics_logger.info(f"SimulationTracker_{NODE_ID}: Converged (no improvement in {convergence_window} episodes)")
        
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