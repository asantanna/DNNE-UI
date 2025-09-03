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
        # Optional: done, custom_metrics (episodic or additional metrics)
        self.setup_inputs(required=["observation", "loss"], 
                         optional=["done", "custom_metrics"])
        self.setup_outputs(["control_metrics"])
        
        self.max_episodes = {MAX_EPISODES}
        self.success_threshold = {SUCCESS_THRESHOLD}
        
        # Telemetry reporting configuration
        self.telemetry_interval_str = {TELEMETRY_INTERVAL}  # e.g., "100_steps", "10_episodes", "30s"
        self.telemetry_level = {TELEMETRY_LEVEL}  # "off", "essential", "extended", "debug"
        
        # Episode tracking
        self.episode_count = 0
        self.timestep_count = 0
        self.current_episode_length = 0
        self.current_episode_loss_sum = 0.0  # Track loss per episode
        
        # Performance tracking
        self.episode_losses = []  # Average loss per episode
        self.episode_lengths = []
        self.episode_successes = []
        self.losses = []  # All loss values
        
        # Running statistics
        self.window_size = 100  # For rolling averages
        self.best_loss = float('inf')  # Lower is better for loss
        self.last_improvement_episode = 0
        
        # Telemetry configuration
        # Allow runtime override via --override
        self.telemetry_level = g.get_node_config(self.node_id, 'telemetry_level', self.telemetry_level)
        
        # Parse telemetry interval using simplified format
        self.telemetry_mode = None  # Will be set based on interval format
        self.telemetry_interval = None
        
        if self.telemetry_level != "off":
            # Parse interval format: "100_steps", "10_episodes", "30s", "5m"
            if '_' in self.telemetry_interval_str:
                parts = self.telemetry_interval_str.split('_')
                try:
                    self.telemetry_interval = int(parts[0])
                    self.telemetry_mode = parts[1]  # "steps" or "episodes"
                except (ValueError, IndexError):
                    self.telemetry_interval = 100
                    self.telemetry_mode = "steps"
                    self.node_logger.warning(f"Invalid interval format '{self.telemetry_interval_str}', using default: 100_steps")
            else:
                # Time-based interval (e.g., "30s", "5m")
                self.telemetry_mode = "time"
                self.telemetry_interval = parse_duration(self.telemetry_interval_str)
            
            self.node_logger.info(f"SimulationTracker telemetry: level={self.telemetry_level}, interval={self.telemetry_interval_str}")
            
            # Telemetry buffers for aggregation
            self.telemetry_loss_buffer = []
            self.telemetry_episode_loss_buffer = []
            self.telemetry_episode_length_buffer = []
            self.telemetry_episode_success_buffer = []
            
            # Telemetry counters
            self.telemetry_last_report_time = time.time()
            self.telemetry_last_report_step = 0
            self.telemetry_last_report_episode = 0
        
        # Track first observation time
        self.start_time = None
        self.last_episode_time = None
    
    
    async def compute(self, observation=None, loss=None, 
                     done=None, custom_metrics=None):
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
        
        # Accumulate loss for episode average
        if loss is not None:
            loss_value = float(loss)
            self.current_episode_loss_sum += loss_value
        
        # Track loss (only if present)
        if loss is not None:
            loss_value = float(loss)
            self.losses.append(loss_value)
            
            # Buffer loss for telemetry aggregation
            if self.telemetry_level != "off":
                self.telemetry_loss_buffer.append(loss_value)
        
        # Episode completed - ANY value on done input triggers episode end
        # (done is None if no signal received)
        episode_done = False
        if done is not None:
            episode_done = True
            self.episode_count += 1
            
            # Calculate average loss for the episode
            avg_episode_loss = self.current_episode_loss_sum / max(1, self.current_episode_length)
            
            # Record episode statistics
            self.episode_losses.append(avg_episode_loss)
            self.episode_lengths.append(self.current_episode_length)
            
            # Check for improvement (lower loss is better)
            if avg_episode_loss < self.best_loss:
                self.best_loss = avg_episode_loss
                self.last_improvement_episode = self.episode_count
            
            # Determine success (task-specific, from custom_metrics)
            success = False
            if custom_metrics and "success" in custom_metrics:
                success = bool(custom_metrics["success"])
            # Could also use loss threshold: success = avg_episode_loss < threshold
            self.episode_successes.append(success)
            
            # Buffer episode metrics for telemetry
            if self.telemetry_level != "off":
                self.telemetry_episode_loss_buffer.append(avg_episode_loss)
                self.telemetry_episode_length_buffer.append(self.current_episode_length)
                self.telemetry_episode_success_buffer.append(success)
            
            # Reset for next episode
            self.current_episode_loss_sum = 0.0
            self.current_episode_length = 0
        
        # Check if we should report telemetry
        if self.telemetry_level != "off":
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
        if self.telemetry_level == "off":
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
        if self.telemetry_level == "off":
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
            
            if self.telemetry_level in ["extended", "debug"]:
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
        
        # Report episode loss statistics (essential metric)
        
        # Report episode statistics
        if self.telemetry_episode_loss_buffer:
            ep_losses = self.telemetry_episode_loss_buffer
            ep_lengths = self.telemetry_episode_length_buffer
            ep_successes = self.telemetry_episode_success_buffer
            
            # Essential metrics
            telemetry.report_custom(self.node_id, "episodes_completed", float(len(ep_losses)))
            telemetry.report_custom(self.node_id, "loss_mean", statistics.mean(ep_losses))
            telemetry.report_custom(self.node_id, "timesteps_total", float(self.timestep_count))
            
            if self.telemetry_level in ["extended", "debug"]:
                # Extended metrics
                if ep_lengths:
                    telemetry.report_custom(self.node_id, "episode_length_mean", statistics.mean(ep_lengths))
                if ep_successes:
                    success_rate = sum(ep_successes) / len(ep_successes)
                    telemetry.report_custom(self.node_id, "success_rate", success_rate)
                if len(ep_losses) > 1:
                    telemetry.report_custom(self.node_id, "loss_std", statistics.stdev(ep_losses))
            
            # Clear buffers
            self.telemetry_episode_loss_buffer = []
            self.telemetry_episode_length_buffer = []
            self.telemetry_episode_success_buffer = []
        
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
        avg_loss = 0.0
        avg_length = 0.0
        success_rate = 0.0
        
        if self.episode_losses:
            recent_losses = self.episode_losses[-self.window_size:]
            avg_loss = sum(recent_losses) / len(recent_losses)
            
        if self.episode_lengths:
            recent_lengths = self.episode_lengths[-self.window_size:]
            avg_length = sum(recent_lengths) / len(recent_lengths)
            
        if self.episode_successes:
            recent_successes = self.episode_successes[-self.window_size:]
            success_rate = sum(recent_successes) / len(recent_successes)
        
        # Calculate improvement rate (negative is good for loss)
        improvement_rate = 0.0
        if len(self.episode_losses) > self.window_size:
            old_avg = sum(self.episode_losses[-2*self.window_size:-self.window_size]) / self.window_size
            new_avg = avg_loss
            if old_avg != 0:
                improvement_rate = (old_avg - new_avg) / abs(old_avg)  # Positive means loss decreased
        
        # Determine if training is done
        training_done = False
        
        # Check max episodes
        if self.episode_count >= self.max_episodes:
            training_done = True
            # if self.telemetry_level != "off":
            #     robotics_logger.info(f"SimulationTracker_{NODE_ID}: Reached max episodes ({self.max_episodes})")
        
        # Check success threshold
        if success_rate >= self.success_threshold and self.episode_count >= self.window_size:
            training_done = True
            # if self.telemetry_level != "off":
            #     robotics_logger.info(f"SimulationTracker_{NODE_ID}: Reached success threshold ({success_rate:.2%} >= {self.success_threshold:.2%})")
        
        # Check for convergence (no improvement in last N episodes)
        convergence_window = min(500, self.max_episodes // 10)
        if self.episode_count - self.last_improvement_episode > convergence_window:
            training_done = True
            # if self.telemetry_level != "off":
            #     robotics_logger.info(f"SimulationTracker_{NODE_ID}: Converged (no improvement in {convergence_window} episodes)")
        
        # Build control metrics dictionary
        control_metrics = {
            # Core control
            "episode": self.episode_count,
            "timestep": self.timestep_count,
            "done": training_done,
            
            # Episode metrics
            "episode_done": episode_done,
            "episode_loss": self.episode_losses[-1] if self.episode_losses else 0.0,
            "avg_loss": avg_loss,
            
            # Performance tracking
            "success_rate": success_rate,
            "improvement_rate": improvement_rate,
            "best_loss": self.best_loss,
            
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
            "episode_losses": self.episode_losses,
            "episode_lengths": self.episode_lengths,
            "episode_successes": self.episode_successes,
            "losses": self.losses,
            "best_loss": self.best_loss,
            "last_improvement_episode": self.last_improvement_episode,
        }
    
    def set_state(self, state):
        """Restore node state from checkpoint."""
        self.episode_count = state.get("episode_count", 0)
        self.timestep_count = state.get("timestep_count", 0)
        self.episode_losses = state.get("episode_losses", [])
        self.episode_lengths = state.get("episode_lengths", [])
        self.episode_successes = state.get("episode_successes", [])
        self.losses = state.get("losses", [])
        self.best_loss = state.get("best_loss", float('inf'))
        self.last_improvement_episode = state.get("last_improvement_episode", 0)