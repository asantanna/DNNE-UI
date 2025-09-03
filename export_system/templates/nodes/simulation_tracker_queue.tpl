# Get logger from globals
from framework.globals import Global as g, dnne_logging
from framework.exceptions import CauseExitException
from framework.time_utils import parse_duration
import time
import statistics

# Robotics subsystem logger
robotics_logger = dnne_logging.getLogger("robotics")

class SimulationTracker_{NODE_ID}(QueueNode):
    """
    Simulation Tracker - Tracks training progress for simulations.
    Monitors episodes, loss metrics, and custom metrics via telemetry.
    All inputs are optional - connect only what you need to track.
    """
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.node_id = node_id
        self.node_logger = robotics_logger  # Use robotics subsystem logger
        
        # Set up input and output queues
        # All inputs are optional - connect what you need to track
        self.setup_inputs(required=[], 
                         optional=["step_done", "episode_done", "loss", "custom_metrics"])
        self.setup_outputs([])  # No outputs - all metrics go through telemetry
        
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
    
    
    async def compute(self, step_done=None, episode_done=None, 
                     loss=None, custom_metrics=None):
        """
        Track simulation progress and report telemetry.
        """
        # Initialize timing on first call
        if self.start_time is None:
            self.start_time = time.time()
            self.last_episode_time = self.start_time
        
        # Track step completion for step-based intervals
        step_completed = step_done is not None
        if step_completed:
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
        
        # Episode completed - ANY value on episode_done input triggers episode end
        episode_completed = False
        if episode_done is not None:
            episode_completed = True
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
            should_report = self._should_report_telemetry(step_completed, episode_completed)
            if should_report:
                self._report_telemetry()
        
        # Check if training is done (for internal tracking)
        if episode_completed:
            training_done = self._check_training_done()
            if training_done:
                print(f"\n🎯 SIMULATION COMPLETE! Episode {self.episode_count}\n")
                raise CauseExitException(f"Training completed after {self.episode_count} episodes")
        
        # No outputs - all metrics sent via telemetry
        return {}
    
    def _should_report_telemetry(self, step_completed, episode_completed):
        """Determine if telemetry should be reported based on configured mode and interval."""
        if self.telemetry_level == "off":
            return False
        
        # Mode-specific checks
        if self.telemetry_mode == 'time':
            time_elapsed = time.time() - self.telemetry_last_report_time
            return time_elapsed >= self.telemetry_interval
        
        elif self.telemetry_mode == 'steps':
            # Only count when step_done signal is received
            if step_completed:
                steps_elapsed = self.timestep_count - self.telemetry_last_report_step
                return steps_elapsed >= self.telemetry_interval
            return False
        
        elif self.telemetry_mode == 'episodes':
            # Only count when episode_done signal is received
            if episode_completed:
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
    
    def _check_training_done(self):
        """
        Check if training should stop based on various criteria.
        """
        # Calculate success rate if we have data
        success_rate = 0.0
        if self.episode_successes:
            recent_successes = self.episode_successes[-self.window_size:]
            success_rate = sum(recent_successes) / len(recent_successes)
        
        # Check max episodes
        if self.episode_count >= self.max_episodes:
            robotics_logger.info(f"SimulationTracker: Reached max episodes ({self.max_episodes})")
            return True
        
        # Check success threshold
        if success_rate >= self.success_threshold and self.episode_count >= self.window_size:
            robotics_logger.info(f"SimulationTracker: Reached success threshold ({success_rate:.2%} >= {self.success_threshold:.2%})")
            return True
        
        # Check for convergence (no improvement in last N episodes)
        convergence_window = min(500, self.max_episodes // 10)
        if self.episode_count - self.last_improvement_episode > convergence_window:
            robotics_logger.info(f"SimulationTracker: Converged (no improvement in {convergence_window} episodes)")
            return True
        
        return False
    
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