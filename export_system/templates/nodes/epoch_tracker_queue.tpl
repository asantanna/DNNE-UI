# Template variables - replaced during export
template_vars = {
    "NODE_ID": "epoch_tracker_1",
    "MAX_EPOCHS": 100
}

from framework.globals import Global as g, dnne_logging
from framework.exceptions import CauseExitException
from framework import telemetry
import statistics
import time

# Training subsystem logger
training_logger = dnne_logging.getLogger("training")

class EpochTrackerNode_{NODE_ID}(QueueNode):
    """Tracks training progress across epochs and displays statistics"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["epoch_stats", "loss", "accuracy"])
        self.setup_outputs(["control_metrics"])
        
        # Training statistics
        self.current_epoch = 0
        self.epoch_losses = []
        self.epoch_accuracies = []
        self.batch_count = 0
        
        # Check for epochs override from command line
        epochs_override = g.get_node_config(self.node_id, 'epochs', None)
        if epochs_override is not None:
            self.total_epochs = epochs_override
            self.node_logger.info(f"Using epochs override from node config: {self.total_epochs}")
        elif g.epochs_override is not None:
            # Fallback to global override for backward compatibility
            self.total_epochs = g.epochs_override
            self.node_logger.info(f"Using global epochs override: {self.total_epochs}")
        else:
            self.total_epochs = {MAX_EPOCHS}
        
        # Flag to track if we've shown the training starting message
        self.training_started = False
        
        # Telemetry configuration - only initialize if telemetry is enabled
        self.telemetry_enabled = g.get_node_config(self.node_id, 'telemetry_enabled', False)
        if self.telemetry_enabled:
            # Only create buffers and config if telemetry is actually enabled
            # Support both batch-based and time-based windows
            self.telemetry_batch_window = g.get_node_config(self.node_id, 'telemetry_batch_window', None)
            self.telemetry_time_window = g.get_node_config(self.node_id, 'telemetry_time_window', None)
            
            # Default to 100 batches if neither specified
            if self.telemetry_batch_window is None and self.telemetry_time_window is None:
                self.telemetry_batch_window = 100
                
            self.telemetry_loss_buffer = []
            self.telemetry_accuracy_buffer = []
            self.telemetry_window_counter = 0
            self.telemetry_last_report_time = time.time()
            
            if self.telemetry_time_window:
                self.node_logger.info(f"Telemetry enabled with time window: {self.telemetry_time_window} seconds")
            elif self.telemetry_batch_window:
                self.node_logger.info(f"Telemetry enabled with batch window: {self.telemetry_batch_window} batches")
            else:
                self.node_logger.info(f"Telemetry enabled")
        
    async def compute(self, epoch_stats, loss, accuracy) -> Dict[str, Any]:
        # Show training starting message once
        if not self.training_started:
            print(f"\n🚀 Training starting... ({self.total_epochs} epochs)\n")
            self.training_started = True
        
        # Track batch-level metrics
        loss_value = loss.item() if hasattr(loss, 'item') else float(loss)
        accuracy_value = float(accuracy)
        self.epoch_losses.append(loss_value)
        self.epoch_accuracies.append(accuracy_value)
        self.batch_count += 1
        
        # Collect telemetry data if enabled (no overhead when disabled)
        if self.telemetry_enabled:
            self.telemetry_loss_buffer.append(loss_value)
            self.telemetry_accuracy_buffer.append(accuracy_value)
            self.telemetry_window_counter += 1
            
            # Check if we should report based on window type
            should_report = False
            current_time = time.time()
            
            # Time-based window check
            if self.telemetry_time_window:
                time_elapsed = current_time - self.telemetry_last_report_time
                if time_elapsed >= self.telemetry_time_window and len(self.telemetry_loss_buffer) > 0:
                    should_report = True
            
            # Batch-based window check
            elif self.telemetry_batch_window:
                if self.telemetry_window_counter >= self.telemetry_batch_window and len(self.telemetry_loss_buffer) > 0:
                    should_report = True
            
            if should_report:
                self._report_window_telemetry()
                self.telemetry_last_report_time = current_time
        
        # Check if epoch completed
        if epoch_stats.get("completed", False):
            # Calculate epoch averages
            avg_loss = sum(self.epoch_losses) / len(self.epoch_losses)
            avg_accuracy = sum(self.epoch_accuracies) / len(self.epoch_accuracies)
            
            # Display epoch summary (always show, regardless of verbose mode)
            epoch_num = epoch_stats["epoch"]
            print(f"📊 EPOCH {epoch_num + 1}/{self.total_epochs} COMPLETE")
            print(f"   Batches: {len(self.epoch_losses)}")
            print(f"   Avg Loss: {avg_loss:.4f}")
            print(f"   Avg Accuracy: {avg_accuracy:.2%}")
            print("=" * 60)
            
            # Report epoch telemetry if enabled
            if self.telemetry_enabled:
                telemetry.report_custom(self.node_id, "epoch_completed", float(epoch_num + 1))
                telemetry.report_custom(self.node_id, "epoch_avg_loss", avg_loss)
                telemetry.report_custom(self.node_id, "epoch_avg_accuracy", avg_accuracy)
                telemetry.report_custom(self.node_id, "epoch_total_batches", float(len(self.epoch_losses)))
                
            # Reset for next epoch
            self.epoch_losses = []
            self.epoch_accuracies = []
            self.current_epoch = epoch_num + 1
            
            # Check if training should stop
            training_done = self.current_epoch >= self.total_epochs
            if training_done:
                print(f"\n🎯 TRAINING COMPLETE! Reached {self.total_epochs} epochs\n")
                # Raise exception to stop the graph runner
                raise CauseExitException(f"Training completed after {self.total_epochs} epochs")
            
            # Build control metrics dictionary
            control_metrics = {
                # Core control
                "epoch": epoch_num + 1,
                "done": training_done,
                
                # Training metrics
                "batch_count": self.batch_count,
                "avg_loss": avg_loss,
                "avg_accuracy": avg_accuracy,
            }
            
            return {"control_metrics": control_metrics}
        else:
            # Show batch progress only in verbose mode
            if g.verbose:
                progress = epoch_stats.get("progress", 0)
                if self.batch_count % 10 == 0:  # Show progress every 10 batches
                    training_logger.info(f"Epoch {epoch_stats['epoch'] + 1} - Batch {epoch_stats['batch']}/{epoch_stats['total_batches']} ({progress:.1%}) - Loss: {self.epoch_losses[-1]:.4f}, Acc: {self.epoch_accuracies[-1]:.2%}")
            
            # Return None for control_metrics during the epoch (not complete yet)
            return {"control_metrics": None}
    
    def _report_window_telemetry(self):
        """Report telemetry statistics for the current window of batches"""
        if not self.telemetry_enabled or len(self.telemetry_loss_buffer) == 0:
            return
            
        # Calculate statistics for loss
        losses = self.telemetry_loss_buffer
        loss_mean = statistics.mean(losses)
        loss_min = min(losses)
        loss_max = max(losses)
        loss_std = statistics.stdev(losses) if len(losses) > 1 else 0.0
        
        # Calculate percentiles for loss
        if len(losses) >= 4:
            loss_quartiles = statistics.quantiles(losses, n=4)
            loss_p25 = loss_quartiles[0]
            loss_p50 = loss_quartiles[1]  # median
            loss_p75 = loss_quartiles[2]
        else:
            # Not enough data for quartiles, use simple approximations
            loss_p25 = loss_min
            loss_p50 = statistics.median(losses)
            loss_p75 = loss_max
        
        # Calculate statistics for accuracy
        accs = self.telemetry_accuracy_buffer
        acc_mean = statistics.mean(accs)
        acc_min = min(accs)
        acc_max = max(accs)
        acc_std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        
        # Calculate percentiles for accuracy
        if len(accs) >= 4:
            acc_quartiles = statistics.quantiles(accs, n=4)
            acc_p25 = acc_quartiles[0]
            acc_p50 = acc_quartiles[1]  # median
            acc_p75 = acc_quartiles[2]
        else:
            # Not enough data for quartiles, use simple approximations
            acc_p25 = acc_min
            acc_p50 = statistics.median(accs)
            acc_p75 = acc_max
        
        # Report all loss statistics
        telemetry.report_custom(self.node_id, "train_loss_mean", loss_mean)
        telemetry.report_custom(self.node_id, "train_loss_min", loss_min)
        telemetry.report_custom(self.node_id, "train_loss_max", loss_max)
        telemetry.report_custom(self.node_id, "train_loss_std", loss_std)
        telemetry.report_custom(self.node_id, "train_loss_p25", loss_p25)
        telemetry.report_custom(self.node_id, "train_loss_p50", loss_p50)
        telemetry.report_custom(self.node_id, "train_loss_p75", loss_p75)
        
        # Report all accuracy statistics
        telemetry.report_custom(self.node_id, "train_acc_mean", acc_mean)
        telemetry.report_custom(self.node_id, "train_acc_min", acc_min)
        telemetry.report_custom(self.node_id, "train_acc_max", acc_max)
        telemetry.report_custom(self.node_id, "train_acc_std", acc_std)
        telemetry.report_custom(self.node_id, "train_acc_p25", acc_p25)
        telemetry.report_custom(self.node_id, "train_acc_p50", acc_p50)
        telemetry.report_custom(self.node_id, "train_acc_p75", acc_p75)
        
        # Report meta information
        telemetry.report_custom(self.node_id, "train_window_size", float(len(losses)))
        telemetry.report_custom(self.node_id, "train_total_batches", float(self.batch_count))
        telemetry.report_custom(self.node_id, "train_current_epoch", float(self.current_epoch + 1))
        
        # Clear buffers for next window
        self.telemetry_loss_buffer = []
        self.telemetry_accuracy_buffer = []
        self.telemetry_window_counter = 0