# Template variables - replaced during export
from framework.globals import Global as g, dnne_logging
from framework.exceptions import TrainingCompleteException

# Training subsystem logger
training_logger = dnne_logging.getLogger("training")

class EpochTrackerNode_{NODE_ID}(QueueNode):
    """Tracks training progress across epochs and displays statistics"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["epoch_stats", "loss", "accuracy"])
        self.setup_outputs(["training_summary"])
        
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
        
    async def compute(self, epoch_stats, loss, accuracy) -> Dict[str, Any]:
        # Show training starting message once
        if not self.training_started:
            print(f"\n🚀 Training starting... ({self.total_epochs} epochs)\n")
            self.training_started = True
        
        # Track batch-level metrics
        self.epoch_losses.append(loss.item() if hasattr(loss, 'item') else float(loss))
        self.epoch_accuracies.append(float(accuracy))
        self.batch_count += 1
        
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
            
            # Reset for next epoch
            summary = {
                "epoch": epoch_num,
                "avg_loss": avg_loss,
                "avg_accuracy": avg_accuracy,
                "batches": len(self.epoch_losses),
                "completed": True
            }
            
            self.epoch_losses = []
            self.epoch_accuracies = []
            self.current_epoch = epoch_num + 1
            
            # Check if training should stop
            if self.current_epoch >= self.total_epochs:
                print(f"\n🎯 TRAINING COMPLETE! Reached {self.total_epochs} epochs\n")
                summary["training_complete"] = True
                
                # Raise exception to stop the graph runner
                raise TrainingCompleteException(f"Training completed after {self.total_epochs} epochs")
            
            return {"training_summary": summary}
        else:
            # Show batch progress only in verbose mode
            if g.verbose:
                progress = epoch_stats.get("progress", 0)
                if self.batch_count % 10 == 0:  # Show progress every 10 batches
                    training_logger.info(f"Epoch {epoch_stats['epoch'] + 1} - Batch {epoch_stats['batch']}/{epoch_stats['total_batches']} ({progress:.1%}) - Loss: {self.epoch_losses[-1]:.4f}, Acc: {self.epoch_accuracies[-1]:.2%}")
            
            return {"training_summary": None}