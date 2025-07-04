# Template variables - replaced during export

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
        self.total_epochs = {MAX_EPOCHS}
        
    async def compute(self, epoch_stats, loss, accuracy) -> Dict[str, Any]:
        # Track batch-level metrics
        self.epoch_losses.append(loss.item() if hasattr(loss, 'item') else float(loss))
        self.epoch_accuracies.append(float(accuracy))
        self.batch_count += 1
        
        # Check if epoch completed
        if epoch_stats.get("completed", False):
            # Calculate epoch averages
            avg_loss = sum(self.epoch_losses) / len(self.epoch_losses)
            avg_accuracy = sum(self.epoch_accuracies) / len(self.epoch_accuracies)
            
            # Display epoch summary
            epoch_num = epoch_stats["epoch"]
            self.logger.info("=" * 60)
            self.logger.info(f"📊 EPOCH {epoch_num} COMPLETE")
            self.logger.info(f"   Batches: {len(self.epoch_losses)}")
            self.logger.info(f"   Avg Loss: {avg_loss:.4f}")
            self.logger.info(f"   Avg Accuracy: {avg_accuracy:.2%}")
            self.logger.info("=" * 60)
            
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
                self.logger.info(f"🎯 TRAINING COMPLETE! Reached {self.total_epochs} epochs")
                summary["training_complete"] = True
            
            return {"training_summary": summary}
        else:
            # Show batch progress
            progress = epoch_stats.get("progress", 0)
            if self.batch_count % 10 == 0:  # Show progress every 10 batches
                self.logger.info(f"Epoch {epoch_stats['epoch']} - Batch {epoch_stats['batch']}/{epoch_stats['total_batches']} ({progress:.1%}) - Loss: {self.epoch_losses[-1]:.4f}, Acc: {self.epoch_accuracies[-1]:.2%}")
            
            return {"training_summary": None}