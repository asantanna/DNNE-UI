# Template variables - replaced during export
template_vars = {
    "NODE_ID": "geometric_loss_1",
    "CLASS_NAME": "GeometricLossNode",
    "ERROR_METRIC": "Euclidean Dist"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Geometric Loss node - computes distance/divergence metrics between predictions and estimates"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(["predictions", "estimates"])
        self.setup_outputs(["output"])
        
        # Configuration from template
        self.error_metric = "{ERROR_METRIC}"
        
        # Statistics tracking
        self.loss_history = []
        self.compute_count = 0
        
    async def compute(self, predictions, estimates) -> Dict[str, Any]:
        """Compute geometric loss based on selected metric"""
        import torch
        from framework.math_utils import (
            max_abs_error, euclidean_distance, manhattan_distance, 
            kl_divergence
        )
        
        # Ensure tensors are on same device and flatten for computation
        predictions_flat = predictions.flatten()
        estimates_flat = estimates.flatten()
        
        if self.error_metric == "Max Abs Error":
            loss = max_abs_error(predictions_flat, estimates_flat)
            self.node_logger.info(f"Max Abs Error: {loss.item():.6f}")
            
        elif self.error_metric == "Euclidean Dist":
            loss = euclidean_distance(predictions_flat, estimates_flat)
            self.node_logger.info(f"Euclidean Distance: {loss.item():.6f}")
            
        elif self.error_metric == "Manhattan Dist":
            loss = manhattan_distance(predictions_flat, estimates_flat)
            self.node_logger.info(f"Manhattan Distance: {loss.item():.6f}")
            
        elif self.error_metric == "KL Div":
            loss = kl_divergence(predictions_flat, estimates_flat, normalize=False)
            self.node_logger.info(f"KL Divergence: {loss.item():.6f}")
            
        elif self.error_metric == "Norm KL Div":
            loss = kl_divergence(predictions_flat, estimates_flat, normalize=True)
            # Get the unnormalized KL for logging
            kl_unnorm = kl_divergence(predictions_flat, estimates_flat, normalize=False)
            n = predictions_flat.numel()
            if n > 1:
                max_kl = torch.log(torch.tensor(n, dtype=torch.float32))
                self.node_logger.info(f"Normalized KL Div: {loss.item():.6f} (KL={kl_unnorm.item():.4f}, max={max_kl.item():.4f})")
            else:
                self.node_logger.info(f"Normalized KL Div: 0.0 (single element)")
            
        else:
            raise ValueError(f"Unknown error metric: {self.error_metric}")
        
        # Track statistics
        self.loss_history.append(loss.item())
        self.compute_count += 1
        
        # Log every 10 computations
        if self.compute_count % 10 == 0:
            recent_losses = self.loss_history[-10:]
            avg_loss = sum(recent_losses) / len(recent_losses)
            self.node_logger.info(
                f"Geometric Loss Stats - Metric: {self.error_metric}, "
                f"Count: {self.compute_count}, Recent avg: {avg_loss:.6f}"
            )
        
        return {
            "output": loss
        }