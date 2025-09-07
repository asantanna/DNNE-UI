# Template variables - replaced during export
template_vars = {
    "NODE_ID": "sequencer_1",
    "CONNECTED_LOSSES": [1, 2],  # Which loss inputs are connected
    "OPTIMIZER_NODE_IDS": ["40", "81"],  # Corresponding optimizer node IDs
    "ORDER": [2, 1],  # Execution order
    "RETAIN_GRAPH": True
}

import asyncio
from typing import Dict, Any
from framework.globals import Global as g

class TrainingSequencer_{NODE_ID}(QueueNode):
    """Orchestrates training for multiple optimizers to prevent gradient conflicts"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        
        # Only connected losses are required
        connected_losses = {CONNECTED_LOSSES}
        required_inputs = [f"loss{i}" for i in connected_losses]
        self.setup_inputs(required=required_inputs)
        
        # Outputs to optimizers
        self.setup_outputs(["to_opt1", "to_opt2", "to_opt3", "to_opt4"])
        
        # Configuration
        self.connected_losses = connected_losses
        self.order = {ORDER}
        self.retain_graph = {RETAIN_GRAPH}
        
        # Optimizer connections (resolved in run())
        self.optimizer_node_ids = {OPTIMIZER_NODE_IDS}
        self.optimizers = []
        
    async def run(self):
        """Resolve optimizer connections before starting"""
        self.running = True
        self.node_logger.info(f"Starting TrainingSequencer {self.node_id}")
        
        try:
            # Resolve optimizer nodes
            for opt_id in self.optimizer_node_ids:
                opt_node = g.graph_runner.get_node(opt_id)
                if not opt_node:
                    raise RuntimeError(f"Optimizer node {opt_id} not found")
                
                # Verify it has required methods
                required_methods = ['zero_grad_only', 'backward_only', 'step_only', 'get_parameters']
                for method in required_methods:
                    if not hasattr(opt_node, method):
                        raise RuntimeError(
                            f"Optimizer {opt_id} missing method '{method}'. "
                            f"Ensure SGDOptimizer template includes TrainingSequencer support."
                        )
                
                self.optimizers.append(opt_node)
                self.node_logger.info(f"Connected to optimizer {opt_id}")
            
            if len(self.optimizers) != len(self.connected_losses):
                raise RuntimeError(
                    f"Mismatch: {len(self.connected_losses)} losses connected but "
                    f"{len(self.optimizers)} optimizers found"
                )
            
            self.node_logger.info(
                f"Sequencer managing {len(self.optimizers)} optimizers. "
                f"Order: {self.order}, retain_graph: {self.retain_graph}"
            )
            
            # Run normal compute loop
            await super().run()
            
        except asyncio.CancelledError:
            self.node_logger.info(f"TrainingSequencer {self.node_id} cancelled")
            raise
        finally:
            self.running = False
    
    async def compute(self, **kwargs) -> Dict[str, Any]:
        """All required losses arrive together - orchestrate backward passes"""
        
        # Map loss inputs to optimizer indices (1-based)
        losses = {}
        for i in self.connected_losses:
            loss_key = f"loss{i}"
            if loss_key in kwargs:
                losses[i] = kwargs[loss_key]
        
        if len(losses) != len(self.connected_losses):
            # This shouldn't happen with required inputs, but fail-fast
            raise RuntimeError(
                f"Expected {len(self.connected_losses)} losses, got {len(losses)}"
            )
        
        # Process losses in specified order
        for step_idx, loss_idx in enumerate(self.order):
            if loss_idx not in losses:
                self.node_logger.warning(f"Skipping loss{loss_idx} - not in current batch")
                continue
            
            # Map loss index to optimizer index (both use same position in connected_losses)
            opt_position = self.connected_losses.index(loss_idx)
            optimizer = self.optimizers[opt_position]
            loss = losses[loss_idx]
            
            # Disable gradients for all OTHER optimizers' parameters
            for other_idx, other_opt in enumerate(self.optimizers):
                if other_idx != opt_position:
                    for param in other_opt.get_parameters():
                        param.requires_grad_(False)
            
            # Zero gradients for THIS optimizer
            optimizer.zero_grad_only()
            
            # Backward pass with appropriate retain_graph
            # All but last need retain_graph=True
            is_last = (step_idx == len(self.order) - 1)
            retain = self.retain_graph and not is_last
            
            if g.verbose:
                self.node_logger.debug(
                    f"Backward pass {step_idx + 1}/{len(self.order)}: "
                    f"loss{loss_idx} → optimizer {optimizer.node_id}, "
                    f"retain_graph={retain}"
                )
            
            optimizer.backward_only(loss, retain_graph=retain)
            
            # Re-enable gradients for all parameters
            for other_idx, other_opt in enumerate(self.optimizers):
                if other_idx != opt_position:
                    for param in other_opt.get_parameters():
                        param.requires_grad_(True)
        
        # After all backward passes, step all optimizers
        for optimizer in self.optimizers:
            optimizer.step_only()
        
        # Send signals to optimizers (they may use these for logging/tracking)
        import time
        outputs = {}
        for i, opt in enumerate(self.optimizers):
            output_key = f"to_opt{self.connected_losses[i]}"
            outputs[output_key] = {
                "signal_type": "training_complete",
                "timestamp": time.time(),
                "source_node": self.node_id,
                "optimizer_id": opt.node_id
            }
        
        if g.verbose:
            self.node_logger.info(f"Sequenced training complete for {len(self.optimizers)} optimizers")
        
        return outputs