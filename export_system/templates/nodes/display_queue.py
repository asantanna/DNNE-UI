# Template variables - replaced during export
template_vars = {
    "NODE_ID": "display_1",
    "CLASS_NAME": "DisplayNode",
    "DISPLAY_TYPE": "tensor_stats",
    "LOG_INTERVAL": 10
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Display/logging node"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["input_0"])
        self.setup_outputs([])  # No outputs
        
        self.display_type = "{DISPLAY_TYPE}"
        self.display_count = 0
        self.log_interval = {LOG_INTERVAL}
    
    async def compute(self, input_0) -> Dict[str, Any]:
        self.display_count += 1
        
        # Only log at intervals
        if self.display_count % self.log_interval == 0:
            if self.display_type == "tensor_stats" and hasattr(input_0, 'shape'):
                self.logger.info(f"[{{self.display_count}}] Tensor shape: {{input_0.shape}}, "
                              f"min: {{input_0.min().item():.4f}}, "
                              f"max: {{input_0.max().item():.4f}}, "
                              f"mean: {{input_0.mean().item():.4f}}")
            elif self.display_type == "value":
                self.logger.info(f"[{{self.display_count}}] Value: {{input_0}}")
            else:
                self.logger.info(f"[{{self.display_count}}] {{type(input_0)}}")
        
        return {{}}  # No outputs
