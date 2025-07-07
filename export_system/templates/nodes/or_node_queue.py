# Template variables - replaced during export
template_vars = {
    "NODE_ID": "or_1",
    "CLASS_NAME": "ORNode"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """OR/ANY Router node - outputs when ANY input becomes available"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=[], optional=["input_a", "input_b", "input_c"])
        self.setup_outputs(["output"])
        
        # State tracking
        self.last_input_source = None
        self.output_count = 0
        
    async def compute(self, input_a=None, input_b=None, input_c=None) -> Dict[str, Any]:
        """Route the first available input to output"""
        import torch
        
        # Check inputs in order of priority (A, B, C)
        if input_a is not None:
            self.last_input_source = "A"
            self.output_count += 1
            self.logger.info(f"OR Node: Routing input A (shape: {{input_a.shape if hasattr(input_a, 'shape') else 'unknown'}}) - output #{{self.output_count}}")
            return {{
                "output": input_a
            }}
        
        elif input_b is not None:
            self.last_input_source = "B"
            self.output_count += 1
            self.logger.info(f"OR Node: Routing input B (shape: {{input_b.shape if hasattr(input_b, 'shape') else 'unknown'}}) - output #{{self.output_count}}")
            return {{
                "output": input_b
            }}
        
        elif input_c is not None:
            self.last_input_source = "C"
            self.output_count += 1
            self.logger.info(f"OR Node: Routing input C (shape: {{input_c.shape if hasattr(input_c, 'shape') else 'unknown'}}) - output #{{self.output_count}}")
            return {{
                "output": input_c
            }}
        
        else:
            # No inputs available - this shouldn't happen in normal operation
            # In async queue system, this node should only execute when at least one input is available
            self.logger.error("OR Node: No inputs available - this should not happen in queue system")
            raise RuntimeError("OR Node: No inputs available")