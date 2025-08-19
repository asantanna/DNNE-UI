# Template variables - replaced during export
template_vars = {
    "NODE_ID": "or_1",
    "CLASS_NAME": "ORNode"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """OR/ANY Router node - outputs when ANY input becomes available"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)  # No wait_mode parameter
        # OR node has optional inputs only - any input triggers output
        self.setup_inputs(required=[], optional=["input_a", "input_b", "input_c"], queue_size=2)
        self.setup_outputs(["output"])
        
        # State tracking
        self.last_input_source = None
        self.output_count = 0
        
    async def run(self):
        """Custom run method: execute when ANY input becomes available"""
        import time
        self.running = True
        self.node_logger.info(f"Starting OR node {self.node_id}")
        
        try:
            while self.running:
                # Wait for ANY input using MultiWaiter (returns dict with single key)
                input_dict = await self.input_waiter.get()
                
                # Extract the single input (there should be exactly one)
                input_name = list(input_dict.keys())[0]
                input_data = input_dict[input_name]
                
                # Execute compute with the available input
                start_time = time.time()
                outputs = await self.compute_single_input(input_name, input_data)
                self.last_compute_time = time.time() - start_time
                self.compute_count += 1
                
                # Send outputs
                for output_name, value in outputs.items():
                    await self.send_output(output_name, value)
                    
        except asyncio.CancelledError:
            self.node_logger.info(f"OR Node {self.node_id} cancelled")
            raise
        finally:
            self.running = False
    
    async def compute(self, **inputs) -> Dict[str, Any]:
        """Required by abstract base class - not used since we override run()"""
        # This method is required by QueueNode abstract base class
        # but not actually called since we override run() method
        raise NotImplementedError("OR node uses custom run() method, not compute()")
    
    async def compute_single_input(self, input_name: str, input_data) -> Dict[str, Any]:
        """Handle single input for OR node"""
        import torch
        
        # Route the input based on which one arrived
        self.last_input_source = input_name.upper()
        self.output_count += 1
        
        shape_info = input_data.shape if hasattr(input_data, 'shape') else 'unknown'
        self.node_logger.info(f"OR Node: Routing {input_name} (shape: {shape_info}) - output #{self.output_count}")
        
        return {
            "output": input_data
        }