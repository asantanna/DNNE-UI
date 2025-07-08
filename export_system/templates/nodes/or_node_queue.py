# Template variables - replaced during export
template_vars = {
    "NODE_ID": "or_1",
    "CLASS_NAME": "ORNode"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """OR/ANY Router node - outputs when ANY input becomes available"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        # Special setup: OR node creates input queues but doesn't require all inputs
        self.setup_inputs(required=[])  # No required inputs
        self.setup_outputs(["output"])
        
        # Manually create input queues for OR node
        self.input_queues["input_a"] = asyncio.Queue(maxsize=2)
        self.input_queues["input_b"] = asyncio.Queue(maxsize=2)
        self.input_queues["input_c"] = asyncio.Queue(maxsize=2)
        
        # State tracking
        self.last_input_source = None
        self.output_count = 0
        
    async def run(self):
        """Custom run method: execute when ANY input becomes available"""
        import asyncio
        import time
        self.running = True
        self.logger.info(f"Starting OR node {self.node_id}")
        
        try:
            while self.running:
                # Wait for ANY input to become available
                input_tasks = [
                    asyncio.create_task(self.input_queues["input_a"].get(), name="input_a"),
                    asyncio.create_task(self.input_queues["input_b"].get(), name="input_b"),
                    asyncio.create_task(self.input_queues["input_c"].get(), name="input_c")
                ]
                
                # Wait for first available input
                done, pending = await asyncio.wait(input_tasks, return_when=asyncio.FIRST_COMPLETED)
                
                # Cancel remaining tasks
                for task in pending:
                    task.cancel()
                
                # Process the first completed input
                completed_task = list(done)[0]
                input_data = completed_task.result()
                input_name = completed_task.get_name()
                
                # Execute compute with the available input
                start_time = time.time()
                outputs = await self.compute_single_input(input_name, input_data)
                self.last_compute_time = time.time() - start_time
                self.compute_count += 1
                
                # Send outputs
                for output_name, value in outputs.items():
                    await self.send_output(output_name, value)
                    
        except asyncio.CancelledError:
            self.logger.info(f"OR Node {self.node_id} cancelled")
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
        self.logger.info(f"OR Node: Routing {input_name} (shape: {shape_info}) - output #{self.output_count}")
        
        return {
            "output": input_data
        }