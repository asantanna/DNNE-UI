# Template variables - replaced during export
template_vars = {
    "NODE_ID": "concat_1",
    "CLASS_NAME": "ConcatNode",
    "MODE": "wait for all",
    "PAD_MODE": "pad with zeros"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Concat node - concatenates multiple tensor inputs with configurable synchronization"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        # Special setup: Concat node creates input queues but doesn't require all inputs
        self.setup_inputs(required=[])  # No required inputs
        self.setup_outputs(["output"])
        
        # Manually create input queues for Concat node
        self.input_queues["input_a"] = asyncio.Queue(maxsize=2)
        self.input_queues["input_b"] = asyncio.Queue(maxsize=2)
        self.input_queues["input_c"] = asyncio.Queue(maxsize=2)
        self.input_queues["input_d"] = asyncio.Queue(maxsize=2)
        
        # Configuration from template
        self.mode = "{MODE}"
        self.pad_mode = "{PAD_MODE}"
        
        # State tracking for "hold previous" mode
        self.previous_values = {}
        
        # Track which inputs are actually connected
        self.connected_inputs = []
        
    def set_connections(self, connections: Dict[str, List]):
        """Override to track which inputs are connected"""
        super().set_connections(connections)
        # Determine which inputs are actually connected
        self.connected_inputs = []
        for input_name in ["input_a", "input_b", "input_c", "input_d"]:
            if input_name in self.connections and self.connections[input_name]:
                self.connected_inputs.append(input_name)
        self.node_logger.info(f"Connected inputs: {self.connected_inputs}")
        
    async def run(self):
        """Custom run method based on configured mode"""
        import asyncio
        import time
        self.running = True
        self.node_logger.info(f"Starting Concat node {self.node_id} in '{self.mode}' mode")
        
        try:
            while self.running:
                if self.mode == "wait for all":
                    await self.run_wait_for_all()
                else:  # "as available"
                    await self.run_as_available()
                    
        except asyncio.CancelledError:
            self.node_logger.info(f"Concat Node {self.node_id} cancelled")
            raise
        finally:
            self.running = False
    
    async def run_wait_for_all(self):
        """Wait for all connected inputs before concatenating"""
        import torch
        import time
        
        # Wait for all connected inputs
        input_data = {}
        for input_name in self.connected_inputs:
            data = await self.input_queues[input_name].get()
            input_data[input_name] = data
            self.previous_values[input_name] = data  # Store for potential later use
        
        # Concatenate all inputs
        start_time = time.time()
        tensors_to_concat = [input_data[name] for name in sorted(self.connected_inputs)]
        
        if tensors_to_concat:
            concatenated = torch.cat(tensors_to_concat, dim=0)
            self.last_compute_time = time.time() - start_time
            self.compute_count += 1
            
            shape_info = f"{concatenated.shape}"
            self.node_logger.info(f"Concat (wait for all): Combined {len(tensors_to_concat)} tensors -> shape {shape_info}")
            
            # Send output
            await self.send_output("output", concatenated)
    
    async def run_as_available(self):
        """Output immediately when any input arrives, padding missing inputs"""
        import torch
        import asyncio
        import time
        
        # Create tasks for all connected inputs
        input_tasks = []
        for input_name in self.connected_inputs:
            task = asyncio.create_task(
                self.input_queues[input_name].get(), 
                name=input_name
            )
            input_tasks.append(task)
        
        # Wait for first available input
        done, pending = await asyncio.wait(input_tasks, return_when=asyncio.FIRST_COMPLETED)
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
        
        # Process the completed input
        completed_task = list(done)[0]
        new_data = completed_task.result()
        input_name = completed_task.get_name()
        
        # Update previous values
        self.previous_values[input_name] = new_data
        
        # Build concatenation list with padding for missing inputs
        start_time = time.time()
        tensors_to_concat = []
        for name in sorted(self.connected_inputs):
            if name in self.previous_values:
                # Use the stored value (either just received or previous)
                tensors_to_concat.append(self.previous_values[name])
            else:
                # Need to pad this input
                if self.pad_mode == "pad with zeros":
                    # Create zero tensor matching the shape of the new data
                    # Assume all inputs have same shape except batch dimension
                    zero_tensor = torch.zeros_like(new_data)
                    tensors_to_concat.append(zero_tensor)
                    self.node_logger.debug(f"Padding {name} with zeros (shape: {zero_tensor.shape})")
                else:  # "hold previous"
                    # Skip if we don't have previous data yet
                    self.node_logger.debug(f"No previous data for {name}, skipping")
                    continue
        
        if tensors_to_concat:
            concatenated = torch.cat(tensors_to_concat, dim=0)
            self.last_compute_time = time.time() - start_time
            self.compute_count += 1
            
            shape_info = f"{concatenated.shape}"
            self.node_logger.info(f"Concat (as available): Updated from {input_name} -> shape {shape_info}")
            
            # Send output
            await self.send_output("output", concatenated)
    
    async def compute(self, **inputs) -> Dict[str, Any]:
        """Required by abstract base class - not used since we override run()"""
        # This method is required by QueueNode abstract base class
        # but not actually called since we override run() method
        raise NotImplementedError("Concat node uses custom run() method, not compute()")