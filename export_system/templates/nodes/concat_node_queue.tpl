# Template variables - replaced during export
template_vars = {
    "NODE_ID": "concat_1",
    "CLASS_NAME": "ConcatNode",
    "MODE": "wait for all",
    "PAD_MODE": "pad with zeros",
    "CONCAT_DIM": 1  # Feature dimension per tensor standards
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Concat node - concatenates multiple tensor inputs along feature dimension"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)  # No wait_mode parameter
        
        # Setup inputs based on mode
        if "{MODE}" == "wait for all":
            # All inputs are required
            self.setup_inputs(required=["input_a", "input_b", "input_c", "input_d"], queue_size=2)
        else:
            # All inputs are optional ("as available" mode)
            self.setup_inputs(optional=["input_a", "input_b", "input_c", "input_d"], queue_size=2)
        
        self.setup_outputs(["output"])
        
        # Configuration from template
        self.mode = "{MODE}"
        self.pad_mode = "{PAD_MODE}"
        self.concat_dim = {CONCAT_DIM}  # Per tensor standards: dim=1 for features
        
        # State tracking for "hold previous" mode
        self.previous_values = {}
        
        # Track which inputs are actually connected
        self.connected_inputs = []
        
        # Get device from global configuration
        from framework.globals import Global as g
        self.device = torch.device(g.get_device())
        
    def set_connections(self, connections: Dict[str, List]):
        """Override to track which inputs are connected"""
        super().set_connections(connections)
        # Determine which inputs are actually connected
        self.connected_inputs = []
        for input_name in ["input_a", "input_b", "input_c", "input_d"]:
            if input_name in self.connections and self.connections[input_name]:
                self.connected_inputs.append(input_name)
        self.node_logger.info(f"Connected inputs: {self.connected_inputs}")
        
        # Update MultiWaiter to only use connected inputs if needed
        # Note: MultiWaiter was already created in __init__ by setup_inputs()
        # We may want to recreate it with only connected inputs for efficiency
        if self.connected_inputs and self.input_waiter:
            # Recreate with only connected inputs
            if "{MODE}" == "wait for all":
                # All connected inputs are required
                from framework import MultiWaiter
                self.input_waiter = MultiWaiter(
                    self.connected_inputs, [],
                    self.input_queues,
                    self.node_id
                )
            else:
                # All connected inputs are optional
                from framework import MultiWaiter
                self.input_waiter = MultiWaiter(
                    [], self.connected_inputs,
                    self.input_queues,
                    self.node_id
                )
        
    async def run(self):
        """Custom run method based on configured mode"""
        import asyncio
        import time
        self.running = True
        self.node_logger.info(f"Starting Concat node {self.node_id} in '{self.mode}' mode, dim={self.concat_dim}")
        
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
        
        # Wait for all connected inputs using MultiWaiter
        input_data = await self.input_waiter.get()  # Returns dict for "all" mode
        
        # Store for potential later use
        self.previous_values.update(input_data)
        
        # Concatenate all inputs
        start_time = time.time()
        tensors_to_concat = [input_data[name] for name in sorted(self.connected_inputs)]
        
        if tensors_to_concat:
            # Concatenate along feature dimension (dim=1)
            concatenated = torch.cat(tensors_to_concat, dim=self.concat_dim)
            
            self.last_compute_time = time.time() - start_time
            self.compute_count += 1
            
            # Send output
            await self.send_output("output", concatenated)
    
    async def run_as_available(self):
        """Output immediately when any input arrives, padding missing inputs"""
        import torch
        import time
        
        # Wait for any input using MultiWaiter (returns dict with single key)
        input_dict = await self.input_waiter.get()
        
        # Extract the single input
        input_name = list(input_dict.keys())[0]
        new_data = input_dict[input_name]
        
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
                    # Assume same feature size as new_data for simplicity
                    batch_size = new_data.shape[0]
                    feature_size = new_data.shape[1] if new_data.dim() > 1 else 1
                    pad_shape = (batch_size, feature_size) + new_data.shape[2:]
                    zero_tensor = torch.zeros(pad_shape, device=new_data.device, dtype=new_data.dtype)
                    
                    # Preserve gradient if input has it
                    if new_data.requires_grad:
                        zero_tensor.requires_grad_(True)
                        
                    tensors_to_concat.append(zero_tensor)
                else:  # "hold previous"
                    # Skip if we don't have previous data yet
                    continue
        
        if tensors_to_concat:
            # Ensure all on same device
            device = tensors_to_concat[0].device
            tensors_to_concat = [t.to(device) for t in tensors_to_concat]
            
            # Concatenate along feature dimension (dim=1)
            concatenated = torch.cat(tensors_to_concat, dim=self.concat_dim)
            
            self.last_compute_time = time.time() - start_time
            self.compute_count += 1
            
            # Send output
            await self.send_output("output", concatenated)
    
    async def compute(self, **inputs) -> Dict[str, Any]:
        """Required by abstract base class - not used since we override run()"""
        # This method is required by QueueNode abstract base class
        # but not actually called since we override run() method
        raise NotImplementedError("Concat node uses custom run() method, not compute()")