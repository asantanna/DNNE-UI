# Template variables - replaced during export
template_vars = {
    "NODE_ID": "barrier_1",
    "CLASS_NAME": "BarrierNode",
    "HOLD_MODE": "FIFO"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Barrier node - holds data until triggered to release"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        # Both inputs are optional - can receive data or triggers at any time
        self.setup_inputs(required=[], optional=["input", "release"], queue_size=100)
        self.setup_outputs(["output"])
        
        # State management
        self.hold_mode = {HOLD_MODE}
        self.fifo_queue = deque()  # Internal queue for held data
        self.release_count = 0  # Number of pending releases
        self.total_held = 0  # Statistics
        self.total_released = 0  # Statistics
        
    async def run(self):
        """Custom run method: handle data and trigger inputs separately"""
        import time
        self.running = True
        self.node_logger.info(f"Starting Barrier node {self.node_id} with mode: {self.hold_mode}")
        
        try:
            while self.running:
                # Wait for ANY input using MultiWaiter (returns dict with single key)
                input_dict = await self.input_waiter.get()
                
                # Extract the single input (there should be exactly one)
                input_name = list(input_dict.keys())[0]
                input_value = input_dict[input_name]
                
                # Route based on input type
                if input_name == "input":
                    # Data arrived - add to queue and process releases
                    await self.handle_data_input(input_value)
                elif input_name == "release":
                    # Trigger arrived - increment count and process releases
                    await self.handle_trigger_input(input_value)
                else:
                    self.node_logger.warning(f"Unknown input: {input_name}")
                    
        except asyncio.CancelledError:
            self.node_logger.info(f"Barrier Node {self.node_id} cancelled")
            raise
        finally:
            self.running = False
            self.node_logger.info(f"Barrier stats - Total held: {self.total_held}, Released: {self.total_released}, Still queued: {len(self.fifo_queue)}")
    
    async def handle_data_input(self, data):
        """Handle incoming data - add to queue and process pending releases"""
        # Add data to FIFO queue
        self.fifo_queue.append(data)
        self.total_held += 1
        
        shape_info = data.shape if hasattr(data, 'shape') else 'unknown'
        self.node_logger.debug(f"Barrier: Data received (shape: {shape_info}), queue size: {len(self.fifo_queue)}")
        
        # Process any pending releases
        await self.process_releases()
    
    async def handle_trigger_input(self, trigger):
        """Handle trigger - increment release count and process"""
        # Increment release count
        self.release_count += 1
        
        trigger_info = trigger if isinstance(trigger, dict) else {"signal": "trigger"}
        self.node_logger.debug(f"Barrier: Trigger received, release_count: {self.release_count}, queue size: {len(self.fifo_queue)}")
        
        # Process releases
        await self.process_releases()
    
    async def process_releases(self):
        """Process pending releases from the queue"""
        releases_made = 0
        
        while self.release_count > 0 and self.fifo_queue:
            # Remove oldest item from queue (FIFO)
            data = self.fifo_queue.popleft()
            
            # Send to output
            await self.send_output("output", data)
            
            # Update counters
            self.release_count -= 1
            self.total_released += 1
            releases_made += 1
            
            shape_info = data.shape if hasattr(data, 'shape') else 'unknown'
            self.node_logger.info(f"Barrier: Released data (shape: {shape_info}), remaining queue: {len(self.fifo_queue)}, pending releases: {self.release_count}")
        
        if releases_made == 0 and self.release_count > 0:
            self.node_logger.debug(f"Barrier: {self.release_count} releases pending, waiting for data")
        elif releases_made == 0 and self.fifo_queue:
            self.node_logger.debug(f"Barrier: {len(self.fifo_queue)} items queued, waiting for triggers")
    
    async def compute(self, **inputs) -> Dict[str, Any]:
        """Required by abstract base class - not used since we override run()"""
        # This method is required by QueueNode abstract base class
        # but not actually called since we override run() method
        raise NotImplementedError("Barrier node uses custom run() method, not compute()")