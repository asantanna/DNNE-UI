# Template variables - replaced during export
template_vars = {
    "NODE_ID": "eat_n_1",
    "CLASS_NAME": "Eat_NNode",
    "NUM_TO_EAT": "1",
    "TRIGGER_MODE": "every_eat"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Eat_N node - consumes first N inputs then becomes passthrough"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        # Input is optional - can receive data at any time
        self.setup_inputs(required=[], optional=["input"], queue_size=100)
        self.setup_outputs(["output", "trigger"])
        
        # Configuration
        self.num_to_eat = {NUM_TO_EAT}
        self.trigger_mode = {TRIGGER_MODE}
        
        # State management
        self.counter = 0  # Number of inputs consumed
        self.is_passthrough = False  # Whether we're in passthrough mode
        
        # Statistics
        self.total_consumed = 0
        self.total_passed = 0
        self.triggers_sent = 0
        
    async def run(self):
        """Custom run method: handle consume vs passthrough modes"""
        import time
        self.running = True
        self.node_logger.info(f"Starting Eat_N node {self.node_id} - will consume {self.num_to_eat} inputs, trigger mode: {self.trigger_mode}")
        
        try:
            while self.running:
                # Log waiting for input (with deadlock monitoring)
                from framework.globals import Global as g
                if g.deadlock_debug:
                    from framework.deadlock_utils import log_queue_get_wait, log_queue_get_success
                    import time
                    log_queue_get_wait(self.node_id, "input")
                    wait_start = time.time()
                
                # Wait for input
                input_dict = await self.input_waiter.get()
                
                # Log successful receipt (with deadlock monitoring)
                if g.deadlock_debug:
                    # Log success for the actual input received
                    input_names = list(input_dict.keys())
                    for input_name in input_names:
                        log_queue_get_success(self.node_id, input_name, time.time() - wait_start)
                
                # Extract the input (should be "input" key)
                if "input" in input_dict:
                    input_value = input_dict["input"]
                    await self.handle_input(input_value)
                else:
                    # Handle unexpected input
                    input_name = list(input_dict.keys())[0] if input_dict else "unknown"
                    self.node_logger.warning(f"Eat_N: Unexpected input '{input_name}'")
                    
        except asyncio.CancelledError:
            self.node_logger.info(f"Eat_N Node {self.node_id} cancelled")
            raise
        finally:
            self.running = False
            self.node_logger.info(
                f"Eat_N stats - Consumed: {self.total_consumed}, "
                f"Passed: {self.total_passed}, Triggers sent: {self.triggers_sent}"
            )
    
    async def handle_input(self, data):
        """Handle incoming data based on current state"""
        shape_info = data.shape if hasattr(data, 'shape') else 'unknown'
        
        if not self.is_passthrough:
            # Consuming state - eat the input
            self.counter += 1
            self.total_consumed += 1
            
            self.node_logger.debug(
                f"Eat_N: Consumed input {self.counter}/{self.num_to_eat} (shape: {shape_info})"
            )
            
            # Send trigger based on mode
            should_send_trigger = False
            if self.trigger_mode == "every_eat":
                # Send trigger for each consumed input
                should_send_trigger = True
            elif self.trigger_mode == "last_only":
                # Send trigger only when reaching num_to_eat
                should_send_trigger = (self.counter == self.num_to_eat)
            
            if should_send_trigger:
                trigger_data = {
                    "source": self.node_id,
                    "consumed_count": self.counter,
                    "is_last": (self.counter == self.num_to_eat)
                }
                await self.send_output("trigger", trigger_data)
                self.triggers_sent += 1
                self.node_logger.info(
                    f"Eat_N: Sent trigger (consumed: {self.counter}/{self.num_to_eat})"
                )
            
            # Check if we should transition to passthrough
            if self.counter >= self.num_to_eat:
                self.is_passthrough = True
                self.node_logger.info(
                    f"Eat_N: Transitioning to passthrough mode after consuming {self.num_to_eat} inputs"
                )
        else:
            # Passthrough state - forward the input
            await self.send_output("output", data)
            self.total_passed += 1
            
            #self.node_logger.debug(
            #    f"Eat_N: Passthrough input (shape: {shape_info}, total passed: {self.total_passed})"
            #)
    
    async def compute(self, **inputs) -> Dict[str, Any]:
        """Required by abstract base class - not used since we override run()"""
        # This method is required by QueueNode abstract base class
        # but not actually called since we override run() method
        raise NotImplementedError("Eat_N node uses custom run() method, not compute()")