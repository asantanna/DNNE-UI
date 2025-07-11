# Queue-Based Node Framework
class QueueNode(ABC):
    """Base class for all queue-based nodes"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.input_queues: Dict[str, Queue] = {}
        self.output_subscribers: Dict[str, List[Queue]] = {}
        self.required_inputs: List[str] = []
        self.output_names: List[str] = []
        self.running = False
        self.compute_count = 0
        self.last_compute_time = 0.0
        self.logger = logging.getLogger(f"Node.{node_id}")
    
    def setup_inputs(self, required: List[str], queue_size: int = 100):
        """Setup input queues"""
        self.required_inputs = required
        for input_name in required:
            self.input_queues[input_name] = Queue(maxsize=queue_size)
    
    def setup_outputs(self, outputs: List[str]):
        """Setup output specifications"""
        self.output_names = outputs
        for output_name in outputs:
            self.output_subscribers[output_name] = []
    
    async def send_output(self, output_name: str, value: Any):
        """Send output to all subscribers"""
        if output_name in self.output_subscribers:
            for queue in self.output_subscribers[output_name]:
                await queue.put(value)
    
    @abstractmethod
    async def compute(self, **inputs) -> Dict[str, Any]:
        """Override this to implement node logic"""
        pass
    
    async def run(self):
        """Main execution loop"""
        self.running = True
        self.logger.info(f"Starting node {self.node_id}")
        
        try:
            while self.running:
                # Gather all required inputs
                inputs = {}
                for input_name in self.required_inputs:
                    value = await self.input_queues[input_name].get()
                    inputs[input_name] = value
                
                # Execute compute
                start_time = time.time()
                outputs = await self.compute(**inputs)
                self.last_compute_time = time.time() - start_time
                self.compute_count += 1
                
                # Send outputs
                for output_name, value in outputs.items():
                    await self.send_output(output_name, value)
                    
        except asyncio.CancelledError:
            self.logger.info(f"Node {self.node_id} cancelled")
            raise
        finally:
            self.running = False


class SensorNode(QueueNode):
    """Base class for sensor nodes that generate data at fixed rates"""
    
    def __init__(self, node_id: str, update_rate: float):
        super().__init__(node_id)
        self.update_rate = update_rate
        self.update_interval = 1.0 / update_rate
    
    async def run(self):
        """Sensor run loop with fixed rate"""
        self.running = True
        self.logger.info(f"Starting sensor {self.node_id} at {self.update_rate}Hz")
        
        try:
            while self.running:
                start_time = time.time()
                
                # Execute compute
                outputs = await self.compute()
                self.compute_count += 1
                
                # Send outputs
                for output_name, value in outputs.items():
                    await self.send_output(output_name, value)
                
                # Sleep to maintain rate
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
                self.last_compute_time = time.time() - start_time
                
        except asyncio.CancelledError:
            self.logger.info(f"Sensor {self.node_id} cancelled")
            raise
        finally:
            self.running = False


class GraphRunner:
    """Manages and runs the complete node graph"""
    
    def __init__(self):
        self.nodes: Dict[str, QueueNode] = {}
        self.tasks: List[asyncio.Task] = []
        self.logger = logging.getLogger("GraphRunner")
        
        # Exit tracking for smart checkpoint saves
        self.exit_reason = None
        self.has_completion_conditions = False
    
    def add_node(self, node: QueueNode):
        """Add a node to the graph"""
        self.nodes[node.node_id] = node
        self.logger.info(f"Added node: {node.node_id}")
    
    def wire_nodes(self, connections: List[tuple]):
        """Wire nodes together: (from_id, output, to_id, input)"""
        for from_id, output_name, to_id, input_name in connections:
            from_node = self.nodes[from_id]
            to_node = self.nodes[to_id]
            
            # Subscribe to_node's input queue to from_node's output
            from_node.output_subscribers[output_name].append(
                to_node.input_queues[input_name]
            )
            self.logger.info(f"Connected {from_id}.{output_name} -> {to_id}.{input_name}")
    
    def _detect_completion_conditions(self):
        """Detect if the workflow has any defined completion conditions"""
        # Check for EpochTracker (indicates supervised learning with defined epochs)
        has_epoch_tracker = any("EpochTracker" in node.__class__.__name__ for node in self.nodes.values())
        
        # Check for timeout specified at runtime (duration parameter)
        # This will be checked in the run method
        
        self.has_completion_conditions = has_epoch_tracker
        return self.has_completion_conditions
    
    async def run(self, duration: Optional[float] = None):
        """Run all nodes"""
        self.logger.info("Starting graph execution")
        
        # Detect completion conditions
        self._detect_completion_conditions()
        if duration is not None:
            self.has_completion_conditions = True  # Timeout is a completion condition
        
        # Check if we're in inference mode
        import builtins
        inference_mode = getattr(builtins, 'INFERENCE_MODE', False)
        
        try:
            if inference_mode:
                self.logger.info("Running in INFERENCE mode - gradients disabled")
                # Import torch only if in inference mode
                import torch
                
                # Run in no_grad context for inference
                with torch.no_grad():
                    await self._run_graph(duration)
            else:
                await self._run_graph(duration)
        finally:
            # Handle checkpoint saves before final cleanup
            await self._handle_exit_checkpoints()
    
    async def _run_graph(self, duration: Optional[float] = None):
        """Internal method to run the graph"""
        # Start all nodes
        for node in self.nodes.values():
            task = asyncio.create_task(node.run())
            self.tasks.append(task)
        
        try:
            if duration:
                await asyncio.sleep(duration)
                self.exit_reason = "timeout"
                self.logger.info(f"Stopping after {duration}s")
            else:
                # Run until cancelled or training completes
                completion_result = await self._run_until_completion()
                if completion_result == "training_complete":
                    self.exit_reason = "training_complete"
                else:
                    self.exit_reason = "indefinite_run"
        except KeyboardInterrupt:
            self.exit_reason = "keyboard_interrupt"
            self.logger.info("Interrupted by user")
        finally:
            # Cancel all tasks
            for task in self.tasks:
                task.cancel()
            
            # Wait for cancellation
            await asyncio.gather(*self.tasks, return_exceptions=True)
            self.logger.info("All nodes stopped")
    
    async def _run_until_completion(self):
        """Run until training completion or cancellation"""
        epoch_tracker = None
        for node in self.nodes.values():
            if "EpochTracker" in node.__class__.__name__:
                epoch_tracker = node
                break
        
        if not epoch_tracker:
            # No epoch tracker, run indefinitely
            await asyncio.gather(*self.tasks)
            return "indefinite_run"
        
        # Create a completion monitoring task
        completion_task = asyncio.create_task(self._monitor_completion(epoch_tracker))
        all_tasks = self.tasks + [completion_task]
        
        try:
            # Wait for either completion or all tasks to finish
            done, pending = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)
            
            if completion_task in done:
                self.logger.info("Training completed - stopping execution")
                return "training_complete"
            else:
                self.logger.info("Training tasks completed")
                return "tasks_complete"
                
        except Exception as e:
            self.logger.error(f"Error during execution: {e}")
            return "error"
        finally:
            # Cancel any remaining tasks
            for task in all_tasks:
                if not task.done():
                    task.cancel()
    
    async def _monitor_completion(self, epoch_tracker):
        """Monitor epoch tracker for completion signal"""
        while True:
            await asyncio.sleep(1.0)  # Check every second
            
            # Check if epoch tracker has signaled completion
            if hasattr(epoch_tracker, 'current_epoch') and hasattr(epoch_tracker, 'total_epochs'):
                if epoch_tracker.current_epoch >= epoch_tracker.total_epochs:
                    self.logger.info(f"Training completion detected: {epoch_tracker.current_epoch}/{epoch_tracker.total_epochs}")
                    return
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get execution statistics"""
        return {
            node_id: {
                "compute_count": node.compute_count,
                "last_compute_time": node.last_compute_time,
                "running": node.running
            }
            for node_id, node in self.nodes.items()
        }
    
    async def _handle_exit_checkpoints(self):
        """Handle checkpoint saves on exit based on exit reason and completion conditions"""
        
        # Skip if in inference mode
        import builtins
        inference_mode = getattr(builtins, 'INFERENCE_MODE', False)
        if inference_mode:
            self.logger.info("💾 Skipping exit checkpoints - inference mode")
            return
        
        # Determine if we should save checkpoint on exit
        should_save = False
        reason_message = ""
        
        if self.exit_reason == "timeout":
            should_save = True
            reason_message = "Training timeout reached - saving final checkpoint"
            
        elif self.exit_reason == "training_complete":
            should_save = True  
            reason_message = "Training completed naturally - saving final checkpoint"
            
        elif self.exit_reason == "keyboard_interrupt":
            if self.has_completion_conditions:
                should_save = False
                reason_message = "Keyboard interrupt with defined completion conditions - not saving checkpoint"
            else:
                should_save = True
                reason_message = "Keyboard interrupt on indefinite run - saving checkpoint"
                
        elif self.exit_reason == "indefinite_run":
            should_save = True
            reason_message = "Indefinite run stopped - saving checkpoint"
            
        # Log the decision
        if should_save:
            self.logger.info(f"💾 {reason_message}")
        else:
            self.logger.info(f"🚫 {reason_message}")
        
        # Trigger checkpoint saves on eligible nodes
        if should_save:
            saved_count = 0
            for node in self.nodes.values():
                if hasattr(node, 'save_checkpoint_on_exit'):
                    try:
                        success = await node.save_checkpoint_on_exit(self.exit_reason)
                        if success:
                            saved_count += 1
                    except Exception as e:
                        self.logger.error(f"Failed to save exit checkpoint for {node.node_id}: {e}")
                        
            if saved_count > 0:
                self.logger.info(f"💾 Saved exit checkpoints for {saved_count} nodes")
            else:
                self.logger.info("💾 No nodes saved exit checkpoints")
