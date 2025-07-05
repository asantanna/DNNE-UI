"""Queue-Based Node Framework"""
import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from asyncio import Queue

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
    
    async def run(self, duration: Optional[float] = None):
        """Run all nodes"""
        self.logger.info("Starting graph execution")
        
        # Start all nodes
        for node in self.nodes.values():
            task = asyncio.create_task(node.run())
            self.tasks.append(task)
        
        try:
            if duration:
                await asyncio.sleep(duration)
                self.logger.info(f"Stopping after {duration}s")
            else:
                # Run until cancelled
                await asyncio.gather(*self.tasks)
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            # Cancel all tasks
            for task in self.tasks:
                task.cancel()
            
            # Wait for cancellation
            await asyncio.gather(*self.tasks, return_exceptions=True)
            self.logger.info("All nodes stopped")
    
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
