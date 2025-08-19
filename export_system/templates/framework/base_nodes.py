# Base Node Classes for Queue-Based Framework

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from asyncio import Queue

from .exceptions import CauseExitException
from .globals import dnne_logging
from .multi_waiter import MultiWaiter


class QueueNode(ABC):
    """Base class for all queue-based nodes"""
    
    def __init__(self, node_id: str, wait_mode: str = "all"):
        self.node_id = node_id
        self.wait_mode = wait_mode
        self.input_queues: Dict[str, Queue] = {}
        self.output_subscribers: Dict[str, List[Queue]] = {}
        self.required_inputs: List[str] = []
        self.output_names: List[str] = []
        self.running = False
        self.compute_count = 0
        self.last_compute_time = 0.0
        self.node_logger = dnne_logging.getLogger(f"node.{node_id}")
        self.input_waiter = None  # Will be set up in setup_inputs
    
    def setup_inputs(self, required: List[str], queue_size: int = 100):
        """Setup input queues"""
        self.required_inputs = required
        for input_name in required:
            self.input_queues[input_name] = Queue(maxsize=queue_size)
        
        # Create MultiWaiter for this node if we have inputs
        if required:
            self.input_waiter = MultiWaiter(
                required, 
                self.input_queues,
                wait_mode=self.wait_mode
            )
    
    def setup_outputs(self, outputs: List[str]):
        """Setup output specifications"""
        self.output_names = outputs
        for output_name in outputs:
            self.output_subscribers[output_name] = []
    
    def set_connections(self, connections: Dict[str, List]):
        """Called by GraphRunner to inform node about its connections.
        Override this method if your node needs to track connections."""
        # Base implementation just stores the connections
        # Subclasses can override to do more sophisticated tracking
        self.connections = connections
    
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
        self.node_logger.info(f"Starting node {self.node_id}")
        
        try:
            while self.running:
                # Get inputs using MultiWaiter
                if self.input_waiter:
                    inputs = await self.input_waiter.get()
                    # For "all" mode: inputs is a dict
                    # For "any" mode: nodes that use "any" should override run()
                else:
                    # No inputs required (e.g., sensor nodes)
                    inputs = {}
                
                # Execute compute
                compute_start = time.time()
                outputs = await self.compute(**inputs)
                self.last_compute_time = time.time() - compute_start
                self.compute_count += 1
                
                # Send outputs
                for output_name, value in outputs.items():
                    await self.send_output(output_name, value)
                    
        except CauseExitException as e:
            # print(f"[DEBUG] QueueNode.run() caught CauseExitException from node {self.node_id}") #DBG_TAG#
            # print(f"[DEBUG] Exception message: {e}") #DBG_TAG#
            self.node_logger.info(f"Node {self.node_id} requested exit: {e.message}")
            raise  # Re-raise to propagate to GraphRunner
        except asyncio.CancelledError:
            self.node_logger.info(f"Node {self.node_id} cancelled")
            raise
        except Exception as e:
            # Catch any other exceptions and exit immediately
            self.node_logger.error(f"FATAL ERROR in node {self.node_id}: {e}")
            import traceback
            self.node_logger.error(traceback.format_exc())
            print(f"\n❌ FATAL ERROR in node {self.node_id}: {e}")
            print("Exiting immediately due to node error.")
            import sys
            sys.exit(1)
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
        self.node_logger.info(f"Starting sensor {self.node_id} at {self.update_rate}Hz")
        
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
            self.node_logger.info(f"Sensor {self.node_id} cancelled")
            raise
        except Exception as e:
            # Catch any other exceptions and exit immediately
            self.node_logger.error(f"FATAL ERROR in sensor {self.node_id}: {e}")
            import traceback
            self.node_logger.error(traceback.format_exc())
            print(f"\n❌ FATAL ERROR in sensor {self.node_id}: {e}")
            print("Exiting immediately due to sensor error.")
            import sys
            sys.exit(1)
        finally:
            self.running = False