# Base Node Classes for Queue-Based Framework

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from asyncio import Queue

from .exceptions import TrainingCompleteException


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
                compute_start = time.time()
                outputs = await self.compute(**inputs)
                self.last_compute_time = time.time() - compute_start
                self.compute_count += 1
                
                # Send outputs
                for output_name, value in outputs.items():
                    await self.send_output(output_name, value)
                    
        except TrainingCompleteException as e:
            print(f"[DEBUG] QueueNode.run() caught TrainingCompleteException from node {self.node_id}")
            print(f"[DEBUG] Exception message: {e}")
            self.logger.info(f"Node {self.node_id} signaled training complete")
            raise  # Re-raise to propagate to GraphRunner
        except asyncio.CancelledError:
            self.logger.info(f"Node {self.node_id} cancelled")
            raise
        except Exception as e:
            # Catch any other exceptions and exit immediately
            self.logger.error(f"FATAL ERROR in node {self.node_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
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
        except Exception as e:
            # Catch any other exceptions and exit immediately
            self.logger.error(f"FATAL ERROR in sensor {self.node_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            print(f"\n❌ FATAL ERROR in sensor {self.node_id}: {e}")
            print("Exiting immediately due to sensor error.")
            import sys
            sys.exit(1)
        finally:
            self.running = False