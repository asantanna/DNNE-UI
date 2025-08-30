# Base Node Classes for Queue-Based Framework

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from asyncio import Queue

from .exceptions import CauseExitException
from .globals import dnne_logging, Global as g
from .multi_waiter import MultiWaiter


class QueueNode(ABC):
    """Base class for all queue-based nodes"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.input_queues: Dict[str, Queue] = {}
        self.output_subscribers: Dict[str, List[Queue]] = {}
        self.required_inputs: List[str] = []
        self.optional_inputs: List[str] = []
        self.output_names: List[str] = []
        self.running = False
        self.compute_count = 0
        self.last_compute_time = 0.0
        self.node_logger = dnne_logging.getLogger(f"node.{node_id}")
        self.input_waiter = None  # Will be set up in setup_inputs
        
        # Register this node with the system
        g.register_node(node_id)
    
    def setup_inputs(self, required: List[str] = None, optional: List[str] = None, 
                     queue_size: int = 10, wait_for_optionals: bool = True):
        """Setup input queues
        
        Args:
            required: List of required input names (must wait for all)
            optional: List of optional input names (can proceed with any)
            queue_size: Maximum size for each queue
            wait_for_optionals: If False, don't block waiting for optional inputs (default True)
        """
        if required is None:
            required = []
        if optional is None:
            optional = []
        
        self.required_inputs = required
        self.optional_inputs = optional
        
        # Create queues for all inputs
        all_inputs = required + optional
        for input_name in all_inputs:
            self.input_queues[input_name] = Queue(maxsize=queue_size)
        
        # Create MultiWaiter if we have inputs
        if all_inputs:
            self.input_waiter = MultiWaiter(
                required, optional,
                self.input_queues,
                self.node_id,
                wait_for_optionals
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
        g.update_node_activity(self.node_id)  # Track output activity
        if output_name in self.output_subscribers:
            # Log output event if deadlock debugging enabled
            if g.deadlock_debug:
                from .deadlock_utils import log_queue_put
                log_queue_put(self.node_id, output_name, len(self.output_subscribers[output_name]))
            
            for queue in self.output_subscribers[output_name]:
                await queue.put(value)
    
    @abstractmethod
    async def compute(self, **inputs) -> Dict[str, Any]:
        """Override this to implement node logic"""
        pass
    
    async def get_config_inputs(self, config_names: List[str]) -> Dict[str, Any]:
        """Get one-time configuration inputs directly from queues
        
        Args:
            config_names: List of configuration input names to retrieve
            
        Returns:
            Dict mapping config names to their values
        """
        configs = {}
        for name in config_names:
            if name not in self.input_queues:
                raise ValueError(f"No queue for config input '{name}'")
            value = await self.input_queues[name].get()
            configs[name] = value
            # Track activity for config receipt
            g.update_node_activity(self.node_id)
        return configs
    
    async def run(self):
        """Main execution loop"""
        self.running = True
        self.node_logger.debug(f"Starting node {self.node_id}")
        
        try:
            while self.running:
                # Get inputs using MultiWaiter
                if self.input_waiter:
                    inputs = await self.input_waiter.get()
                    # inputs is always a dict, regardless of mode ("all" or "any")
                    # If using "any" mode AND doing something exotic like reading queues manually
                    # (e.g. one-time messages), you must override run() to prevent "double-getter deadlock"
                    # See dnne_docs/architecture/queue_framework.md for details
                else:
                    # Node has no inputs (e.g., sensor nodes)
                    inputs = None
                
                # Execute compute
                if g.deadlock_debug:
                    from .deadlock_utils import log_node_compute_start, log_node_compute_end
                    log_node_compute_start(self.node_id)
                
                compute_start = time.time()
                outputs = await self.compute(**inputs)
                self.last_compute_time = time.time() - compute_start
                self.compute_count += 1
                
                if g.deadlock_debug:
                    log_node_compute_end(self.node_id, self.last_compute_time)
                
                # Send outputs
                for output_name, value in outputs.items():
                    await self.send_output(output_name, value)
                    
        except CauseExitException as e:
            # print(f"[DEBUG] QueueNode.run() caught CauseExitException from node {self.node_id}") #DBG_TAG#
            # print(f"[DEBUG] Exception message: {e}") #DBG_TAG#
            self.node_logger.info(f"Node {self.node_id} requested exit: {e.message}")
            raise  # Re-raise to propagate to GraphRunner
        except asyncio.CancelledError:
            self.node_logger.debug(f"Node {self.node_id} cancelled")
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
    
    async def _call_run_when_ready(self):
        """Internal method that ensures system is ready before calling run().
        
        DO NOT OVERRIDE THIS METHOD! Override run() instead.
        
        This method handles the system initialization barrier to ensure
        all nodes are created and all connections are established before
        any node starts processing data.
        """
        try:
            # Report that this node is ready (initialized)
            g.report_node_ready(self.node_id)
            
            # Wait for entire system to be ready
            await g.wait_for_system_ready()
            
            # Double-check that connections were established
            if not g._connections_established:
                raise RuntimeError(f"Node {self.node_id} starting but connections not established! Internal error!")
            
            self.node_logger.debug(f"System ready, starting node {self.node_id}")
            
            # Log node start if deadlock debugging - do this here so ALL nodes log it
            # even if they override run()
            if g.deadlock_debug:
                from .deadlock_utils import log_node_start
                log_node_start(self.node_id, self.__class__.__name__)
            
            # Now call the actual run method
            await self.run()
            
        except CauseExitException:
            # Let CauseExitException propagate normally
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.node_logger.error(f"Fatal error in node {self.node_id}: {e}")
            import traceback
            self.node_logger.error(traceback.format_exc())
            # Re-raise to ensure the error propagates
            raise


class SensorNode(QueueNode):
    """Base class for sensor nodes that generate data at fixed rates"""
    
    def __init__(self, node_id: str, update_rate: float):
        super().__init__(node_id)  # No wait_mode needed
        self.update_rate = update_rate
        self.update_interval = 1.0 / update_rate
    
    async def run(self):
        """Sensor run loop with fixed rate"""
        self.running = True
        self.node_logger.debug(f"Starting sensor {self.node_id} at {self.update_rate}Hz")
        
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
            self.node_logger.debug(f"Sensor {self.node_id} cancelled")
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