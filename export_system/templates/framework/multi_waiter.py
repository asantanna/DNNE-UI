"""
MultiWaiter - Efficient async utility for waiting on multiple input queues
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from .globals import Global as g


class MultiWaiter:
    """
    Efficient multi-input queue waiter that uses persistent listener tasks
    for "any" mode or simple sequential waits for "all" mode.
    
    Instead of creating/destroying tasks on each wait, "any" mode maintains
    persistent listener tasks that forward data through an internal queue.
    """
    
    def __init__(self, required: List[str], optional: List[str], 
                 input_queues: Dict[str, asyncio.Queue], node_id: str,
                 wait_for_optionals: bool = True):
        """
        Initialize the MultiWaiter.
        
        Args:
            required: List of required input names (must wait for all)
            optional: List of optional input names (can proceed with any)
            input_queues: Dictionary mapping input names to their queues
            node_id: ID of the node this waiter belongs to (for activity tracking)
            wait_for_optionals: If False, don't block waiting for optional inputs (default True)
        """
        self.required = required
        self.optional = optional
        self.input_names = required + optional
        self.input_queues = input_queues
        self.node_id = node_id
        self.wait_for_optionals = wait_for_optionals
        
        # Auto-infer wait mode
        if len(self.required) == len(self.input_names):
            self.wait_mode = "all"  # All inputs are required
        else:
            self.wait_mode = "any"  # Has optional inputs
        
        if self.wait_mode == "any":
            # For "any" mode with required inputs
            self.required_and_received = {}
            self.have_all_required = asyncio.Event()
            self.have_all_required.clear()  # Initially blocked
            
            # Internal queue for communication between listeners and waiter
            self.internal_queue = asyncio.Queue()
            
            # Start persistent listener tasks
            self.listener_tasks = []
            for input_name in self.input_names:
                if input_name in input_queues:
                    task = asyncio.create_task(
                        self._listener_loop(input_name, input_queues[input_name]),
                        name=f"listener_{input_name}"
                    )
                    self.listener_tasks.append(task)
    
    async def _listener_loop(self, input_name: str, input_queue: asyncio.Queue):
        """
        Persistent listener loop for a single input queue.
        
        This task runs continuously, forwarding data from the input queue
        to the internal queue along with source information.
        """
        try:
            while True:
                # Wait for data from this input
                data = await input_queue.get()
                
                # Block if this is a duplicate required input
                if input_name in self.required:
                    if input_name in self.required_and_received:
                        # Block if this is a duplicate required input
                        await self.have_all_required.wait()
                    # no longer a duplicate but the next one will be
                    self.required_and_received[input_name] = data
                
                # Forward to internal queue as tuple (payload, source)
                await self.internal_queue.put((data, input_name))
                
        except asyncio.CancelledError:
            # Clean shutdown when node is cancelled
            pass
    
    async def get(self) -> Any:
        """
        Get data from inputs based on wait mode.
        
        Returns:
            For "all" mode: Dict mapping input names to their data
            For "any" mode: Dict with either all required inputs or single optional input
        """
        if self.wait_mode == "all":
            # Simple sequential wait on all inputs
            collected = {}
            
            # Log queue states before starting to wait (for deadlock analysis)
            if g.deadlock_debug:
                from .deadlock_utils import log_queue_state, log_queue_get_wait, log_queue_get_success
                queue_states = {}
                for name, queue in self.input_queues.items():
                    queue_states[name] = queue.qsize()
                log_queue_state(self.node_id, queue_states)
            
            for input_name in self.input_names:
                # Log wait start if deadlock debugging
                if g.deadlock_debug:
                    log_queue_get_wait(self.node_id, input_name)
                    import time
                    wait_start = time.time()
                
                data = await self.input_queues[input_name].get()
                
                # Log successful get
                if g.deadlock_debug:
                    wait_time = time.time() - wait_start
                    log_queue_get_success(self.node_id, input_name, wait_time)
                
                collected[input_name] = data
                
            # Track activity after collecting all inputs
            g.update_node_activity(self.node_id)
            return collected
        
        else:  # "any" mode (handles both pure-any and mixed)
            
            while True:
                # Check if we should skip waiting for optionals
                if not self.wait_for_optionals and self.internal_queue.empty():
                    return {}
                
                data, input_name = await self.internal_queue.get()
                g.update_node_activity(self.node_id)
                
                if input_name in self.required:
                    
                    if len(self.required_and_received) == len(self.required):
                        # Have all required inputs
                        result = self.required_and_received.copy()
                        self.required_and_received.clear()
                        
                        # Unblock listeners
                        self.have_all_required.set()
                        self.have_all_required.clear()
                        
                        return result
                    # Continue collecting required
                    
                else:
                    # Optional input - return immediately
                    return {input_name: data}
    
    def reset(self):
        """Reset/clear partially collected required inputs and internal queue"""
        if self.wait_mode == "any":
            # Clear partially collected required inputs
            if hasattr(self, 'required_data'):
                self.required_and_received.clear()
                # Unblock listeners
                self.have_all_required.set()
                self.have_all_required.clear()
            
            # Clear internal queue
            while not self.internal_queue.empty():
                try:
                    self.internal_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
    
    def clear(self):
        """Backward compatibility - redirects to reset()"""
        self.reset()