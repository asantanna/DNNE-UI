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
                 input_queues: Dict[str, asyncio.Queue], node_id: str):
        """
        Initialize the MultiWaiter.
        
        Args:
            required: List of required input names (must wait for all)
            optional: List of optional input names (can proceed with any)
            input_queues: Dictionary mapping input names to their queues
            node_id: ID of the node this waiter belongs to (for activity tracking)
        """
        self.required = required
        self.optional = optional
        self.input_names = required + optional
        self.input_queues = input_queues
        self.node_id = node_id
        
        # Auto-infer wait mode
        if len(self.required) == len(self.input_names):
            self.wait_mode = "all"  # All inputs are required
        else:
            self.wait_mode = "any"  # Has optional inputs
        
        if self.wait_mode == "any":
            # For "any" mode with required inputs
            self.required_data = {}
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
                if input_name in self.required and input_name in self.required_data:
                    await self.have_all_required.wait()
                
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
            for input_name in self.input_names:
                if input_name in self.input_queues:
                    data = await self.input_queues[input_name].get()
                    collected[input_name] = data
            # Track activity after collecting all inputs
            g.update_node_activity(self.node_id)
            return collected
        else:  # "any" mode (handles both pure-any and mixed)
            while True:
                data, input_name = await self.internal_queue.get()
                g.update_node_activity(self.node_id)
                
                if input_name in self.required:
                    # Fail-fast check
                    if input_name in self.required_data:
                        raise RuntimeError(f"MultiWaiter listener did not block {input_name}! Internal logic error!")
                    
                    self.required_data[input_name] = data
                    
                    if len(self.required_data) == len(self.required):
                        # Have all required inputs
                        result = self.required_data.copy()
                        self.required_data.clear()
                        
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
                self.required_data.clear()
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