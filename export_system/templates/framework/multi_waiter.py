"""
MultiWaiter - Efficient async utility for waiting on multiple input queues
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple


class MultiWaiter:
    """
    Efficient multi-input queue waiter that uses persistent listener tasks
    for "any" mode or simple sequential waits for "all" mode.
    
    Instead of creating/destroying tasks on each wait, "any" mode maintains
    persistent listener tasks that forward data through an internal queue.
    """
    
    def __init__(self, input_names: List[str], input_queues: Dict[str, asyncio.Queue], 
                 wait_mode: str = "all"):
        """
        Initialize the MultiWaiter.
        
        Args:
            input_names: List of input names to monitor
            input_queues: Dictionary mapping input names to their queues
            wait_mode: "all" (default) or "any"
        """
        self.input_names = input_names
        self.input_queues = input_queues
        self.wait_mode = wait_mode
        
        if wait_mode == "any":
            # Internal queue for communication between listeners and waiter
            self.internal_queue = asyncio.Queue()
            
            # Start persistent listener tasks
            self.listener_tasks = []
            for input_name in input_names:
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
            For "any" mode: Tuple of (payload, source_name)
        """
        if self.wait_mode == "any":
            # Wait on the internal queue for first available
            return await self.internal_queue.get()
        else:  # "all" mode
            # Simple sequential wait on all inputs
            collected = {}
            for input_name in self.input_names:
                if input_name in self.input_queues:
                    data = await self.input_queues[input_name].get()
                    collected[input_name] = data
            return collected
    
    def clear(self):
        """Clear any pending data in the internal queue (only relevant for "any" mode)"""
        if self.wait_mode == "any":
            while not self.internal_queue.empty():
                try:
                    self.internal_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break