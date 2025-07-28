"""
Thread-safe version of sync_adaptive_yield that can work from executor threads.
"""

import asyncio
import time
import threading
import queue
from typing import Optional


class ThreadSafeYielder:
    """Manages yielding from threads back to the main event loop"""
    
    _instance: Optional['ThreadSafeYielder'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.main_loop: Optional[asyncio.AbstractEventLoop] = None
        self.yield_queue = queue.Queue()
        self.response_queues = {}  # thread_id -> response queue
        self.yielder_task = None
        
    @classmethod
    def get_instance(cls) -> 'ThreadSafeYielder':
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    async def start(self, loop: asyncio.AbstractEventLoop):
        """Start the yielder task in the main event loop"""
        self.main_loop = loop
        if self.yielder_task is None:
            self.yielder_task = asyncio.create_task(self._yield_processor())
            # print(f"[THREAD_SAFE_YIELDER] Started yield processor in main loop") #DBG_TAG#
    
    async def _yield_processor(self):
        """Process yield requests from threads"""
        # print(f"[THREAD_SAFE_YIELDER] Yield processor running") #DBG_TAG#
        
        while True:
            try:
                # Check for yield requests (non-blocking)
                try:
                    thread_id, delay = self.yield_queue.get_nowait()
                    # print(f"[THREAD_SAFE_YIELDER] Processing yield request from thread {thread_id}, delay={delay}") #DBG_TAG#
                    
                    # Perform the actual yield
                    await asyncio.sleep(delay)
                    
                    # Signal completion back to thread
                    if thread_id in self.response_queues:
                        self.response_queues[thread_id].put(True)
                        
                except queue.Empty:
                    # No requests, sleep briefly
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                # print(f"[THREAD_SAFE_YIELDER] Error in yield processor: {e}") #DBG_TAG#
                await asyncio.sleep(0.1)
    
    def sync_yield_from_thread(self, delay: float = 0.05):
        """Called from thread to yield control back to main loop"""
        thread_id = threading.get_ident()
        
        # Create response queue for this thread if needed
        if thread_id not in self.response_queues:
            self.response_queues[thread_id] = queue.Queue()
        
        # print(f"[THREAD_SAFE_YIELDER] Thread {thread_id} requesting yield, delay={delay}") #DBG_TAG#
        
        # Send yield request
        self.yield_queue.put((thread_id, delay))
        
        # Wait for completion
        try:
            self.response_queues[thread_id].get(timeout=delay + 1.0)
            # print(f"[THREAD_SAFE_YIELDER] Thread {thread_id} yield completed") #DBG_TAG#
        except queue.Empty:
            # print(f"[THREAD_SAFE_YIELDER] Thread {thread_id} yield timeout!") #DBG_TAG#
            pass


def thread_safe_sync_adaptive_yield(delay: float = 0.5):
    """
    Thread-safe version of sync_adaptive_yield.
    Works from both async context and thread context.
    """
    
    # Try to get running loop
    try:
        loop = asyncio.get_running_loop()
        # We're in async context, use original approach
        # print(f"[SYNC_YIELD] In async context, using direct yield") #DBG_TAG#
        
        # Quick yield using sleep
        done = False
        
        def set_done():
            nonlocal done
            done = True
        
        loop.call_later(delay, set_done)
        
        while not done:
            loop._run_once()
            
    except RuntimeError:
        # No running loop - we're in a thread
        # print(f"[SYNC_YIELD] In thread context, using thread-safe yield") #DBG_TAG#
        
        # Use thread-safe yielder
        yielder = ThreadSafeYielder.get_instance()
        
        if yielder.main_loop is None:
            # print(f"[SYNC_YIELD] Warning: No main loop registered, skipping yield") #DBG_TAG#
            return
            
        yielder.sync_yield_from_thread(delay)