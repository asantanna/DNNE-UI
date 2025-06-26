"""
DNNE-UI Execution Engine
Minimal execution system for robotics workflows.
"""

import json
import gc
import threading
import time
import uuid
from typing import Dict, Any, List, Optional
import queue
from enum import Enum

class CacheType(Enum):
    """Cache types for execution"""
    CLASSIC = "classic"
    LOW_VRAM = "low_vram"
    NONE = "none"

class PromptQueue:
    """Minimal prompt queue for DNNE"""
    
    def __init__(self, server):
        self.server = server
        self.queue = queue.Queue()
        self.currently_running = {}
        self.history = {}
        self.mutex = threading.RLock()
        
    def put(self, item):
        """Add item to queue"""
        prompt_id = str(uuid.uuid4())
        with self.mutex:
            self.queue.put((prompt_id, item))
            print(f"Added prompt {prompt_id} to queue")
        return prompt_id
    
    def get(self, timeout=1.0):
        """Get next item from queue"""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def task_done(self, prompt_id, outputs):
        """Mark task as completed"""
        with self.mutex:
            if prompt_id in self.currently_running:
                del self.currently_running[prompt_id]
            self.history[prompt_id] = {
                "outputs": outputs,
                "status": {"completed": True}
            }
    
    def get_current_queue(self):
        """Get current queue state"""
        with self.mutex:
            running = list(self.currently_running.keys())
            pending = []
            # Get pending items without consuming them
            temp_items = []
            while not self.queue.empty():
                try:
                    item = self.queue.get_nowait()
                    temp_items.append(item)
                    pending.append(item[0])  # prompt_id
                except queue.Empty:
                    break
            # Put items back
            for item in temp_items:
                self.queue.put(item)
            
            return {
                "queue_running": running,
                "queue_pending": pending
            }
    
    def get_current_queue_volatile(self):
        """Get current queue state (volatile version for frequent polling)"""
        with self.mutex:
            running = list(self.currently_running.keys())
            pending = []
            # Get pending items without consuming them
            temp_items = []
            while not self.queue.empty():
                try:
                    item = self.queue.get_nowait()
                    temp_items.append(item)
                    pending.append(item[0])  # prompt_id
                except queue.Empty:
                    break
            # Put items back
            for item in temp_items:
                self.queue.put(item)
            
            # Return as tuple: (running, pending) - server.py expects this format
            return (running, pending)
    
    def get_tasks_remaining(self):
        """Get number of tasks remaining"""
        with self.mutex:
            return self.queue.qsize() + len(self.currently_running)
    
    def get_history(self, max_items=100):
        """Get execution history"""
        with self.mutex:
            # Return recent history items
            history_items = list(self.history.items())
            if max_items and len(history_items) > max_items:
                history_items = history_items[-max_items:]
            
            return {item_id: item_data for item_id, item_data in history_items}
    
    def delete_queue_item(self, prompt_id):
        """Delete item from queue"""
        # For minimal implementation, just return success
        return True
    
    def clear_queue(self):
        """Clear the queue"""
        with self.mutex:
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
            self.currently_running.clear()

class PromptExecutor:
    """Minimal prompt executor for DNNE"""
    
    def __init__(self, server, cache_type=None, cache_size=100):
        self.server = server
        self.cache_type = cache_type or CacheType.CLASSIC
        self.cache_size = cache_size
        self.outputs = {}
        print(f"PromptExecutor initialized with cache_type={self.cache_type}, cache_size={self.cache_size}")
        
    def execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        """Execute a workflow prompt"""
        try:
            print(f"Executing prompt {prompt_id}")
            
            # Basic execution - for now just validate and return
            if not isinstance(prompt, dict):
                raise ValueError("Prompt must be a dictionary")
            
            # Simulate some processing
            outputs = {}
            for node_id, node_data in prompt.items():
                if isinstance(node_data, dict) and "class_type" in node_data:
                    print(f"  Processing node {node_id}: {node_data['class_type']}")
                    # For now, just create dummy output
                    outputs[node_id] = {"result": "processed"}
            
            self.outputs[prompt_id] = outputs
            
            return {
                "success": True, 
                "prompt_id": prompt_id,
                "outputs": outputs
            }
            
        except Exception as e:
            print(f"Execution error: {e}")
            return {
                "success": False, 
                "error": str(e),
                "prompt_id": prompt_id
            }
    
    def interrupt(self):
        """Interrupt current execution"""
        print("Execution interrupted")
        
    def get_outputs(self, prompt_id):
        """Get outputs for a prompt"""
        return self.outputs.get(prompt_id, {})

# Additional classes that might be imported
class ExecutionBlocker:
    """Minimal execution blocker"""
    def __init__(self, prompt_id):
        self.prompt_id = prompt_id

def recursive_execute(server, prompt, outputs, current_item, extra_data, executed):
    """Minimal recursive execution function"""
    return True

def recursive_will_execute(prompt, outputs, current_item):
    """Minimal recursive will execute function"""
    return []

def recursive_output_delete_if_changed(prompt, old_prompt, outputs, current_item):
    """Minimal recursive output delete function"""
    return []

# Export commonly used functions/classes
__all__ = [
    "PromptQueue", 
    "PromptExecutor", 
    "ExecutionBlocker",
    "CacheType",
    "recursive_execute",
    "recursive_will_execute", 
    "recursive_output_delete_if_changed"
]
