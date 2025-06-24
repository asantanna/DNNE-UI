#!/usr/bin/env python3
"""
DNNE-UI Import Cleanup Script
Fixes hardcoded imports to disabled SD components.
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple

class ImportCleanup:
    def __init__(self, comfy_root: str):
        self.root = Path(comfy_root)
        self.backup_suffix = ".backup"
        
    def backup_file(self, file_path: Path) -> Path:
        """Create backup of file before modifying"""
        backup_path = file_path.with_suffix(file_path.suffix + self.backup_suffix)
        if not backup_path.exists():
            shutil.copy2(file_path, backup_path)
            print(f"  📋 Backed up: {file_path.name}")
        return backup_path
    
    def restore_file(self, file_path: Path) -> bool:
        """Restore file from backup"""
        backup_path = file_path.with_suffix(file_path.suffix + self.backup_suffix)
        if backup_path.exists():
            shutil.copy2(backup_path, file_path)
            print(f"  ✅ Restored: {file_path.name}")
            return True
        return False
    
    def comment_out_imports(self, file_path: Path, import_patterns: List[str]) -> bool:
        """Comment out specific import statements"""
        if not file_path.exists():
            return False
            
        self.backup_file(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        modified = False
        
        for pattern in import_patterns:
            # Match import statements
            regex_pattern = rf'^(\s*)(import\s+{re.escape(pattern)}.*?)$'
            matches = re.finditer(regex_pattern, content, re.MULTILINE)
            
            for match in reversed(list(matches)):  # Reverse to maintain line positions
                indent, import_stmt = match.groups()
                commented_line = f"{indent}# DNNE: Disabled SD import - {import_stmt}"
                content = content[:match.start()] + commented_line + content[match.end():]
                modified = True
                print(f"    Commented: {import_stmt.strip()}")
            
            # Also match "from X import Y" statements
            regex_pattern = rf'^(\s*)(from\s+{re.escape(pattern)}.*?)$'
            matches = re.finditer(regex_pattern, content, re.MULTILINE)
            
            for match in reversed(list(matches)):
                indent, import_stmt = match.groups()
                commented_line = f"{indent}# DNNE: Disabled SD import - {import_stmt}"
                content = content[:match.start()] + commented_line + content[match.end():]
                modified = True
                print(f"    Commented: {import_stmt.strip()}")
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ Updated: {file_path.name}")
        
        return modified
    
    def fix_nodes_py(self):
        """Fix nodes.py imports"""
        print("🔧 Fixing nodes.py...")
        
        nodes_file = self.root / "nodes.py"
        
        # SD-specific imports to disable
        sd_imports = [
            "comfy.diffusers_load",
            "comfy.sd",
            "comfy.ldm",
            "comfy.samplers",
            "comfy.clip",
            "comfy.controlnet",
            "comfy.t2i_adapter"
        ]
        
        self.comment_out_imports(nodes_file, sd_imports)
        
        # Also need to comment out node class registrations
        if nodes_file.exists():
            with open(nodes_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find and comment out SD node class mappings
            lines = content.split('\n')
            modified_lines = []
            in_node_mapping = False
            
            for line in lines:
                # Check if we're in a NODE_CLASS_MAPPINGS section
                if 'NODE_CLASS_MAPPINGS' in line:
                    in_node_mapping = True
                    modified_lines.append(line)
                elif in_node_mapping and line.strip() == '}':
                    in_node_mapping = False
                    modified_lines.append(line)
                elif in_node_mapping and any(sd_class in line for sd_class in [
                    'CheckpointLoader', 'VAELoader', 'CLIPLoader', 'DiffusersLoader',
                    'KSampler', 'VAEEncode', 'VAEDecode', 'CLIPTextEncode'
                ]):
                    modified_lines.append(f"    # DNNE: Disabled SD node - {line}")
                    print(f"    Commented SD node: {line.strip()}")
                else:
                    modified_lines.append(line)
            
            # Write back if modified
            new_content = '\n'.join(modified_lines)
            if new_content != content:
                with open(nodes_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print("  ✅ Commented out SD node registrations")
    
    def fix_execution_py(self):
        """Fix execution.py imports"""
        print("🔧 Fixing execution.py...")
        
        execution_file = self.root / "execution.py"
        
        sd_imports = [
            "comfy.model_management",
            "comfy.samplers",
            "comfy.sd"
        ]
        
        self.comment_out_imports(execution_file, sd_imports)
    
    def fix_server_py(self):
        """Fix server.py imports"""
        print("🔧 Fixing server.py...")
        
        server_file = self.root / "server.py"
        
        sd_imports = [
            "comfy.samplers",
            "comfy.sd",
            "comfy.extras",
            "folder_paths"  # May need this one, but let's see
        ]
        
        # Only comment out the truly SD-specific ones
        self.comment_out_imports(server_file, [
            "comfy.samplers",
            "comfy.sd"
        ])
    
    def create_minimal_nodes_py(self):
        """Create a minimal nodes.py with just base classes"""
        print("🔧 Creating minimal nodes.py...")
        
        nodes_file = self.root / "nodes.py"
        self.backup_file(nodes_file)
        
        minimal_nodes_content = '''"""
DNNE-UI Nodes
Minimal node system for robotics applications.
"""

import os
import json
import sys
import importlib
import importlib.util
import traceback
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

# Base node class
class DNNENode:
    """Base class for all DNNE nodes"""
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define input types - override in subclasses"""
        return {"required": {}}
    
    RETURN_TYPES: Tuple = ()
    RETURN_NAMES: Optional[Tuple] = None
    FUNCTION: str = "compute"
    CATEGORY: str = "dnne"
    
    def compute(self, **kwargs):
        """Main computation function - override in subclasses"""
        raise NotImplementedError("Subclasses must implement compute()")

# Basic utility nodes
class FloatConstant:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0})
            }
        }
    
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "get_value"
    CATEGORY = "dnne/constants"
    
    def get_value(self, value):
        return (value,)

class IntConstant:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -10000, "max": 10000})
            }
        }
    
    RETURN_TYPES = ("INT",)
    FUNCTION = "get_value"
    CATEGORY = "dnne/constants"
    
    def get_value(self, value):
        return (value,)

class StringConstant:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"default": ""})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_value"  
    CATEGORY = "dnne/constants"
    
    def get_value(self, value):
        return (value,)

# Debug/Display nodes
class PrintValue:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*", {}),
                "prefix": ("STRING", {"default": "Value:"})
            }
        }
    
    RETURN_TYPES = ("*",)
    FUNCTION = "print_value"
    CATEGORY = "dnne/debug"
    OUTPUT_NODE = True
    
    def print_value(self, value, prefix):
        print(f"{prefix} {value}")
        return (value,)

# Node mappings for the system
NODE_CLASS_MAPPINGS = {
    "FloatConstant": FloatConstant,
    "IntConstant": IntConstant, 
    "StringConstant": StringConstant,
    "PrintValue": PrintValue,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FloatConstant": "Float Constant",
    "IntConstant": "Integer Constant",
    "StringConstant": "String Constant", 
    "PrintValue": "Print Value",
}

# Global variables to track loaded nodes
custom_nodes_loaded = False
api_nodes_loaded = False
LOADED_MODULE_DIRS = {}  # Track loaded custom node directories
EXTENSION_WEB_DIRS = {}  # Track extension web directories

def init_extra_nodes(init_custom_nodes=True, init_api_nodes=True):
    """
    Initialize extra nodes (custom nodes and API nodes)
    This function is called by main.py during startup
    """
    global custom_nodes_loaded, api_nodes_loaded
    
    print("Initializing DNNE extra nodes...")
    
    if init_custom_nodes and not custom_nodes_loaded:
        load_custom_nodes()
        custom_nodes_loaded = True
    
    if init_api_nodes and not api_nodes_loaded:
        load_api_nodes()
        api_nodes_loaded = True
    
    print(f"DNNE nodes loaded: {len(NODE_CLASS_MAPPINGS)} total")

def load_custom_nodes():
    """Load custom nodes from custom_nodes directory"""
    print("Loading custom nodes...")
    
    custom_nodes_dir = os.path.join(os.path.dirname(__file__), "custom_nodes")
    if not os.path.exists(custom_nodes_dir):
        print(f"Custom nodes directory not found: {custom_nodes_dir}")
        return
    
    # Look for Python files and packages in custom_nodes
    for item in os.listdir(custom_nodes_dir):
        if item.startswith('.') or item.startswith('_'):
            continue
            
        item_path = os.path.join(custom_nodes_dir, item)
        
        try:
            if os.path.isfile(item_path) and item.endswith('.py'):
                # Load Python file
                module_name = item[:-3]  # Remove .py
                load_custom_node_file(item_path, module_name)
                
            elif os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, '__init__.py')):
                # Load Python package
                load_custom_node_package(item_path, item)
                
        except Exception as e:
            print(f"Failed to load custom node {item}: {e}")
            traceback.print_exc()

def load_custom_node_file(file_path, module_name):
    """Load a single custom node Python file"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Track the loaded module
        module_dir = os.path.dirname(file_path)
        LOADED_MODULE_DIRS[module_name] = module_dir
        
        # Check for web directory (less common for single files)
        web_dir = os.path.join(module_dir, "web")
        if os.path.exists(web_dir):
            EXTENSION_WEB_DIRS[module_name] = web_dir
            print(f"  Found web directory: {module_name}")
        
        # Register node classes
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            for name, cls in module.NODE_CLASS_MAPPINGS.items():
                NODE_CLASS_MAPPINGS[name] = cls
                print(f"  Loaded node: {name}")
        
        # Register display names
        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

def load_custom_node_package(package_path, package_name):
    """Load a custom node Python package"""
    sys.path.insert(0, os.path.dirname(package_path))
    try:
        module = importlib.import_module(package_name)
        
        # Track the loaded module directory
        LOADED_MODULE_DIRS[package_name] = package_path
        
        # Check for web directory
        web_dir = os.path.join(package_path, "web")
        if os.path.exists(web_dir):
            EXTENSION_WEB_DIRS[package_name] = web_dir
            print(f"  Found web directory: {package_name}")
        
        # Register node classes
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            for name, cls in module.NODE_CLASS_MAPPINGS.items():
                NODE_CLASS_MAPPINGS[name] = cls
                print(f"  Loaded node: {name}")
        
        # Register display names  
        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            
    finally:
        if sys.path[0] == os.path.dirname(package_path):
            sys.path.pop(0)

def load_api_nodes():
    """Load API nodes (placeholder for now)"""
    print("Loading API nodes...")
    # For now, just a placeholder
    # TODO: Implement API node loading if needed

def get_object_info():
    """
    Get information about all available nodes
    This is used by the frontend to populate the node palette
    """
    object_info = {}
    
    for name, cls in NODE_CLASS_MAPPINGS.items():
        try:
            info = {
                "input": cls.INPUT_TYPES(),
                "output": getattr(cls, 'RETURN_TYPES', ()),
                "output_name": getattr(cls, 'RETURN_NAMES', ()),
                "name": NODE_DISPLAY_NAME_MAPPINGS.get(name, name),
                "display_name": NODE_DISPLAY_NAME_MAPPINGS.get(name, name),
                "description": getattr(cls, '__doc__', ''),
                "category": getattr(cls, 'CATEGORY', 'dnne'),
                "output_node": getattr(cls, 'OUTPUT_NODE', False),
            }
            object_info[name] = info
        except Exception as e:
            print(f"Error getting info for node {name}: {e}")
    
    return object_info

def load_checkpoint_guess_config(*args, **kwargs):
    """Stub for checkpoint loading - not used in robotics"""
    raise NotImplementedError("Checkpoint loading not supported in DNNE")

def unload_all_models():
    """Stub for model unloading"""
    print("Models unloaded (DNNE stub)")

def soft_empty_cache():
    """Stub for cache clearing"""
    print("Cache cleared (DNNE stub)")

def interrupt_processing(value=True):
    """Stub for interrupt processing"""
    print(f"Processing interrupt: {value}")

# Export for ComfyUI
__all__ = [
    "NODE_CLASS_MAPPINGS", 
    "NODE_DISPLAY_NAME_MAPPINGS",
    "LOADED_MODULE_DIRS",
    "EXTENSION_WEB_DIRS",
    "init_extra_nodes",
    "get_object_info",
    "load_checkpoint_guess_config",
    "unload_all_models", 
    "soft_empty_cache",
    "interrupt_processing"
]
'''
        
        with open(nodes_file, 'w', encoding='utf-8') as f:
            f.write(minimal_nodes_content)
        
        print("  ✅ Created minimal nodes.py")
    
    def fix_all_imports(self):
        """Fix all import issues"""
        print("🛠️  Fixing import issues...")
        
        # Try the targeted approach first
        self.fix_nodes_py()
        self.fix_execution_py() 
        self.fix_server_py()
        
        print("\n💡 If issues persist, creating minimal nodes.py...")
        
    def create_minimal_setup(self):
        """Create completely minimal setup"""
        print("🚀 Creating minimal DNNE setup...")
        
        self.create_minimal_nodes_py()
        
        # Create minimal execution.py if needed
        execution_file = self.root / "execution.py"
        if execution_file.exists():
            self.backup_file(execution_file)
            
            minimal_execution = '''"""
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
'''
            
            with open(execution_file, 'w', encoding='utf-8') as f:
                f.write(minimal_execution)
            print("  ✅ Created minimal execution.py")
    
    def restore_backups(self):
        """Restore all backed up files"""
        print("🔄 Restoring from backups...")
        
        backup_files = list(self.root.rglob(f"*{self.backup_suffix}"))
        
        for backup_file in backup_files:
            original_file = backup_file.with_suffix('')
            if backup_file.suffix == self.backup_suffix:
                original_file = Path(str(backup_file)[:-len(self.backup_suffix)])
            
            shutil.copy2(backup_file, original_file)
            print(f"  ✅ Restored: {original_file.name}")
        
        print(f"Restored {len(backup_files)} files")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="DNNE-UI Import Cleanup")
    parser.add_argument("action", choices=["fix", "minimal", "restore"],
                       help="Action: fix imports, create minimal setup, or restore backups")
    parser.add_argument("--root", default=".", help="DNNE-UI root directory")
    
    args = parser.parse_args()
    
    cleanup = ImportCleanup(args.root)
    
    if args.action == "fix":
        cleanup.fix_all_imports()
    elif args.action == "minimal":
        cleanup.create_minimal_setup()
    elif args.action == "restore":
        cleanup.restore_backups()

if __name__ == "__main__":
    main()