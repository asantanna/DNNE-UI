"""
Script Loader Utility for Custom Computation Node
Provides cached loading of custom computation scripts with validation.
"""

import os
import importlib.util
import dnne_config

# Cache for loaded modules
_script_cache = {}

def load_custom_script(src_path):
    """Load a custom computation script, with caching.
    
    Args:
        src_path: Path to the Python script file (absolute or relative to custom_compute_funcs)
        
    Returns:
        The loaded module with required functions validated
        
    Raises:
        ImportError: If script cannot be loaded
        AttributeError: If script missing required functions
    """
    # Check cache first
    if src_path in _script_cache:
        return _script_cache[src_path]
    
    # Resolve full path if needed
    full_path = src_path
    if os.sep not in src_path:
        dnne_root = dnne_config.get_dnne_root()
        full_path = os.path.join(dnne_root, "user", "default", "custom_compute_funcs", src_path)
    
    # Convert to absolute path
    full_path = os.path.abspath(full_path)
    
    # Check file exists
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Script file not found: {full_path}")
    
    # Load module
    module_name = f"custom_compute_{os.path.basename(full_path).replace('.py', '')}"
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    if not spec or not spec.loader:
        raise ImportError(f"Failed to load script: {full_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Verify required functions - FAIL FAST
    required_functions = ['get_output_type', 'get_script_output_schema', 'compute']
    for func_name in required_functions:
        if not hasattr(module, func_name):
            raise AttributeError(f"Script {full_path} missing required function: {func_name}()")
    
    # Cache and return
    _script_cache[src_path] = module
    return module

def clear_script_cache():
    """Clear the script cache (useful for development/reload)."""
    _script_cache.clear()