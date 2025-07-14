#!/usr/bin/env python3
"""
Timing Wrapper Module - Wraps C++ functions for profiling

Provides monkey patching capabilities to time C++ extension calls
that are invisible to Python's cProfile.
"""

import time
import functools
from typing import Dict, Any, Callable
from collections import defaultdict

class TimingWrapper:
    """Tracks timing for wrapped C++ functions"""
    
    def __init__(self):
        self.timing_data = defaultdict(lambda: {'total_time': 0.0, 'call_count': 0})
        self.enabled = True
        self._wrapped_functions = []
    
    def wrap_function(self, obj: Any, method_name: str, display_name: str = None) -> None:
        """
        Wrap a method on an object with timing instrumentation
        
        Args:
            obj: Object containing the method to wrap
            method_name: Name of the method to wrap
            display_name: Name to use in timing reports (defaults to method_name)
        """
        if not hasattr(obj, method_name):
            print(f"Warning: {obj.__class__.__name__} has no method '{method_name}'")
            return
            
        original_method = getattr(obj, method_name)
        if hasattr(original_method, '_is_timing_wrapped'):
            # Already wrapped
            return
            
        display_name = display_name or method_name
        
        @functools.wraps(original_method)
        def timed_method(*args, **kwargs):
            if not self.enabled:
                return original_method(*args, **kwargs)
                
            start_time = time.perf_counter()
            try:
                result = original_method(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time
                self.timing_data[display_name]['total_time'] += elapsed
                self.timing_data[display_name]['call_count'] += 1
        
        # Mark as wrapped to avoid double-wrapping
        timed_method._is_timing_wrapped = True
        timed_method._original_method = original_method
        
        setattr(obj, method_name, timed_method)
        self._wrapped_functions.append((obj, method_name, original_method))
    
    def wrap_isaacgym_calls(self, gym_obj: Any) -> None:
        """
        Wrap common Isaac Gym API calls
        
        Args:
            gym_obj: The gym object from gymapi.acquire_gym()
        """
        # List of methods to wrap with their display names
        methods_to_wrap = [
            # Core simulation calls
            ('simulate', 'gym.simulate'),
            ('fetch_results', 'gym.fetch_results'),
            
            # Tensor refresh calls
            ('refresh_dof_state_tensor', 'gym.refresh_dof_state_tensor'),
            ('refresh_actor_root_state_tensor', 'gym.refresh_actor_root_state_tensor'),
            ('refresh_net_contact_force_tensor', 'gym.refresh_net_contact_force_tensor'),
            ('refresh_rigid_body_state_tensor', 'gym.refresh_rigid_body_state_tensor'),
            ('refresh_jacobian_tensors', 'gym.refresh_jacobian_tensors'),
            ('refresh_mass_matrix_tensors', 'gym.refresh_mass_matrix_tensors'),
            
            # Environment management
            ('step_graphics', 'gym.step_graphics'),
            ('draw_viewer', 'gym.draw_viewer'),
            ('sync_frame_time', 'gym.sync_frame_time')
        ]
        
        # Wrap each method, skipping those that don't exist
        wrapped_count = 0
        for method_name, display_name in methods_to_wrap:
            try:
                self.wrap_function(gym_obj, method_name, display_name)
                wrapped_count += 1
            except AttributeError as e:
                # Method doesn't exist on this gym object, skip it
                print(f"[TIMING] Could not wrap {method_name}: {e}")
            except Exception as e:
                # Other errors
                print(f"[TIMING] Error wrapping {method_name}: {type(e).__name__}: {e}")
        
        print(f"[TIMING] Successfully wrapped {wrapped_count} Isaac Gym methods")
    
    def get_timing_results(self) -> Dict[str, Dict[str, Any]]:
        """Get timing results in milliseconds"""
        results = {}
        for name, data in self.timing_data.items():
            if data['call_count'] > 0:
                results[name] = {
                    'total_ms': data['total_time'] * 1000,
                    'call_count': data['call_count'],
                    'avg_ms': (data['total_time'] * 1000) / data['call_count']
                }
        return results
    
    def reset(self) -> None:
        """Reset timing data"""
        self.timing_data.clear()
    
    def restore_original_functions(self) -> None:
        """Restore all wrapped functions to their original implementations"""
        for obj, method_name, original_method in self._wrapped_functions:
            setattr(obj, method_name, original_method)
        self._wrapped_functions.clear()
    
    def __enter__(self):
        """Context manager support"""
        self.enabled = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.restore_original_functions()


# Global instance for convenience
_global_wrapper = TimingWrapper()

def get_global_wrapper() -> TimingWrapper:
    """Get the global timing wrapper instance"""
    return _global_wrapper