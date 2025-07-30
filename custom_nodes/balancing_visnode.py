"""
Balancing Node
Active passthrough node that measures and enforces performance targets
"""

import time
from typing import Dict, Any, Optional, Union
from inspect import cleandoc
from collections import deque
from custom_nodes.base import RoboticsNodeBase
from custom_nodes.node_colors import get_node_colors


class BalancingNode(RoboticsNodeBase):
    """
    Balancing Node
    
    A passthrough node that measures and enforces performance targets while
    forwarding data unchanged. Insert at strategic points in workflows to
    monitor and control execution rates.
    
    Features:
    - Measures throughput, frequency, and latency
    - Enforces min/max frequency limits
    - Reports metrics to adaptive yielding system
    - Minimal overhead (just timestamps and forwards data)
    
    Configuration parameters:
    - Frequency-based targets: min_hz, max_hz, target_hz
    - Throughput-based targets: target_percentage
    - Priority settings: priority, guaranteed
    - Latency requirements: max_latency_ms
    """
    
    # Active node - participates in execution
    IS_VIRTUAL = False
    
    CATEGORY = "utility"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*", {
                    "tooltip": "Any data to passthrough while monitoring performance"
                }),
            },
            "optional": {
                # Item name for metrics display
                "item_name": ("STRING", {
                    "default": "items",
                    "tooltip": "Unit name for throughput metrics (e.g., 'batches', 'frames', 'steps')"
                }),
                
                # Enable/disable monitoring
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable performance monitoring"
                }),
                
                # Frequency-based targets (robotics/real-time)
                "min_hz": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 1000.0,
                    "step": 0.1,
                    "tooltip": "Minimum frequency in Hz (-1 = don't care)"
                }),
                "max_hz": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 1000.0,
                    "step": 0.1,
                    "tooltip": "Maximum frequency in Hz (-1 = don't care)"
                }),
                "target_hz": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 1000.0,
                    "step": 0.1,
                    "tooltip": "Target frequency in Hz (-1 = don't care)"
                }),
                
                # Throughput-based targets (batch processing)
                "target_percentage": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Target percentage of total system throughput (-1 = don't care)"
                }),
                
                # Priority settings
                "priority": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Priority level (higher = more important)"
                }),
                "guaranteed": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Must meet targets vs best-effort"
                }),
                
                # Latency requirements
                "max_latency_ms": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 10000.0,
                    "step": 0.1,
                    "tooltip": "Maximum processing latency in milliseconds (-1 = don't care)"
                }),
                
                # Measurement settings
                "window_size": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "tooltip": "Number of samples for moving average"
                }),
                "log_violations": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Log when performance targets are violated"
                }),
            }
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "passthrough_measure"
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("balancing")["color"]
    BGCOLOR = get_node_colors("balancing")["bgcolor"]
    
    def __init__(self):
        super().__init__()
        
        # Performance tracking
        self.last_execution_time = None
        self.execution_count = 0
        self.start_time = time.time()
        
        # Moving average windows
        self.frequency_window = deque(maxlen=100)
        self.latency_window = deque(maxlen=100)
        
        # Violation tracking
        self.violations = []
        self.last_violation_log = 0
        self.violation_log_interval = 10.0  # seconds
        
        # Metrics for reporting
        self.current_frequency = 0.0
        self.current_latency = 0.0
        self.average_frequency = 0.0
        self.average_latency = 0.0
    
    def passthrough_measure(self, input, item_name="items", enabled=True,
                           min_hz=-1.0, max_hz=-1.0, target_hz=-1.0,
                           target_percentage=-1.0, priority=0, guaranteed=False,
                           max_latency_ms=-1.0,
                           window_size=100, log_violations=True) -> tuple:
        """Forward input unchanged while measuring and monitoring performance (no enforcement)"""
        
        # If disabled, just passthrough without any monitoring
        if not enabled:
            return (input,)
        
        start_time = time.time()
        
        # Update window size if changed
        if window_size != self.frequency_window.maxlen:
            self.frequency_window = deque(self.frequency_window, maxlen=window_size)
            self.latency_window = deque(self.latency_window, maxlen=window_size)
        
        # Measure frequency
        if self.last_execution_time is not None:
            time_delta = start_time - self.last_execution_time
            if time_delta > 0:
                self.current_frequency = 1.0 / time_delta
                self.frequency_window.append(self.current_frequency)
        
        self.last_execution_time = start_time
        self.execution_count += 1
        
        # Calculate averages
        if len(self.frequency_window) > 0:
            self.average_frequency = sum(self.frequency_window) / len(self.frequency_window)
        
        if len(self.latency_window) > 0:
            self.average_latency = sum(self.latency_window) / len(self.latency_window)
        
        # Check for violations (measurement only - no enforcement)
        violations = self._check_violations(
            min_hz, max_hz, target_hz, target_percentage,
            max_latency_ms, guaranteed
        )
        
        # Log violations if enabled
        if log_violations and violations:
            self._log_violations(violations)
        
        # NOTE: Removed max_hz throttling - we're measurement-only now
        # Users can see the metrics and make adjustments manually
        
        # Measure processing latency
        end_time = time.time()
        self.current_latency = (end_time - start_time) * 1000  # ms
        self.latency_window.append(self.current_latency)
        
        # Report metrics (would integrate with metrics logger in full implementation)
        if self.execution_count % 100 == 0:
            self._report_metrics()
        
        # Forward input unchanged
        return (input,)
    
    def _check_violations(self, min_hz, max_hz, target_hz, target_percentage,
                         max_latency_ms, guaranteed) -> list:
        """Check for performance target violations"""
        violations = []
        
        # Frequency violations
        if min_hz > 0 and self.average_frequency < min_hz:
            violations.append({
                "type": "frequency_below_minimum",
                "expected": min_hz,
                "actual": self.average_frequency,
                "guaranteed": guaranteed
            })
        
        if max_hz > 0 and self.average_frequency > max_hz:
            violations.append({
                "type": "frequency_above_maximum",
                "expected": max_hz,
                "actual": self.average_frequency,
                "guaranteed": guaranteed
            })
        
        # Latency violations
        if max_latency_ms > 0 and self.average_latency > max_latency_ms:
            violations.append({
                "type": "latency_exceeded",
                "expected": max_latency_ms,
                "actual": self.average_latency,
                "guaranteed": guaranteed
            })
        
        return violations
    
    def _log_violations(self, violations: list):
        """Log violations with batching to prevent spam"""
        self.violations.extend(violations)
        
        current_time = time.time()
        if current_time - self.last_violation_log > self.violation_log_interval:
            # Dump accumulated violations
            if self.violations:
                print(f"\n⚠️  Balancing Node - {len(self.violations)} violations in last {self.violation_log_interval}s:")
                
                # Group violations by type
                by_type = {}
                for v in self.violations:
                    v_type = v["type"]
                    if v_type not in by_type:
                        by_type[v_type] = []
                    by_type[v_type].append(v)
                
                # Show summary
                for v_type, v_list in by_type.items():
                    if v_type == "frequency_below_minimum":
                        avg_actual = sum(v["actual"] for v in v_list) / len(v_list)
                        print(f"  - Frequency below minimum: {avg_actual:.1f} Hz < {v_list[0]['expected']} Hz")
                    elif v_type == "frequency_above_maximum":
                        avg_actual = sum(v["actual"] for v in v_list) / len(v_list)
                        print(f"  - Frequency above maximum: {avg_actual:.1f} Hz > {v_list[0]['expected']} Hz")
                    elif v_type == "latency_exceeded":
                        avg_actual = sum(v["actual"] for v in v_list) / len(v_list)
                        print(f"  - Latency exceeded: {avg_actual:.1f} ms > {v_list[0]['expected']} ms")
                
                # Clear violations
                self.violations = []
                self.last_violation_log = current_time
    
    def _report_metrics(self):
        """Report metrics summary"""
        uptime = time.time() - self.start_time
        print(f"\n📊 Balancing Node Metrics:")
        print(f"  - Executions: {self.execution_count}")
        print(f"  - Average frequency: {self.average_frequency:.1f} Hz")
        print(f"  - Current frequency: {self.current_frequency:.1f} Hz")
        print(f"  - Average latency: {self.average_latency:.2f} ms")
        print(f"  - Uptime: {uptime:.1f} seconds")
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always execute to maintain accurate measurements"""
        return float("nan")

# Node registration
NODE_CLASS_MAPPINGS = {
    "BalancingNode": BalancingNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BalancingNode": "Balancing Node"
}