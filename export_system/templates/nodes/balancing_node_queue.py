#!/usr/bin/env python3
"""
Balancing Node - Active passthrough node with performance monitoring
"""

import torch
import time
import asyncio
from typing import Dict, Any, Optional
from collections import deque
from framework import QueueNode
from framework.globals import Global

# Template variables
template_vars = {
    "NODE_ID": "balancing_1",
    "CLASS_NAME": "BalancingNode",
    "ENABLED": True,
    "MIN_HZ": -1.0,
    "MAX_HZ": -1.0,
    "TARGET_HZ": -1.0,
    "TARGET_PERCENTAGE": -1.0,
    "PRIORITY": 0,
    "GUARANTEED": False,
    "MAX_LATENCY_MS": -1.0,
    "WINDOW_SIZE": 100,
    "LOG_VIOLATIONS": True,
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """
    Balancing Node that measures and enforces performance targets
    while forwarding data unchanged
    """
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["input"])
        self.setup_outputs(["output"])
        
        # Configuration
        self.enabled = {ENABLED}
        self.min_hz = {MIN_HZ}
        self.max_hz = {MAX_HZ}
        self.target_hz = {TARGET_HZ}
        self.target_percentage = {TARGET_PERCENTAGE}
        self.priority = {PRIORITY}
        self.guaranteed = {GUARANTEED}
        self.max_latency_ms = {MAX_LATENCY_MS}
        self.window_size = {WINDOW_SIZE}
        self.log_violations = {LOG_VIOLATIONS}
        
        # Performance tracking
        self.last_execution_time = None
        self.execution_count = 0
        self.start_time = time.time()
        
        # Moving average windows
        self.frequency_window = deque(maxlen=self.window_size)
        self.latency_window = deque(maxlen=self.window_size)
        
        # Violation tracking
        self.violations = []
        self.last_violation_log = 0
        self.violation_log_interval = 10.0  # seconds
        
        # Metrics for reporting
        self.current_frequency = 0.0
        self.current_latency = 0.0
        self.average_frequency = 0.0
        self.average_latency = 0.0
        
        # Register with Global for adaptive yielding
        self._register_with_global()
    
    def _register_with_global(self):
        """Register performance targets with metrics logger and Global system"""
        if not self.enabled:
            return
            
        if any([self.min_hz >= 0, self.max_hz >= 0, self.target_hz >= 0, 
                self.target_percentage >= 0, self.max_latency_ms >= 0]):
            config = {
                "node_id": self.node_id,
                "frequency": {
                    "min_hz": self.min_hz if self.min_hz >= 0 else None,
                    "max_hz": self.max_hz if self.max_hz >= 0 else None,
                    "target_hz": self.target_hz if self.target_hz >= 0 else None,
                },
                "throughput": {
                    "target_percentage": self.target_percentage if self.target_percentage >= 0 else None,
                },
                "scheduling": {
                    "priority": self.priority,
                    "guaranteed": self.guaranteed,
                },
                "latency": {
                    "max_latency_ms": self.max_latency_ms if self.max_latency_ms >= 0 else None,
                }
            }
            
            # Register with metrics logger (fail-fast if not available)
            from framework.metrics_logger import get_metrics_logger
            logger = get_metrics_logger()
            logger.register_node(self.node_id, f"BalancingNode_{self.node_id}", config)
            
            # TODO: Global.register_monitor_node(self.node_id, config)
            self.node_logger.info(f"Registered balancing targets for node {self.node_id}")
    
    async def compute(self, input) -> Dict[str, Any]:
        """Monitor performance and forward input unchanged (measurement only)"""
        
        # If disabled, just passthrough
        if not self.enabled:
            return {"output": input}
        
        start_time = time.time()
        
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
        
        # Report metrics to Global
        Global.update_node_execution(self.node_id)
        
        # Use metrics logger (fail-fast if not available)
        from framework.metrics_logger import get_metrics_logger
        logger = get_metrics_logger()
        
        # Record current metrics
        logger.record_metric(self.node_id, f"BalancingNode_{self.node_id}", "frequency", 
                           self.current_frequency)
        logger.record_metric(self.node_id, f"BalancingNode_{self.node_id}", "latency", 
                           self.current_latency)
        
        # Check and record violations
        violations = self._check_violations()
        for v in violations:
            logger.record_violation(
                self.node_id, f"BalancingNode_{self.node_id}",
                v["type"], v["expected"], v["actual"], v["guaranteed"]
            )
        
        # NOTE: Removed max_hz throttling - measurement only
        
        # Measure processing latency
        end_time = time.time()
        self.current_latency = (end_time - start_time) * 1000  # ms
        self.latency_window.append(self.current_latency)
        
        # Report metrics periodically
        if self.execution_count % 100 == 0:
            self._report_metrics()
        
        # Forward input unchanged
        return {"output": input}
    
    def _check_violations(self) -> list:
        """Check for performance target violations"""
        violations = []
        
        # Frequency violations
        if self.min_hz >= 0 and self.average_frequency < self.min_hz:
            violations.append({
                "type": "frequency_below_minimum",
                "expected": self.min_hz,
                "actual": self.average_frequency,
                "guaranteed": self.guaranteed
            })
        
        if self.max_hz >= 0 and self.average_frequency > self.max_hz:
            violations.append({
                "type": "frequency_above_maximum",
                "expected": self.max_hz,
                "actual": self.average_frequency,
                "guaranteed": self.guaranteed
            })
        
        # Latency violations
        if self.max_latency_ms >= 0 and self.average_latency > self.max_latency_ms:
            violations.append({
                "type": "latency_exceeded",
                "expected": self.max_latency_ms,
                "actual": self.average_latency,
                "guaranteed": self.guaranteed
            })
        
        return violations
    
    def _log_violations(self, violations: list):
        """Log violations with batching to prevent spam"""
        self.violations.extend(violations)
        
        current_time = time.time()
        if current_time - self.last_violation_log > self.violation_log_interval:
            # Dump accumulated violations
            if self.violations:
                self.node_logger.warning(f"⚠️  Balancing Node {self.node_id} - {len(self.violations)} violations in last {self.violation_log_interval}s:")
                
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
                        self.node_logger.warning(f"  - Frequency below minimum: {avg_actual:.1f} Hz < {v_list[0]['expected']} Hz")
                    elif v_type == "frequency_above_maximum":
                        avg_actual = sum(v["actual"] for v in v_list) / len(v_list)
                        self.node_logger.warning(f"  - Frequency above maximum: {avg_actual:.1f} Hz > {v_list[0]['expected']} Hz")
                    elif v_type == "latency_exceeded":
                        avg_actual = sum(v["actual"] for v in v_list) / len(v_list)
                        self.node_logger.warning(f"  - Latency exceeded: {avg_actual:.1f} ms > {v_list[0]['expected']} ms")
                
                # Clear violations
                self.violations = []
                self.last_violation_log = current_time
    
    def _report_metrics(self):
        """Report metrics summary"""
        uptime = time.time() - self.start_time
        self.node_logger.info(f"📊 Balancing Node {self.node_id} Metrics:")
        self.node_logger.info(f"  - Executions: {self.execution_count}")
        self.node_logger.info(f"  - Average frequency: {self.average_frequency:.1f} Hz")
        self.node_logger.info(f"  - Current frequency: {self.current_frequency:.1f} Hz")
        self.node_logger.info(f"  - Average latency: {self.average_latency:.2f} ms")
        self.node_logger.info(f"  - Uptime: {uptime:.1f} seconds")