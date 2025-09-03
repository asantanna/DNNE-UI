#!/usr/bin/env python3
"""
Lightweight telemetry client for DNNE nodes.
Provides fire-and-forget UDP telemetry with zero blocking.
"""

import socket
import time
import json
from typing import Optional, Dict, Any
import os


class SimpleRateLimiter:
    """
    Simple rate limiter for fire-and-forget telemetry.
    Limits messages per second to prevent overwhelming the network.
    """
    
    def __init__(self, max_msgs_per_sec: int = 10):
        """
        Initialize rate limiter.
        
        Args:
            max_msgs_per_sec: Maximum messages allowed per second
        """
        self.max_msgs_per_sec = max_msgs_per_sec
        self.last_rate_window_start = time.time()
        self.msgs_sent_this_sec = 0
    
    def should_send(self) -> bool:
        """
        Check if a message can be sent within rate limits.
        
        Returns:
            True if message can be sent, False if rate limit exceeded
        """
        now = time.time()
        # Reset counter if we've moved to a new second
        if now - self.last_rate_window_start >= 1.0:
            self.msgs_sent_this_sec = 0
            self.last_rate_window_start = now
        
        # Check if we can send another message
        if self.msgs_sent_this_sec < self.max_msgs_per_sec:
            self.msgs_sent_this_sec += 1
            return True
        return False


class TelemetryClient:
    """
    Fire-and-forget UDP telemetry client for exported nodes.
    
    Sends metrics to dnne_client via UDP for aggregation and forwarding.
    Designed for minimal overhead and zero blocking on the sending node.
    """
    
    def __init__(self, enabled: bool = False, host: str = "localhost", port: int = 9999,
                 violation_rate_limit: int = 10):
        """
        Initialize telemetry client.
        
        Args:
            enabled: Whether telemetry is enabled (can be disabled via env var)
            host: UDP destination host (default: localhost for local dnne_client)
            port: UDP destination port (default: 9999)
            violation_rate_limit: Max violation messages per second (default: 10)
        """
        # Check environment variable override
        if os.environ.get('DNNE_TELEMETRY_DISABLED', '').lower() in ('1', 'true'):
            enabled = False
            
        self.enabled = enabled
        self.host = host
        self.port = port
        self.socket = None
        
        # Create rate limiter for violations
        self.violation_rate_limiter = SimpleRateLimiter(violation_rate_limit)
        
        if enabled:
            try:
                # Create non-blocking UDP socket
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.socket.setblocking(False)
                self.target = (host, port)
            except Exception as e:
                print(f"Warning: Failed to create telemetry socket: {e}")
                self.enabled = False
    
    def report_throughput(self, node_id: str, items_per_second: float):
        """
        Report node throughput in items/second.
        
        Args:
            node_id: Unique identifier for the node
            items_per_second: Current throughput rate
        """
        if not self.enabled:
            return
        self._send_metric("throughput", node_id, items_per_second)
    
    def report_latency(self, node_id: str, latency_ms: float):
        """
        Report node processing latency in milliseconds.
        
        Args:
            node_id: Unique identifier for the node
            latency_ms: Processing latency in milliseconds
        """
        if not self.enabled:
            return
        self._send_metric("latency", node_id, latency_ms)
    
    def report_queue_depth(self, node_id: str, queue_name: str, depth: int):
        """
        Report queue depth for a node input/output.
        
        Args:
            node_id: Unique identifier for the node
            queue_name: Name of the queue (e.g., "input_data", "output")
            depth: Current number of items in queue
        """
        if not self.enabled:
            return
        
        # Special packet format for queue metrics
        packet_data = {
            "type": "queue",
            "node_id": node_id,
            "queue": queue_name,
            "depth": depth,
            "timestamp": time.time()
        }
        self._send_json(packet_data)
    
    def report_custom(self, node_id: str, metric_name: str, value: float):
        """
        Report custom metric.
        
        Args:
            node_id: Unique identifier for the node
            metric_name: Name of the custom metric
            value: Metric value
        """
        if not self.enabled:
            return
        self._send_metric(metric_name, node_id, value)
    
    def report_metric(self, node_id: str, metric_name: str, value: float, aggregate: bool = True):
        """
        Report a metric with aggregation hint.
        
        Args:
            node_id: Unique identifier for the node
            metric_name: Name of the metric
            value: Metric value
            aggregate: Hint for aggregation behavior (not used in UDP implementation)
        """
        # For UDP implementation, just forward to report_custom
        # The aggregation hint could be used by future implementations
        self.report_custom(node_id, metric_name, value)
    
    def start_window(self, node_id: str, window_type: str):
        """
        Start a telemetry window for aggregation.
        
        Args:
            node_id: Unique identifier for the node
            window_type: Type of window (e.g., "epoch", "episode", "batch")
        """
        # For UDP implementation, this is a no-op
        # Future implementations could use this for stateful aggregation
        pass
    
    def end_window(self, node_id: str, stats_dict: Dict[str, Any]):
        """
        End a telemetry window and report aggregated stats.
        
        Args:
            node_id: Unique identifier for the node
            stats_dict: Dictionary of aggregated statistics to report
        """
        if not self.enabled:
            return
        
        # Report each stat as a separate metric
        for metric_name, value in stats_dict.items():
            if isinstance(value, (int, float)):
                self.report_custom(node_id, metric_name, float(value))
    
    def report_violation(self, node_id: str, violation_type: str, 
                        expected: float, actual: float, extra_args: str = None):
        """
        Report performance target violation with rate limiting.
        
        Args:
            node_id: Unique identifier for the node
            violation_type: Type of violation (e.g., "frequency_below_minimum")
            expected: Expected value
            actual: Actual value
            extra_args: Optional context for finer grouping (e.g., "input_queue", "gpu_0")
        """
        if not self.enabled:
            return
        
        # Apply rate limiting for violations
        if not self.violation_rate_limiter.should_send():
            return  # Drop message if rate limit exceeded
            
        packet_data = {
            "type": "violation",
            "node_id": node_id,
            "violation": violation_type,
            "expected": expected,
            "actual": actual,
            "timestamp": time.time()
        }
        if extra_args:
            packet_data["extra_args"] = extra_args
        self._send_json(packet_data)
    
    def _send_metric(self, metric_type: str, node_id: str, value: float):
        """Send a simple metric packet."""
        # Simple pipe-delimited format for efficiency
        packet = f"{metric_type}|{node_id}|{value}|{time.time()}"
        self._send_raw(packet.encode())
    
    def _send_json(self, data: Dict[str, Any]):
        """Send a JSON packet for complex data."""
        try:
            packet = json.dumps(data)
            self._send_raw(packet.encode())
        except:
            pass  # Fire-and-forget
    
    def _send_raw(self, data: bytes):
        """
        Send raw packet data via UDP.
        
        This is truly fire-and-forget - all errors are ignored to ensure
        the sending node is never blocked by telemetry issues.
        """
        if not self.enabled or not self.socket:
            return
            
        try:
            # Non-blocking send
            self.socket.sendto(data, self.target)
        except:
            # Ignore all errors - this is fire-and-forget
            pass
    
    def close(self):
        """Close the telemetry socket."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
            self.enabled = False


# Global telemetry instance for easy access
_telemetry_instance = None

def get_telemetry() -> TelemetryClient:
    """
    Get the global telemetry client instance.
    
    This should only be called if telemetry is enabled for the node.
    Configuration must be available or this will fail.
    
    Returns:
        TelemetryClient: The global telemetry client
    """
    global _telemetry_instance
    if _telemetry_instance is None:
        # Try to get from dnne_config - this MUST succeed if telemetry is enabled
        try:
            from framework.dnne_config import DNNEConfig
            config = DNNEConfig()
            # In exported configs, the 'exported' prefix is removed
            host = config.get('agent_client.telemetry_host')
            port = config.get('agent_client.telemetry_port')
            
            if host is None or port is None:
                raise RuntimeError(
                    "Telemetry enabled but configuration missing from exported_config.json. "
                    "Need 'exported.agent_client.telemetry_host' and 'telemetry_port'."
                )
        except ImportError:
            raise RuntimeError(
                "Telemetry enabled but dnne_config module not found. "
                "Cannot initialize telemetry without configuration."
            )
        
        # Create instance with valid configuration
        _telemetry_instance = TelemetryClient(enabled=True, host=host, port=port)
    return _telemetry_instance


# Convenience functions for direct use
def report_throughput(node_id: str, items_per_second: float):
    """Report throughput metric."""
    get_telemetry().report_throughput(node_id, items_per_second)

def report_latency(node_id: str, latency_ms: float):
    """Report latency metric."""
    get_telemetry().report_latency(node_id, latency_ms)

def report_queue_depth(node_id: str, queue_name: str, depth: int):
    """Report queue depth metric."""
    get_telemetry().report_queue_depth(node_id, queue_name, depth)

def report_custom(node_id: str, metric_name: str, value: float):
    """Report custom metric."""
    get_telemetry().report_custom(node_id, metric_name, value)

def report_violation(node_id: str, violation_type: str, 
                    expected: float, actual: float, extra_args: str = None):
    """Report performance violation with optional context."""
    get_telemetry().report_violation(node_id, violation_type, expected, actual, extra_args)