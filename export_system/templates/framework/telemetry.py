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


class TelemetryClient:
    """
    Fire-and-forget UDP telemetry client for exported nodes.
    
    Sends metrics to dnne_client via UDP for aggregation and forwarding.
    Designed for minimal overhead and zero blocking on the sending node.
    """
    
    def __init__(self, enabled: bool = True, host: str = "localhost", port: int = 9999):
        """
        Initialize telemetry client.
        
        Args:
            enabled: Whether telemetry is enabled (can be disabled via env var)
            host: UDP destination host (default: localhost for local dnne_client)
            port: UDP destination port (default: 9999)
        """
        # Check environment variable override
        if os.environ.get('DNNE_TELEMETRY_DISABLED', '').lower() in ('1', 'true'):
            enabled = False
            
        self.enabled = enabled
        self.host = host
        self.port = port
        self.socket = None
        
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
    
    def report_violation(self, node_id: str, violation_type: str, 
                        expected: float, actual: float, guaranteed: bool = False):
        """
        Report performance target violation.
        
        Args:
            node_id: Unique identifier for the node
            violation_type: Type of violation (e.g., "frequency_below_minimum")
            expected: Expected value
            actual: Actual value
            guaranteed: Whether this was a guaranteed target
        """
        if not self.enabled:
            return
            
        packet_data = {
            "type": "violation",
            "node_id": node_id,
            "violation": violation_type,
            "expected": expected,
            "actual": actual,
            "guaranteed": guaranteed,
            "timestamp": time.time()
        }
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
    
    Returns:
        TelemetryClient: The global telemetry client
    """
    global _telemetry_instance
    if _telemetry_instance is None:
        # Check for environment variable configuration
        host = os.environ.get('DNNE_TELEMETRY_HOST', 'localhost')
        port = int(os.environ.get('DNNE_TELEMETRY_PORT', '9999'))
        _telemetry_instance = TelemetryClient(host=host, port=port)
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
                    expected: float, actual: float, guaranteed: bool = False):
    """Report performance violation."""
    get_telemetry().report_violation(node_id, violation_type, expected, actual, guaranteed)