#!/usr/bin/env python3
"""
Centralized Metrics Logger for DNNE Balancing System
Collects and logs performance metrics from all balancing nodes
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime
import threading

@dataclass
class MetricSample:
    """Single metric sample"""
    timestamp: float
    node_id: str
    node_name: str
    metric_type: str  # 'frequency', 'latency', 'throughput', 'queue_depth'
    value: float
    metadata: Dict[str, Any] = None

@dataclass
class ViolationEvent:
    """Performance target violation event"""
    timestamp: float
    node_id: str
    node_name: str
    violation_type: str  # 'frequency_below_min', 'latency_exceeded', etc.
    expected: float
    actual: float
    guaranteed: bool

class MetricsLogger:
    """
    Centralized metrics collection and logging for balancing nodes.
    Focuses on measurement and analysis, not enforcement.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize metrics logger"""
        self.start_time = time.time()
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.violations: List[ViolationEvent] = []
        
        # Node registry
        self.node_configs: Dict[str, Dict] = {}
        
        # Logging configuration
        self.log_dir = Path("metrics_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger("MetricsLogger")
        self.logger.setLevel(logging.INFO)
        
        # File handler for metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = self.log_dir / f"metrics_{timestamp}.log"
        fh = logging.FileHandler(metrics_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Console handler for important events
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # Summary intervals
        self.last_summary_time = time.time()
        self.summary_interval = 30.0  # seconds
        
        self.logger.info(f"=== Metrics Logger Started ===")
    
    def register_node(self, node_id: str, node_name: str, config: Dict[str, Any]):
        """Register a node with its performance targets"""
        self.node_configs[node_id] = {
            "name": node_name,
            "config": config,
            "registered_at": time.time()
        }
        
        self.logger.info(f"Registered node '{node_name}' (ID: {node_id}) with config: {json.dumps(config, indent=2)}")
    
    def record_metric(self, node_id: str, node_name: str, metric_type: str, 
                     value: float, metadata: Optional[Dict] = None):
        """Record a metric sample"""
        sample = MetricSample(
            timestamp=time.time(),
            node_id=node_id,
            node_name=node_name,
            metric_type=metric_type,
            value=value,
            metadata=metadata or {}
        )
        
        key = f"{node_id}:{metric_type}"
        self.metrics[key].append(sample)
        
        # Log detailed metrics at debug level
        self.logger.debug(f"Metric: {node_name} - {metric_type}: {value:.2f}")
    
    def record_violation(self, node_id: str, node_name: str, violation_type: str,
                        expected: float, actual: float, guaranteed: bool = False):
        """Record a performance violation"""
        violation = ViolationEvent(
            timestamp=time.time(),
            node_id=node_id,
            node_name=node_name,
            violation_type=violation_type,
            expected=expected,
            actual=actual,
            guaranteed=guaranteed
        )
        
        self.violations.append(violation)
        
        # Log violations immediately
        severity = "CRITICAL" if guaranteed else "WARNING"
        self.logger.warning(
            f"{severity} - Violation in '{node_name}': {violation_type} "
            f"(expected: {expected:.2f}, actual: {actual:.2f})"
        )
    
    def get_node_metrics(self, node_id: str, metric_type: str, 
                        time_window: Optional[float] = None) -> List[MetricSample]:
        """Get metrics for a specific node and type"""
        key = f"{node_id}:{metric_type}"
        samples = list(self.metrics.get(key, []))
        
        if time_window:
            cutoff_time = time.time() - time_window
            samples = [s for s in samples if s.timestamp >= cutoff_time]
        
        return samples
    
    def calculate_statistics(self, samples: List[MetricSample]) -> Dict[str, float]:
        """Calculate statistics for a list of samples"""
        if not samples:
            return {
                "count": 0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
                "latest": 0.0
            }
        
        values = [s.value for s in samples]
        
        # Calculate standard deviation
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = variance ** 0.5
        
        return {
            "count": len(values),
            "mean": mean,
            "min": min(values),
            "max": max(values),
            "std": std,
            "latest": values[-1]
        }
    
    def generate_summary(self):
        """Generate and log summary statistics"""
        current_time = time.time()
        
        if current_time - self.last_summary_time < self.summary_interval:
            return
        
        self.last_summary_time = current_time
        uptime = current_time - self.start_time
        
        self.logger.info("=" * 60)
        self.logger.info(f"=== Performance Summary (Uptime: {uptime:.1f}s) ===")
        
        # Summary for each registered node
        for node_id, node_info in self.node_configs.items():
            node_name = node_info["name"]
            self.logger.info(f"\nNode: {node_name} (ID: {node_id})")
            
            # Frequency metrics
            freq_samples = self.get_node_metrics(node_id, "frequency", self.summary_interval)
            if freq_samples:
                stats = self.calculate_statistics(freq_samples)
                self.logger.info(f"  Frequency: {stats['mean']:.1f} Hz "
                               f"(min: {stats['min']:.1f}, max: {stats['max']:.1f}, "
                               f"std: {stats['std']:.1f})")
            
            # Latency metrics
            lat_samples = self.get_node_metrics(node_id, "latency", self.summary_interval)
            if lat_samples:
                stats = self.calculate_statistics(lat_samples)
                self.logger.info(f"  Latency: {stats['mean']:.2f} ms "
                               f"(min: {stats['min']:.2f}, max: {stats['max']:.2f}, "
                               f"std: {stats['std']:.2f})")
            
            # Throughput metrics
            tput_samples = self.get_node_metrics(node_id, "throughput", self.summary_interval)
            if tput_samples:
                stats = self.calculate_statistics(tput_samples)
                self.logger.info(f"  Throughput: {stats['mean']:.1f} items/sec "
                               f"(min: {stats['min']:.1f}, max: {stats['max']:.1f})")
        
        # Violation summary
        recent_violations = [v for v in self.violations 
                           if v.timestamp >= current_time - self.summary_interval]
        if recent_violations:
            self.logger.info(f"\nViolations in last {self.summary_interval}s:")
            violation_counts = defaultdict(int)
            for v in recent_violations:
                key = f"{v.node_name}:{v.violation_type}"
                violation_counts[key] += 1
            
            for key, count in violation_counts.items():
                self.logger.info(f"  {key}: {count} times")
        
        self.logger.info("=" * 60)
    
    def export_metrics(self, output_file: Optional[str] = None) -> Dict:
        """Export all metrics to JSON format"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.log_dir / f"metrics_export_{timestamp}.json"
        
        export_data = {
            "start_time": self.start_time,
            "export_time": time.time(),
            "uptime": time.time() - self.start_time,
            "nodes": {},
            "violations": []
        }
        
        # Export metrics for each node
        for node_id, node_info in self.node_configs.items():
            node_data = {
                "name": node_info["name"],
                "config": node_info["config"],
                "metrics": {}
            }
            
            # Get all metric types for this node
            for key in self.metrics:
                if key.startswith(f"{node_id}:"):
                    metric_type = key.split(":", 1)[1]
                    samples = [asdict(s) for s in self.metrics[key]]
                    node_data["metrics"][metric_type] = samples
            
            export_data["nodes"][node_id] = node_data
        
        # Export violations
        export_data["violations"] = [asdict(v) for v in self.violations]
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported metrics to {output_file}")
        return export_data
    
    def shutdown(self):
        """Shutdown logger and export final metrics"""
        self.logger.info("=== Metrics Logger Shutting Down ===")
        self.generate_summary()
        self.export_metrics()

# Global instance getter
def get_metrics_logger():
    """Get the global metrics logger instance"""
    return MetricsLogger.get_instance()