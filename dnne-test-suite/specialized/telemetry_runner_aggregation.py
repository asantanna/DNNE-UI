#!/usr/bin/env python3
"""
Aggregation telemetry test - deployed and executed by test_telemetry.py orchestrator.
This tests the aggregation of telemetry data at the agent and DNNE server levels.
Based on the telemetry architecture in dnne-docs/architecture/telemetry.md
"""

import socket
import json
import time
import random
import sys
import asyncio
from typing import Optional, Dict, Any


class TelemetryClient:
    """
    Production-like telemetry client that sends UDP packets to client agent.
    """
    
    def __init__(self, host: str = "localhost", port: int = 9999):
        """Initialize UDP telemetry client."""
        self.host = host
        self.port = port
        self.target = (host, port)
        
        # Create non-blocking UDP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        
        print(f"📡 Telemetry client initialized for {host}:{port}")
    
    def send_metric(self, metric_type: str, node_id: str, value: float):
        """
        Send a metric in pipe-delimited format.
        Format: metric_type|node_id|value|timestamp
        """
        packet = f"{metric_type}|{node_id}|{value}|{time.time()}"
        self._send_raw(packet.encode())
    
    def send_violation(self, node_id: str, violation_type: str, 
                      expected: float, actual: float, 
                      extra_args: Optional[str] = None):
        """
        Send a violation in JSON format.
        """
        packet = {
            "type": "violation",
            "node_id": node_id,
            "violation": violation_type,
            "expected": expected,
            "actual": actual,
            "timestamp": time.time()
        }
        
        if extra_args:
            packet["extra_args"] = extra_args
        
        self._send_raw(json.dumps(packet).encode())
    
    def send_queue_depth(self, node_id: str, queue_name: str, depth: int):
        """
        Send queue depth metric in JSON format.
        """
        packet = {
            "type": "queue",
            "node_id": node_id,
            "queue": queue_name,
            "depth": depth,
            "timestamp": time.time()
        }
        self._send_raw(json.dumps(packet).encode())
    
    def send_custom(self, node_id: str, metric_name: str, value: float):
        """
        Send custom metric using pipe-delimited format.
        """
        packet = f"{metric_name}|{node_id}|{value}|{time.time()}"
        self._send_raw(packet.encode())
    
    def _send_raw(self, data: bytes):
        """
        Send raw bytes via UDP (fire-and-forget).
        """
        try:
            self.socket.sendto(data, self.target)
        except BlockingIOError:
            # Non-blocking socket, ignore if buffer full
            pass
        except Exception as e:
            print(f"⚠️ UDP send error: {e}")
    
    def close(self):
        """Close the UDP socket."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None


async def run_aggregation_test():
    """
    Aggregation test - sends metrics and violations with different patterns
    to test aggregation at agent and server levels.
    Tests features from dnne-docs/architecture/telemetry.md:
    - Agent batches every 100ms
    - Violations grouped by node:type or node:type:extra_args
    - Summaries after 5 details, then every 10 seconds
    - File storage in efficient formats
    """
    print("🚀 Aggregation Telemetry Test Started")
    print("=" * 60)
    print("📊 Testing telemetry aggregation and batching")
    print("   - Agent batching (100ms intervals)")
    print("   - Violation grouping (with/without extra_args)")
    print("   - Summary generation (after 5 details, then every 10s)")
    
    # Create telemetry client
    telemetry = TelemetryClient()
    
    try:
        # Phase 1: Metrics aggregation test (test 100ms batching)
        print("\n📈 Phase 1: Metrics batching test (5 nodes, 4 metric types)")
        print("   Testing agent 100ms batch intervals...")
        
        nodes = ["node_10", "node_11", "node_12", "node_13", "node_14"]
        metric_types = ["throughput", "latency", "memory", "cpu"]
        
        phase1_start = time.time()
        # Send bursts of metrics to test batching
        for burst in range(10):
            # Send a burst of metrics
            for node_id in nodes:
                for metric_type in metric_types:
                    base_value = {
                        "throughput": 100,
                        "latency": 20,
                        "memory": 512,
                        "cpu": 50
                    }[metric_type]
                    
                    value = base_value + random.uniform(-10, 10)
                    telemetry.send_metric(metric_type, node_id, value)
            
            # Wait 200ms between bursts (should create 2 batches per burst)
            await asyncio.sleep(0.2)
        
        phase1_duration = time.time() - phase1_start
        print(f"   ✓ Sent {10 * len(nodes) * len(metric_types)} metrics in {phase1_duration:.2f}s")
        print(f"   📊 Should see ~{int(phase1_duration * 10)} batches at agent (100ms intervals)")
        
        # Wait for aggregation
        await asyncio.sleep(1)
        
        # Phase 2: Queue depth monitoring (backpressure simulation)
        print("\n📈 Phase 2: Queue depth monitoring (3 nodes, 5 queues each)")
        print("   Testing queue backpressure metrics...")
        
        queue_nodes = ["node_10", "node_11", "node_12"]
        queue_names = ["input_queue", "process_queue", "output_queue", "error_queue", "retry_queue"]
        
        phase2_start = time.time()
        for i in range(20):
            for node_id in queue_nodes:
                for queue_name in queue_names:
                    # Simulate varying queue depths
                    base_depth = {"input_queue": 20, "process_queue": 15, 
                                 "output_queue": 10, "error_queue": 2, "retry_queue": 5}
                    depth = base_depth.get(queue_name, 10) + random.randint(-5, 15)
                    depth = max(0, depth)  # Keep non-negative
                    telemetry.send_queue_depth(node_id, queue_name, depth)
            
            await asyncio.sleep(0.15)  # ~6.7Hz update rate
        
        phase2_duration = time.time() - phase2_start
        print(f"   ✓ Sent {20 * len(queue_nodes) * len(queue_names)} queue updates in {phase2_duration:.2f}s")
        print(f"   📊 Queue metrics should appear in node_*.dat files")
        
        # Wait for aggregation
        await asyncio.sleep(1)
        
        # Phase 3: Violation grouping with extra_args
        print("\n📈 Phase 3: Violation grouping test (node:type:extra_args)")
        print("   Testing violation aggregation with context grouping...")
        
        violation_nodes = ["node_10", "node_11", "node_12"]
        # Test different grouping patterns
        grouping_patterns = [
            ("memory_exceeded", None),           # Basic grouping
            ("memory_exceeded", "gpu_0"),        # GPU-specific
            ("memory_exceeded", "gpu_1"),        # Different GPU
            ("compute_timeout", None),           # Different violation type
            ("compute_timeout", "batch_32"),     # With batch size context
            ("compute_timeout", "batch_64"),     # Different batch size
        ]
        
        phase3_start = time.time()
        violation_counts = {}
        
        # Send violations to test grouping and summary generation
        for i in range(30):  # 30 iterations to ensure summaries
            for node_id in violation_nodes:
                # Pick a random grouping pattern
                violation_type, extra_args = random.choice(grouping_patterns)
                
                # Track counts for validation
                key = f"{node_id}:{violation_type}:{extra_args or 'none'}"
                violation_counts[key] = violation_counts.get(key, 0) + 1
                
                # Send violation
                expected = 100.0 if "memory" in violation_type else 30.0
                actual = expected * 1.5 + random.uniform(-20, 20)
                
                telemetry.send_violation(
                    node_id=node_id,
                    violation_type=violation_type,
                    expected=expected,
                    actual=actual,
                    extra_args=extra_args
                )
            
            await asyncio.sleep(0.4)  # Slow rate to avoid rate limiting
        
        phase3_duration = time.time() - phase3_start
        total_violations = sum(violation_counts.values())
        unique_groups = len(violation_counts)
        
        print(f"   ✓ Sent {total_violations} violations in {phase3_duration:.2f}s")
        print(f"   📊 Created {unique_groups} unique groupings")
        print(f"   📊 Groups with 5+ violations should show summaries")
        
        # Show which groups should have summaries
        summary_groups = [k for k, v in violation_counts.items() if v >= 5]
        if summary_groups:
            print(f"   📋 Expected summaries for {len(summary_groups)} groups:")
            for group in summary_groups[:3]:  # Show first 3
                print(f"      - {group}: {violation_counts[group]} violations")
        
        # Phase 4: Custom metrics test
        print("\n📈 Phase 4: Custom metrics test")
        print("   Testing custom metric reporting...")
        
        custom_metrics = [
            ("model_accuracy", lambda: random.uniform(0.85, 0.95)),
            ("training_loss", lambda: random.uniform(0.1, 0.5)),
            ("validation_score", lambda: random.uniform(0.8, 0.9)),
            ("gradient_norm", lambda: random.uniform(0.5, 2.0)),
            ("learning_rate", lambda: 0.001 * (0.9 ** random.randint(0, 10)))
        ]
        
        phase4_start = time.time()
        for _ in range(15):
            for node_id in ["node_13", "node_14"]:  # Use fewer nodes for custom metrics
                for metric_name, value_gen in custom_metrics:
                    telemetry.send_custom(node_id, metric_name, value_gen())
            
            await asyncio.sleep(0.3)
        
        phase4_duration = time.time() - phase4_start
        print(f"   ✓ Sent {15 * 2 * len(custom_metrics)} custom metrics in {phase4_duration:.2f}s")
        print(f"   📊 Custom metrics should appear in node_*.dat files")
        
        # Phase 5: Mixed load to test concurrent aggregation
        print("\n📈 Phase 5: Mixed load test (all types concurrent)")
        print("   Testing concurrent aggregation of all telemetry types...")
        
        phase5_start = time.time()
        for _ in range(10):
            # Send a mix of everything
            # Metrics
            telemetry.send_metric("throughput", "node_10", random.uniform(90, 110))
            telemetry.send_metric("latency", "node_11", random.uniform(15, 25))
            
            # Queue depth
            telemetry.send_queue_depth("node_12", "mixed_queue", random.randint(0, 50))
            
            # Violation with extra_args
            telemetry.send_violation(
                "node_10", "mixed_violation", 100, 150,
                extra_args="phase5" if random.random() > 0.5 else None
            )
            
            # Custom metric
            telemetry.send_custom("node_13", "mixed_metric", random.uniform(0, 1))
            
            await asyncio.sleep(0.1)
        
        phase5_duration = time.time() - phase5_start
        print(f"   ✓ Sent mixed telemetry load in {phase5_duration:.2f}s")
        
        # Final summary
        total_duration = time.time() - phase1_start
        print("\n" + "=" * 60)
        print("✅ Aggregation test completed")
        print(f"⏱️ Total test duration: {total_duration:.1f}s")
        print("\n📊 Expected validation results:")
        print("   1. Agent should batch messages every 100ms")
        print("   2. Metrics in node_*.dat files (pipe-delimited)")
        print("   3. Violations in node_*_violations.log files")
        print("   4. First 5 violations as details, then SUMMARY entries")
        print("   5. Violations grouped by extra_args when present")
        print("   6. Queue depths recorded as 'queue_{name}' metrics")
        print("   7. Custom metrics stored with their names")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        telemetry.close()
    
    print("\n✅ Aggregation test runner completed successfully")


if __name__ == "__main__":
    # Run the aggregation test
    asyncio.run(run_aggregation_test())