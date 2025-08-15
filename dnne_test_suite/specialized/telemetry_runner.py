#!/usr/bin/env python3
"""
Telemetry runner - deployed and executed by test_telemetry.py orchestrator.
This simulates a real workflow sending telemetry via UDP to the client agent.
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
            
        self._send_json(packet)
    
    def send_queue_depth(self, node_id: str, queue_name: str, depth: int):
        """Send queue depth metric."""
        packet = {
            "type": "queue",
            "node_id": node_id,
            "queue": queue_name,
            "depth": depth,
            "timestamp": time.time()
        }
        self._send_json(packet)
    
    def _send_json(self, data: Dict[str, Any]):
        """Send JSON packet."""
        try:
            json_data = json.dumps(data)
            self._send_raw(json_data.encode())
        except:
            pass  # Fire-and-forget
    
    def _send_raw(self, data: bytes):
        """
        Send raw UDP packet - fire and forget.
        """
        try:
            self.socket.sendto(data, self.target)
        except:
            pass  # Fire-and-forget
    
    def close(self):
        """Close the UDP socket."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None


async def run_telemetry_test():
    """
    Main telemetry test - simulates a real workflow sending telemetry.
    """
    print("🚀 Telemetry Runner Started")
    print("=" * 60)
    
    # Create telemetry client
    telemetry = TelemetryClient()
    
    try:
        # Phase 1: Basic telemetry from multiple nodes
        print("📊 Sending basic telemetry...")
        for node_num in range(3):
            node_id = f"node_{10 + node_num}"
            
            # Throughput metrics
            for i in range(5):
                throughput = random.uniform(90, 110) + node_num * 10
                telemetry.send_metric("throughput", node_id, throughput)
                await asyncio.sleep(0.01)
            
            # Latency metrics
            for i in range(3):
                latency = random.uniform(10, 50)
                telemetry.send_metric("latency", node_id, latency)
                await asyncio.sleep(0.01)
            
            # Queue depth
            telemetry.send_queue_depth(node_id, "input_queue", random.randint(5, 50))
            
            print(f"   ✓ Node {node_id}: Sent telemetry")
            await asyncio.sleep(0.1)
        
        # Phase 2: Violations with rate limiting test
        print("🚨 Sending violations (testing rate limiting)...")
        node_id = "node_10"
        
        # Send 15 violations rapidly
        for i in range(15):
            telemetry.send_violation(
                node_id=node_id,
                violation_type="frequency_below_minimum",
                expected=30.0,
                actual=25.0 + i * 0.1
            )
            await asyncio.sleep(0.01)  # Small delay between violations
        
        print(f"   ✓ Sent 15 violations for rate limiting test")
        
        # Phase 3: Violations with grouping
        print("🏷️ Sending grouped violations...")
        
        # Memory violations for different GPUs
        for gpu_id in ["gpu_0", "gpu_1", "gpu_2"]:
            for i in range(3):
                telemetry.send_violation(
                    node_id="node_11",
                    violation_type="memory_exceeded",
                    expected=8192,
                    actual=8500 + random.randint(0, 1000),
                    extra_args=gpu_id
                )
                await asyncio.sleep(0.01)
        
        print(f"   ✓ Sent memory violations for 3 GPUs")
        
        # Phase 4: Burst test
        print("💥 Running burst test (200 packets at 100Hz)...")
        start_time = time.time()
        packets_sent = 0
        
        for i in range(100):
            node_id = f"burst_node_{i % 5}"  # Distribute across 5 nodes
            
            # Send throughput and latency metrics
            telemetry.send_metric("throughput", node_id, random.uniform(100, 200))
            telemetry.send_metric("latency", node_id, random.uniform(5, 25))
            packets_sent += 2
            
            await asyncio.sleep(0.01)  # 100Hz
        
        elapsed = time.time() - start_time
        print(f"   ✓ Sent {packets_sent} packets in {elapsed:.2f}s ({packets_sent/elapsed:.1f} Hz)")
        
        # Phase 5: Custom metrics
        print("📈 Sending custom metrics...")
        for i in range(5):
            telemetry.send_metric("gpu_utilization", "node_12", random.uniform(0, 100))
            telemetry.send_metric("memory_usage", "node_12", random.uniform(0, 8192))
            await asyncio.sleep(0.05)
        
        print(f"   ✓ Sent custom metrics")
        
        # Phase 6: Continuous operation for a bit
        print("⏳ Running continuous telemetry for 5 seconds...")
        end_time = time.time() + 5
        cycle = 0
        
        while time.time() < end_time:
            # Send some periodic telemetry
            for node_num in range(3):
                node_id = f"node_{10 + node_num}"
                telemetry.send_metric("throughput", node_id, random.uniform(90, 110) + node_num * 10)
            
            cycle += 1
            await asyncio.sleep(0.5)
        
        print(f"   ✓ Sent {cycle} cycles of continuous telemetry")
        
        print("\n" + "=" * 60)
        print("✅ Telemetry Runner Complete")
        print(f"   Total runtime: {time.time() - start_time:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Error in telemetry runner: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        telemetry.close()
    
    return 0


def main():
    """Main entry point."""
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version}")
    print()
    
    result = asyncio.run(run_telemetry_test())
    sys.exit(result)


if __name__ == "__main__":
    main()