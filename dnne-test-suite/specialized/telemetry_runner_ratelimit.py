#!/usr/bin/env python3
"""
Rate limiting telemetry test - deployed and executed by test_telemetry.py orchestrator.
This tests that violations are rate-limited to 10/second per node.
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


async def run_ratelimit_test():
    """
    Rate limiting test - sends 100 violations in 1 second to test rate limiting.
    Should only see ~10 violations recorded due to rate limiting.
    """
    print("🚀 Rate Limiting Telemetry Test Started")
    print("=" * 60)
    print("📊 Testing violation rate limiting (max 10/sec per node)")
    
    # Create telemetry client
    telemetry = TelemetryClient()
    
    try:
        # Phase 1: Burst test on single node
        print("\n📈 Phase 1: Burst test on single node (node_30)")
        print("   Sending 100 violations in 1 second...")
        
        burst_start = time.time()
        for i in range(100):
            telemetry.send_violation(
                node_id="node_30",
                violation_type="burst_test",
                expected=100,
                actual=200 + i
            )
            await asyncio.sleep(0.01)  # 100Hz
        
        burst_duration = time.time() - burst_start
        print(f"   ✓ Sent 100 violations in {burst_duration:.2f}s")
        print(f"   ⏱️ Rate: {100/burst_duration:.1f} violations/sec")
        print(f"   📊 Expected to record: ~10 (rate limited)")
        
        # Wait for aggregation
        await asyncio.sleep(2)
        
        # Phase 2: Multiple nodes with independent rate limits
        print("\n📈 Phase 2: Testing independent rate limits per node")
        print("   Sending 50 violations/sec to 3 different nodes...")
        
        multi_start = time.time()
        for i in range(50):
            # Send to three different nodes
            for node_id in ["node_31", "node_32", "node_33"]:
                telemetry.send_violation(
                    node_id=node_id,
                    violation_type="multi_node_test",
                    expected=50,
                    actual=75 + random.uniform(-10, 10)
                )
            await asyncio.sleep(0.02)  # 50Hz for each node
        
        multi_duration = time.time() - multi_start
        print(f"   ✓ Sent 150 violations total in {multi_duration:.2f}s")
        print(f"   📊 Expected per node: ~10 each (independently rate limited)")
        
        # Wait for aggregation
        await asyncio.sleep(2)
        
        # Phase 3: Slow violations (should all be recorded)
        print("\n📈 Phase 3: Slow violations (5/sec, below limit)")
        print("   Sending 25 violations at 5/sec...")
        
        slow_start = time.time()
        for i in range(25):
            telemetry.send_violation(
                node_id="node_34",
                violation_type="slow_test",
                expected=10,
                actual=15 + random.uniform(-2, 2)
            )
            await asyncio.sleep(0.2)  # 5Hz
        
        slow_duration = time.time() - slow_start
        print(f"   ✓ Sent 25 violations in {slow_duration:.2f}s")
        print(f"   📊 Expected to record: 25 (all should pass)")
        
        # Send some normal metrics too
        print("\n📊 Sending normal metrics (not rate limited)...")
        for i in range(20):
            for node_id in ["node_30", "node_31", "node_32", "node_33", "node_34"]:
                telemetry.send_metric("throughput", node_id, random.uniform(80, 120))
            await asyncio.sleep(0.1)
        
        print("   ✓ Sent 100 metric updates")
        
        # Final summary
        print("\n" + "=" * 60)
        print("✅ Rate limiting test completed")
        print("📊 Expected results:")
        print("   - node_30: ~10 violations (100 sent, rate limited)")
        print("   - node_31/32/33: ~10 each (50 sent each, rate limited)")
        print("   - node_34: 25 violations (all below rate limit)")
        print("   - All nodes: Metrics should be recorded (not limited)")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        telemetry.close()
        print("\n🏁 Rate limiting test finished")


if __name__ == "__main__":
    asyncio.run(run_ratelimit_test())