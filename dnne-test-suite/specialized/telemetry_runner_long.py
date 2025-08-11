#!/usr/bin/env python3
"""
Long-running telemetry test - deployed and executed by test_telemetry.py orchestrator.
This tests the 10-second aggregation intervals for SUMMARY violations.
Runs for 35 seconds to generate at least 3 SUMMARY entries.
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


async def run_long_telemetry_test():
    """
    Long-running telemetry test - 35 seconds to test aggregation intervals.
    Sends continuous violations to trigger multiple SUMMARY entries.
    """
    print("🚀 Long-Running Telemetry Test Started")
    print("=" * 60)
    print("📊 Test will run for 35 seconds to validate 10-second aggregation")
    
    # Create telemetry client
    telemetry = TelemetryClient()
    
    try:
        start_time = time.time()
        violation_count = 0
        last_summary_time = start_time
        summary_count = 0
        
        # Run for 35 seconds
        while time.time() - start_time < 35:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check if we expect a summary (every 10 seconds after first 5 violations)
            if current_time - last_summary_time >= 10 and violation_count > 5:
                summary_count += 1
                last_summary_time = current_time
                print(f"\n⏱️ [{elapsed:.1f}s] Expected SUMMARY #{summary_count} at ~{elapsed:.0f}s")
            
            # Send violations continuously (3 per second)
            for node_id in ["node_20", "node_21", "node_22"]:
                # Different violation types for each node
                if node_id == "node_20":
                    violation_type = "frequency_below_minimum"
                    expected = 30.0
                    actual = 25.0 + random.uniform(-2, 2)
                elif node_id == "node_21":
                    violation_type = "latency_exceeded"
                    expected = 10.0
                    actual = 15.0 + random.uniform(-3, 3)
                else:
                    violation_type = "queue_overflow"
                    expected = 100
                    actual = 120 + random.randint(-10, 10)
                
                telemetry.send_violation(
                    node_id=node_id,
                    violation_type=violation_type,
                    expected=expected,
                    actual=actual
                )
                violation_count += 1
                
                # Also send some metrics
                telemetry.send_metric("throughput", node_id, random.uniform(80, 120))
                
                await asyncio.sleep(0.1)  # 10Hz per node
            
            # Print progress every 5 seconds
            if int(elapsed) % 5 == 0 and int(elapsed) != int(elapsed - 0.3):
                print(f"📈 [{elapsed:.0f}s] Sent {violation_count} violations so far...")
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n" + "=" * 60)
        print(f"✅ Long-running test completed in {total_time:.1f} seconds")
        print(f"📊 Statistics:")
        print(f"   - Total violations sent: {violation_count}")
        print(f"   - Expected SUMMARY entries: {summary_count} (at 10s, 20s, 30s)")
        print(f"   - Violation rate: {violation_count/total_time:.1f} per second")
        
        # Send a final burst to ensure last summary
        print(f"\n💥 Sending final burst to ensure last SUMMARY...")
        for i in range(20):
            telemetry.send_violation(
                node_id="node_20",
                violation_type="final_burst",
                expected=100,
                actual=200
            )
            await asyncio.sleep(0.01)
        
        print("✅ Final burst sent")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        telemetry.close()
        print("\n🏁 Long-running telemetry test finished")


if __name__ == "__main__":
    asyncio.run(run_long_telemetry_test())