#!/usr/bin/env python3
"""
Simple test to verify telemetry client sends data correctly.
Run this with the agent client running to see telemetry flow.
"""

import time
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the telemetry module directly
from export_system.templates.framework.telemetry import TelemetryClient

def main():
    print("🧪 Testing Telemetry Client")
    print("=" * 60)
    
    # Create telemetry client
    client = TelemetryClient(enabled=True, host="localhost", port=9999, violation_rate_limit=10)
    
    print("📡 Telemetry client created")
    print(f"   Target: localhost:9999")
    print(f"   Enabled: {client.enabled}")
    
    if not client.enabled:
        print("❌ Telemetry client failed to initialize")
        return 1
    
    # Test different telemetry types
    node_id = "test_10"
    
    print("\n📊 Sending test telemetry...")
    
    # 1. Send some metrics
    for i in range(5):
        throughput = 100 + i * 10
        client.report_throughput(node_id, throughput)
        print(f"   Sent throughput: {throughput} items/sec")
        time.sleep(0.1)
    
    # 2. Send queue depths
    for i in range(3):
        client.report_queue_depth(node_id, "input_queue", 5 + i)
        client.report_queue_depth(node_id, "output_queue", 10 + i)
        print(f"   Sent queue depths: input={5+i}, output={10+i}")
        time.sleep(0.1)
    
    # 3. Send violations (should be rate limited to 10/sec)
    print("\n🚨 Testing violation rate limiting (sending 15 violations)...")
    violations_sent = 0
    for i in range(15):
        client.report_violation(
            node_id, 
            "frequency_below_minimum",
            expected=30.0,
            actual=25.0 + i * 0.1
        )
        violations_sent += 1
        print(f"   Violation {violations_sent}: actual={25.0 + i * 0.1:.1f}", end="")
        if violations_sent > 10:
            print(" (may be dropped by rate limiter)")
        else:
            print()
        time.sleep(0.05)  # Send faster than 10/sec to test rate limiting
    
    # 4. Test violations with extra_args
    print("\n🏷️ Testing violations with extra context...")
    client.report_violation(
        node_id,
        "memory_exceeded",
        expected=8192,
        actual=9500,
        extra_args="gpu_0"
    )
    print("   Sent memory violation for gpu_0")
    
    client.report_violation(
        node_id,
        "memory_exceeded", 
        expected=8192,
        actual=8800,
        extra_args="gpu_1"
    )
    print("   Sent memory violation for gpu_1")
    
    # 5. Send custom metrics
    print("\n📈 Sending custom metrics...")
    for i in range(3):
        client.report_custom(node_id, "custom_metric", 42.0 + i)
        print(f"   Sent custom_metric: {42.0 + i}")
        time.sleep(0.1)
    
    # Close client
    client.close()
    print("\n✅ Telemetry test complete!")
    print("\nCheck the agent client logs to see if telemetry was received.")
    print("If agent is running, check remote_clients/*/telemetry/ for output files.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())