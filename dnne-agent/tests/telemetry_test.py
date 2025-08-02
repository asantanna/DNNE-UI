#!/usr/bin/env python3
"""
Test workflow that sends telemetry data.
Simulates a node sending various metrics via UDP.
"""

import socket
import time
import json
import random
import sys

def send_telemetry(sock, metric_type, node_id, value, target=('localhost', 9999)):
    """Send a telemetry packet via UDP"""
    # Simple format: type|node_id|value|timestamp
    packet = f"{metric_type}|{node_id}|{value}|{time.time()}"
    try:
        sock.sendto(packet.encode(), target)
        print(f"Sent: {metric_type}={value} for {node_id}")
    except Exception as e:
        print(f"Failed to send telemetry: {e}")

def send_json_telemetry(sock, data, target=('localhost', 9999)):
    """Send JSON telemetry packet"""
    try:
        packet = json.dumps(data)
        sock.sendto(packet.encode(), target)
        print(f"Sent JSON: {data['type']}")
    except Exception as e:
        print(f"Failed to send JSON telemetry: {e}")

def main():
    print("Telemetry test workflow started")
    
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    
    # Node ID for testing
    node_id = "test_node_42"
    
    print("Sending various telemetry metrics...")
    
    # Send different types of metrics
    for i in range(10):
        # Throughput (varying)
        throughput = random.uniform(50, 150)
        send_telemetry(sock, "throughput", node_id, throughput)
        
        # Latency (in ms)
        latency = random.uniform(10, 50)
        send_telemetry(sock, "latency", node_id, latency)
        
        # Queue depth
        if i % 3 == 0:
            send_json_telemetry(sock, {
                "type": "queue",
                "node_id": node_id,
                "queue": "input_data",
                "depth": random.randint(0, 100),
                "timestamp": time.time()
            })
        
        # Occasional violation
        if i % 5 == 0:
            send_json_telemetry(sock, {
                "type": "violation",
                "node_id": node_id,
                "violation": "frequency_below_minimum",
                "expected": 100.0,
                "actual": 85.5,
                "guaranteed": False,
                "timestamp": time.time()
            })
        
        time.sleep(0.5)
    
    # Send a burst of metrics
    print("\nSending burst of 100 metrics...")
    for i in range(100):
        throughput = random.uniform(100, 200)
        send_telemetry(sock, "throughput", f"burst_node_{i%5}", throughput)
        time.sleep(0.01)  # 100Hz
    
    sock.close()
    print("\nTelemetry test completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())