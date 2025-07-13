#!/usr/bin/env python3
"""
DNNE Execution Trace - Instrument the actual DNNE code to trace execution timing
"""

import sys
import time
from pathlib import Path

# Add export directory to path
sys.path.insert(0, str(Path("/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO")))

# Import the actual DNNE components
from framework.base import QueueNode
import asyncio
from typing import Dict, Any

# Global trace collector
trace_events = []
start_time = None

def trace_event(node_id: str, event_type: str, details: str = ""):
    """Record a trace event with microsecond precision"""
    global start_time
    if start_time is None:
        start_time = time.perf_counter()
    
    elapsed_us = (time.perf_counter() - start_time) * 1_000_000
    trace_events.append({
        "time_us": elapsed_us,
        "node": node_id,
        "event": event_type,
        "details": details
    })

# Monkey-patch the QueueNode class to add tracing
original_run = QueueNode.run

async def traced_run(self):
    """Traced version of QueueNode.run"""
    self.running = True
    self.logger.info(f"Starting node {self.node_id}")
    
    try:
        iteration = 0
        while self.running and iteration < 5:  # Only trace 5 iterations
            iteration += 1
            
            # Trace: Start of iteration
            trace_event(self.node_id, "iteration_start", f"iter_{iteration}")
            
            # Gather all required inputs (with tracing)
            trace_event(self.node_id, "input_wait_start")
            input_wait_start = time.perf_counter()
            inputs = {}
            
            for input_name in self.required_inputs:
                trace_event(self.node_id, "queue_get_start", input_name)
                value = await self.input_queues[input_name].get()
                trace_event(self.node_id, "queue_get_end", input_name)
                inputs[input_name] = value
                
            input_wait_time = (time.perf_counter() - input_wait_start) * 1000
            trace_event(self.node_id, "input_wait_end", f"{input_wait_time:.2f}ms")
            
            # Execute compute (with tracing)
            trace_event(self.node_id, "compute_start")
            compute_start = time.perf_counter()
            outputs = await self.compute(**inputs)
            compute_time = (time.perf_counter() - compute_start) * 1000
            trace_event(self.node_id, "compute_end", f"{compute_time:.2f}ms")
            
            self.last_compute_time = compute_time / 1000
            self.compute_count += 1
            
            # Send outputs (with tracing)
            trace_event(self.node_id, "output_send_start")
            output_send_start = time.perf_counter()
            
            for output_name, value in outputs.items():
                trace_event(self.node_id, "queue_put_start", output_name)
                await self.send_output(output_name, value)
                trace_event(self.node_id, "queue_put_end", output_name)
                
            output_send_time = (time.perf_counter() - output_send_start) * 1000
            trace_event(self.node_id, "output_send_end", f"{output_send_time:.2f}ms")
            
            # Trace: End of iteration
            trace_event(self.node_id, "iteration_end", f"iter_{iteration}")
            
        # Stop after 5 iterations
        self.running = False
        
    except asyncio.CancelledError:
        self.logger.info(f"Node {self.node_id} cancelled")
        raise
    finally:
        self.running = False

# Apply the monkey patch
QueueNode.run = traced_run

async def run_traced_execution():
    """Run DNNE with tracing for a few iterations"""
    print("Running DNNE with execution tracing...")
    print("Will trace 5 iterations of each node")
    print("=" * 60)
    
    # Import and setup DNNE components
    from nodes.isaacgymstepnode_9 import IsaacGymStepNode_9
    from nodes.isaacgymenvnode_7 import IsaacGymEnvNode_7
    from nodes.ppotrainernode_6 import PPOTrainerNode_6
    from nodes.ornode_2 import ORNode_2
    from nodes.ppoagentnode_3 import PPOAgentNode_3
    from nodes.cartpoleactionnode_11 import CartpoleActionNode_11
    from framework.base import GraphRunner
    
    # Create nodes
    node_6 = PPOTrainerNode_6("6")
    node_9 = IsaacGymStepNode_9("9")
    node_2 = ORNode_2("2")
    node_3 = PPOAgentNode_3("3")
    node_11 = CartpoleActionNode_11("11")
    node_7 = IsaacGymEnvNode_7("7")
    
    # Create runner
    runner = GraphRunner()
    
    # Add nodes
    runner.add_node(node_6)
    runner.add_node(node_9)
    runner.add_node(node_2)
    runner.add_node(node_3)
    runner.add_node(node_11)
    runner.add_node(node_7)
    
    # Wire connections
    connections = [
        ("2", "output", "3", "observations"),
        ("3", "policy_output", "6", "policy_output"),
        ("3", "model", "6", "model"),
        ("7", "observations", "2", "input_a"),
        ("7", "sim_handle", "9", "sim_handle"),
        ("9", "observations", "6", "state"),
        ("9", "observations", "2", "input_b"),
        ("6", "training_complete", "9", "trigger"),
        ("9", "rewards", "6", "reward"),
        ("9", "done", "6", "done"),
        ("11", "action", "9", "actions"),
        ("3", "policy_output", "11", "policy"),
    ]
    runner.wire_nodes(connections)
    
    # Run for a short time
    try:
        await asyncio.wait_for(runner.run(), timeout=5.0)
    except asyncio.TimeoutError:
        pass
    
    # Analyze trace
    print("\nExecution Trace Analysis:")
    print("=" * 60)
    
    # Group events by iteration
    iterations = {}
    current_iters = {}
    
    for event in trace_events:
        node = event["node"]
        if event["event"] == "iteration_start":
            iter_num = event["details"]
            if iter_num not in iterations:
                iterations[iter_num] = {}
            iterations[iter_num][node] = {"start": event["time_us"], "events": []}
            current_iters[node] = iter_num
        elif node in current_iters:
            iter_num = current_iters[node]
            if iter_num in iterations and node in iterations[iter_num]:
                iterations[iter_num][node]["events"].append(event)
                if event["event"] == "iteration_end":
                    iterations[iter_num][node]["end"] = event["time_us"]
    
    # Print timing for first complete iteration
    for iter_num in ["iter_1", "iter_2"]:
        if iter_num in iterations:
            print(f"\n{iter_num} Timing Breakdown:")
            print("-" * 40)
            
            iter_data = iterations[iter_num]
            
            # Find the overall iteration time
            min_start = min(data["start"] for data in iter_data.values() if "start" in data)
            max_end = max(data.get("end", 0) for data in iter_data.values())
            total_time = (max_end - min_start) / 1000  # Convert to ms
            
            print(f"Total iteration time: {total_time:.2f}ms")
            print("\nPer-node breakdown:")
            
            for node_id in sorted(iter_data.keys()):
                node_data = iter_data[node_id]
                if "start" in node_data and "end" in node_data:
                    node_time = (node_data["end"] - node_data["start"]) / 1000
                    print(f"\nNode {node_id}: {node_time:.2f}ms total")
                    
                    # Extract key timings
                    for event in node_data["events"]:
                        if "end" in event["event"] and "ms" in event["details"]:
                            print(f"  - {event['event'].replace('_end', '')}: {event['details']}")
    
    # Print raw trace for detailed analysis
    print("\n\nDetailed Event Sequence (first 50 events):")
    print("-" * 60)
    print(f"{'Time (ms)':>10} {'Node':>6} {'Event':>20} {'Details':>20}")
    print("-" * 60)
    
    for event in trace_events[:50]:
        time_ms = event["time_us"] / 1000
        print(f"{time_ms:10.2f} {event['node']:>6} {event['event']:>20} {event['details']:>20}")

if __name__ == "__main__":
    # Need to run in the conda environment
    import subprocess
    import os
    
    if os.environ.get("CONDA_DEFAULT_ENV") != "DNNE_PY38":
        print("Activating conda environment...")
        cmd = [
            "bash", "-c",
            "source /home/asantanna/miniconda/bin/activate DNNE_PY38 && python " + __file__
        ]
        subprocess.run(cmd)
    else:
        # We're in the right environment
        import isaacgym  # Import first to avoid issues
        asyncio.run(run_traced_execution())