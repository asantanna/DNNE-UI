#!/usr/bin/env python3
"""
DNNE Warmup Profiler - Profile DNNE performance after proper warmup
"""

import sys
import time
import asyncio
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

# Add export directory to path
sys.path.insert(0, str(Path("/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO")))

class WarmupProfiler:
    """Profiler that only collects data after warmup period"""
    
    def __init__(self, warmup_iterations: int = 5):
        self.warmup_iterations = warmup_iterations
        self.iteration_counts = defaultdict(int)
        self.timings = defaultdict(lambda: defaultdict(list))
        self.is_warmed_up = defaultdict(bool)
        self.enabled = True
        
    def record_timing(self, node_id: str, phase: str, duration_ms: float):
        """Record timing, but only after warmup"""
        self.iteration_counts[node_id] += 1
        
        # Check if this node has warmed up
        if self.iteration_counts[node_id] > self.warmup_iterations:
            if not self.is_warmed_up[node_id]:
                self.is_warmed_up[node_id] = True
                print(f"Node {node_id} warmed up after {self.warmup_iterations} iterations")
            
            # Only record post-warmup timings
            self.timings[node_id][phase].append(duration_ms)
    
    def get_stats(self) -> Dict:
        """Get statistics for warmed-up measurements"""
        stats = {}
        
        for node_id, phases in self.timings.items():
            stats[node_id] = {}
            
            for phase, times in phases.items():
                if times:
                    stats[node_id][phase] = {
                        'count': len(times),
                        'avg': sum(times) / len(times),
                        'min': min(times),
                        'max': max(times),
                        'total': sum(times)
                    }
        
        return stats

# Global profiler instance
profiler = WarmupProfiler(warmup_iterations=5)

# Monkey-patch imports
from framework.base import QueueNode
original_run = QueueNode.run

async def profiled_run(self):
    """Run with warmup-aware profiling"""
    self.running = True
    self.logger.info(f"Starting node {self.node_id}")
    
    node_iteration = 0
    
    try:
        while self.running:
            node_iteration += 1
            
            # Input gathering phase
            input_start = time.perf_counter()
            inputs = {}
            for input_name in self.required_inputs:
                value = await self.input_queues[input_name].get()
                inputs[input_name] = value
            input_time_ms = (time.perf_counter() - input_start) * 1000
            
            # Compute phase
            compute_start = time.perf_counter()
            outputs = await self.compute(**inputs)
            compute_time_ms = (time.perf_counter() - compute_start) * 1000
            
            # Output sending phase
            output_start = time.perf_counter()
            for output_name, value in outputs.items():
                await self.send_output(output_name, value)
            output_time_ms = (time.perf_counter() - output_start) * 1000
            
            # Total time
            total_time_ms = input_time_ms + compute_time_ms + output_time_ms
            
            # Record timings (will be ignored during warmup)
            if profiler.enabled:
                profiler.record_timing(self.node_id, 'input_wait', input_time_ms)
                profiler.record_timing(self.node_id, 'compute', compute_time_ms)
                profiler.record_timing(self.node_id, 'output_send', output_time_ms)
                profiler.record_timing(self.node_id, 'total', total_time_ms)
            
            # Update node stats
            self.compute_count += 1
            self.last_compute_time = compute_time_ms / 1000
            
    except asyncio.CancelledError:
        self.logger.info(f"Node {self.node_id} cancelled")
        raise
    finally:
        self.running = False

# Apply the monkey patch
QueueNode.run = profiled_run

async def run_warmup_profiling(duration_seconds: int = 30):
    """Run DNNE with warmup profiling"""
    print("DNNE Warmup Profiler")
    print("=" * 60)
    print(f"Warmup iterations: {profiler.warmup_iterations}")
    print(f"Profile duration: {duration_seconds} seconds")
    print("=" * 60)
    
    # Import Isaac Gym first
    import isaacgym
    
    # Import nodes
    from nodes.isaacgymstepnode_9 import IsaacGymStepNode_9
    from nodes.isaacgymenvnode_7 import IsaacGymEnvNode_7
    from nodes.ppotrainernode_6 import PPOTrainerNode_6
    from nodes.ornode_2 import ORNode_2
    from nodes.ppoagentnode_3 import PPOAgentNode_3
    from nodes.cartpoleactionnode_11 import CartpoleActionNode_11
    from framework.base import GraphRunner
    
    # Create nodes
    nodes = {
        "6": PPOTrainerNode_6("6"),
        "9": IsaacGymStepNode_9("9"),
        "2": ORNode_2("2"),
        "3": PPOAgentNode_3("3"),
        "11": CartpoleActionNode_11("11"),
        "7": IsaacGymEnvNode_7("7")
    }
    
    # Create runner
    runner = GraphRunner()
    
    # Add nodes
    for node in nodes.values():
        runner.add_node(node)
    
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
    
    # Run with timeout
    print("\nStarting execution...")
    start_time = time.time()
    
    try:
        await asyncio.wait_for(runner.run(), timeout=duration_seconds)
    except asyncio.TimeoutError:
        pass
    
    actual_duration = time.time() - start_time
    print(f"\nExecution completed. Duration: {actual_duration:.1f} seconds")
    
    # Get and display statistics
    stats = profiler.get_stats()
    
    print("\n" + "=" * 80)
    print("PERFORMANCE STATISTICS (Post-Warmup Only)")
    print("=" * 80)
    
    # Calculate total throughput
    total_iterations = {}
    for node_id, phases in stats.items():
        if 'total' in phases:
            total_iterations[node_id] = phases['total']['count']
    
    if total_iterations:
        # Use the node with most iterations as reference
        max_iterations = max(total_iterations.values())
        throughput_fps = max_iterations / actual_duration
        ms_per_iteration = 1000 / throughput_fps
        
        print(f"\nSystem Throughput:")
        print(f"  Total iterations (post-warmup): {max_iterations}")
        print(f"  Throughput: {throughput_fps:.1f} iterations/second")
        print(f"  Time per iteration: {ms_per_iteration:.1f} ms")
    
    # Node-level statistics
    print("\nPer-Node Timing Breakdown (milliseconds):")
    print("-" * 80)
    print(f"{'Node':>6} {'Phase':>12} {'Count':>8} {'Avg':>10} {'Min':>10} {'Max':>10} {'Total':>12}")
    print("-" * 80)
    
    for node_id in sorted(stats.keys()):
        node_stats = stats[node_id]
        first_row = True
        
        for phase in ['input_wait', 'compute', 'output_send', 'total']:
            if phase in node_stats:
                s = node_stats[phase]
                if first_row:
                    print(f"{node_id:>6} {phase:>12} {s['count']:>8} "
                          f"{s['avg']:>10.2f} {s['min']:>10.2f} {s['max']:>10.2f} {s['total']:>12.1f}")
                    first_row = False
                else:
                    print(f"{'':>6} {phase:>12} {s['count']:>8} "
                          f"{s['avg']:>10.2f} {s['min']:>10.2f} {s['max']:>10.2f} {s['total']:>12.1f}")
        print()
    
    # Identify bottlenecks
    print("\nBottleneck Analysis:")
    print("-" * 40)
    
    # Find the slowest total time
    slowest_node = None
    slowest_time = 0
    
    for node_id, phases in stats.items():
        if 'total' in phases:
            avg_total = phases['total']['avg']
            if avg_total > slowest_time:
                slowest_time = avg_total
                slowest_node = node_id
    
    if slowest_node:
        print(f"Slowest node: {slowest_node} ({slowest_time:.1f} ms average)")
        
        # Break down the slowest node
        node_stats = stats[slowest_node]
        if 'compute' in node_stats and 'input_wait' in node_stats:
            compute_pct = (node_stats['compute']['avg'] / slowest_time) * 100
            wait_pct = (node_stats['input_wait']['avg'] / slowest_time) * 100
            
            print(f"  - Compute: {node_stats['compute']['avg']:.1f} ms ({compute_pct:.1f}%)")
            print(f"  - Input wait: {node_stats['input_wait']['avg']:.1f} ms ({wait_pct:.1f}%)")
            
            if compute_pct > 80:
                print(f"  → Node {slowest_node} is COMPUTE-BOUND")
            elif wait_pct > 80:
                print(f"  → Node {slowest_node} is WAITING for inputs")
            else:
                print(f"  → Node {slowest_node} has MIXED bottlenecks")

async def run_comparison_test():
    """Run comparison between warmed and non-warmed stats"""
    print("\nBonus: Comparing with and without warmup...")
    print("-" * 60)
    
    # This would show the difference between including initialization
    # vs excluding it from measurements
    pass

if __name__ == "__main__":
    # Run in the DNNE environment
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
        asyncio.run(run_warmup_profiling(duration_seconds=20))