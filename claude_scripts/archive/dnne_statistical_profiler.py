#!/usr/bin/env python3
"""
DNNE Statistical Profiler - Comprehensive performance analysis with proper statistics
"""

import sys
import time
import asyncio
import statistics
import cProfile
import pstats
import io
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

# Add export directory to path
sys.path.insert(0, str(Path("/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO")))

class StatisticalProfiler:
    """Profiler with comprehensive statistical analysis"""
    
    def __init__(self, warmup_iterations: int = 10, outlier_percentile: int = 95):
        self.warmup_iterations = warmup_iterations
        self.outlier_percentile = outlier_percentile
        self.iteration_counts = defaultdict(int)
        self.raw_timings = defaultdict(lambda: defaultdict(list))
        self.is_warmed_up = defaultdict(bool)
        
    def record_timing(self, node_id: str, phase: str, duration_ms: float):
        """Record timing, but only after warmup"""
        self.iteration_counts[node_id] += 1
        
        if self.iteration_counts[node_id] > self.warmup_iterations:
            if not self.is_warmed_up[node_id]:
                self.is_warmed_up[node_id] = True
                print(f"✓ Node {node_id} warmed up after {self.warmup_iterations} iterations")
            
            self.raw_timings[node_id][phase].append(duration_ms)
    
    def get_statistics(self, timings: List[float]) -> Dict:
        """Calculate comprehensive statistics for a timing series"""
        if not timings:
            return {}
        
        # Remove outliers for some calculations
        sorted_timings = sorted(timings)
        percentile_cutoff = np.percentile(sorted_timings, self.outlier_percentile)
        filtered_timings = [t for t in timings if t <= percentile_cutoff]
        
        stats = {
            'count': len(timings),
            'mean': statistics.mean(timings),
            'median': statistics.median(timings),
            'stdev': statistics.stdev(timings) if len(timings) > 1 else 0,
            'min': min(timings),
            'max': max(timings),
            'p50': np.percentile(timings, 50),
            'p90': np.percentile(timings, 90),
            'p95': np.percentile(timings, 95),
            'p99': np.percentile(timings, 99),
            'mean_filtered': statistics.mean(filtered_timings) if filtered_timings else 0,
            'outliers': len(timings) - len(filtered_timings),
            'cv': statistics.stdev(timings) / statistics.mean(timings) if len(timings) > 1 and statistics.mean(timings) > 0 else 0
        }
        
        return stats
    
    def print_report(self):
        """Print comprehensive statistical report"""
        print("\n" + "=" * 100)
        print("STATISTICAL PERFORMANCE REPORT")
        print("=" * 100)
        
        # Per-node statistics
        for node_id in sorted(self.raw_timings.keys()):
            print(f"\nNode {node_id}:")
            print("-" * 80)
            
            # Header
            print(f"{'Phase':>12} {'Count':>6} {'Mean':>8} {'Median':>8} {'StdDev':>8} {'Min':>8} {'Max':>8} {'P90':>8} {'P95':>8} {'P99':>8} {'CV':>6}")
            print("-" * 80)
            
            for phase in ['input_wait', 'compute', 'output_send', 'total']:
                if phase in self.raw_timings[node_id]:
                    timings = self.raw_timings[node_id][phase]
                    stats = self.get_statistics(timings)
                    
                    print(f"{phase:>12} {stats['count']:>6} {stats['mean']:>8.2f} {stats['median']:>8.2f} "
                          f"{stats['stdev']:>8.2f} {stats['min']:>8.2f} {stats['max']:>8.2f} "
                          f"{stats['p90']:>8.2f} {stats['p95']:>8.2f} {stats['p99']:>8.2f} {stats['cv']:>6.2f}")
                    
                    # Show outlier information if significant
                    if stats['outliers'] > 0:
                        print(f"{'':>12} → {stats['outliers']} outliers removed: mean_filtered = {stats['mean_filtered']:.2f}ms")
        
        # System-wide analysis
        print("\n\nSYSTEM-WIDE ANALYSIS")
        print("=" * 100)
        
        # Find the bottleneck phase
        bottleneck_node = None
        bottleneck_phase = None
        max_median = 0
        
        for node_id, phases in self.raw_timings.items():
            for phase, timings in phases.items():
                if phase == 'total' and timings:
                    median = statistics.median(timings)
                    if median > max_median:
                        max_median = median
                        bottleneck_node = node_id
                        bottleneck_phase = phase
        
        if bottleneck_node:
            print(f"\nBottleneck: Node {bottleneck_node} (median total time: {max_median:.2f}ms)")
            
            # Analyze bottleneck node in detail
            node_phases = self.raw_timings[bottleneck_node]
            if 'compute' in node_phases and 'input_wait' in node_phases:
                compute_median = statistics.median(node_phases['compute'])
                wait_median = statistics.median(node_phases['input_wait'])
                total_median = compute_median + wait_median
                
                compute_pct = (compute_median / total_median) * 100 if total_median > 0 else 0
                wait_pct = (wait_median / total_median) * 100 if total_median > 0 else 0
                
                print(f"  - Compute: {compute_median:.2f}ms ({compute_pct:.1f}%)")
                print(f"  - Wait: {wait_median:.2f}ms ({wait_pct:.1f}%)")

# Global profiler instance
profiler = StatisticalProfiler(warmup_iterations=10)

async def profile_with_cprofile():
    """Run DNNE with cProfile for detailed analysis"""
    print("\nRunning with cProfile for detailed analysis...")
    print("=" * 60)
    
    import isaacgym
    from nodes.ppoagentnode_3 import PPOAgentNode_3
    import torch
    
    agent = PPOAgentNode_3("3")
    observations = torch.randn(512, 4, device="cuda")
    
    # Warmup
    for _ in range(5):
        await agent.compute(observations)
    
    # Profile 50 calls
    pr = cProfile.Profile()
    pr.enable()
    
    for _ in range(50):
        await agent.compute(observations)
    
    pr.disable()
    
    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    
    print("\ncProfile Results (Top 30 by cumulative time):")
    print(s.getvalue())
    
    # Also save to file
    pr.dump_stats('ppo_agent_profile.stats')
    print("\nDetailed profile saved to: ppo_agent_profile.stats")
    print("View with: python -m pstats ppo_agent_profile.stats")

async def profile_with_line_profiler():
    """Demonstrate line_profiler usage"""
    print("\n\nLine Profiler Instructions:")
    print("=" * 60)
    print("For line-by-line profiling, install line_profiler:")
    print("  pip install line_profiler")
    print("\nThen add @profile decorator to methods and run:")
    print("  kernprof -l -v your_script.py")
    print("\nThis gives line-by-line timing within decorated functions.")

async def run_statistical_profiling(duration_seconds: int = 30):
    """Run DNNE with statistical profiling"""
    print("DNNE Statistical Profiler")
    print("=" * 100)
    print(f"Configuration:")
    print(f"  - Warmup iterations: {profiler.warmup_iterations}")
    print(f"  - Test duration: {duration_seconds} seconds")
    print(f"  - Outlier percentile: {profiler.outlier_percentile}%")
    print("=" * 100)
    
    # Monkey-patch the framework
    from framework.base import QueueNode
    original_run = QueueNode.run
    
    async def profiled_run(self):
        """Run with statistical profiling"""
        self.running = True
        self.logger.info(f"Starting node {self.node_id}")
        
        try:
            while self.running:
                # Timing points
                iteration_start = time.perf_counter()
                
                # Input phase
                input_start = time.perf_counter()
                inputs = {}
                for input_name in self.required_inputs:
                    value = await self.input_queues[input_name].get()
                    inputs[input_name] = value
                input_time = (time.perf_counter() - input_start) * 1000
                
                # Compute phase
                compute_start = time.perf_counter()
                outputs = await self.compute(**inputs)
                compute_time = (time.perf_counter() - compute_start) * 1000
                
                # Output phase
                output_start = time.perf_counter()
                for output_name, value in outputs.items():
                    await self.send_output(output_name, value)
                output_time = (time.perf_counter() - output_start) * 1000
                
                # Total time
                total_time = (time.perf_counter() - iteration_start) * 1000
                
                # Record all timings
                profiler.record_timing(self.node_id, 'input_wait', input_time)
                profiler.record_timing(self.node_id, 'compute', compute_time)
                profiler.record_timing(self.node_id, 'output_send', output_time)
                profiler.record_timing(self.node_id, 'total', total_time)
                
                self.compute_count += 1
                self.last_compute_time = compute_time / 1000
                
        except asyncio.CancelledError:
            self.logger.info(f"Node {self.node_id} cancelled")
            raise
        finally:
            self.running = False
    
    QueueNode.run = profiled_run
    
    # Import and run DNNE
    import isaacgym
    
    from nodes.isaacgymstepnode_9 import IsaacGymStepNode_9
    from nodes.isaacgymenvnode_7 import IsaacGymEnvNode_7
    from nodes.ppotrainernode_6 import PPOTrainerNode_6
    from nodes.ornode_2 import ORNode_2
    from nodes.ppoagentnode_3 import PPOAgentNode_3
    from nodes.cartpoleactionnode_11 import CartpoleActionNode_11
    from framework.base import GraphRunner
    
    # Create and wire nodes
    nodes = {
        "6": PPOTrainerNode_6("6"),
        "9": IsaacGymStepNode_9("9"),
        "2": ORNode_2("2"),
        "3": PPOAgentNode_3("3"),
        "11": CartpoleActionNode_11("11"),
        "7": IsaacGymEnvNode_7("7")
    }
    
    runner = GraphRunner()
    for node in nodes.values():
        runner.add_node(node)
    
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
    
    # Run
    print("\nStarting execution...")
    start_time = time.time()
    
    try:
        await asyncio.wait_for(runner.run(), timeout=duration_seconds)
    except asyncio.TimeoutError:
        pass
    
    actual_duration = time.time() - start_time
    print(f"\nExecution completed. Duration: {actual_duration:.1f} seconds")
    
    # Print statistical report
    profiler.print_report()
    
    # Also run cProfile on just the PPO agent
    await profile_with_cprofile()
    
    # Show line profiler instructions
    await profile_with_line_profiler()

if __name__ == "__main__":
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
        asyncio.run(run_statistical_profiling(duration_seconds=20))